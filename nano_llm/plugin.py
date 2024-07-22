#!/usr/bin/env python3
import time
import queue
import threading
import logging
import traceback

from nano_llm.web import WebServer
from nano_llm.utils import AttributeDict, inspect_function, json_type, python_type


class Plugin(threading.Thread):
    """
    Base class for plugins that process incoming/outgoing data from connections
    with other plugins, forming a pipeline or graph.  Plugins can run either
    single-threaded or in an independent thread that processes data out of a queue.

    Frequent categories of plugins:
    
      * sources:  text prompts, images/video
      * process:  LLM queries, RAG, dynamic LLM calls, image post-processors
      * outputs:  print to stdout, save images/video
      
    Inherited plugins should implement the :func:`process` function to handle incoming data.

    """
    Instances = []  #: Global list of plugin instances
        
    def __init__(self, name=None, title=None, inputs=1, outputs=1,
                 relay=False, drop_inputs=False, threaded=True, **kwargs):
        """
        Base initializer for Plugin implementations.
        
        Args:
            name (str):  specify the name of this plugin instance (otherwise initialized from class name)
            output_channels (int): the number of sets of output connections the plugin has
            relay (bool): if true, will relay any inputs as outputs after processing
            drop_inputs (bool): if true, only the most recent input in the queue will be used
            threaded (bool): if true, will spawn independent thread for processing the queue.
        """
        super().__init__(daemon=True)

        self.name = name if name else self.__class__.__name__
        self.title = title
        self.relay = relay
        self.drop_inputs = drop_inputs
        self.threaded = threaded
        self.stop_flag = False
        self.interrupted = False
        self.processing = False
        self.parameters = {}
        self.tools = {}
        
        inputs = kwargs.get('input_channels', inputs)
        outputs = kwargs.get('output_channels', outputs)
        
        if isinstance(inputs, str):
            inputs = [inputs]
            
        if isinstance(inputs, list):
            self.input_names = inputs
            inputs = len(inputs)
        else:
            self.input_names = ['0']

        if isinstance(outputs, str):
            outputs = [outputs]
            
        if isinstance(outputs, list):
            self.output_names = outputs
            outputs = len(outputs)
        elif isinstance(outputs, int):
            self.output_names = [str(x) for x in range(outputs)]
        elif outputs is None:
            self.output_names = [] #['0']
            outputs = 0
        else:
            raise TypeError(f"outputs should have been int, str, list[str], or None  (was {type(outputs)})")

        self.outputs = [[] for i in range(outputs)]

        self.add_parameter('layout_grid', type=dict, default={}, hidden=True)
        self.add_parameter('layout_node', type=dict, default={}, hidden=True)
        
        if threaded:
            self.input_queue = queue.Queue()
            self.input_event = threading.Event()

        from nano_llm import BotFunctions
        self.BotFunctions = BotFunctions
        
        Plugin.Instances.append(self)
     
    def __del__(self):
        """
        Stop the plugin from running and unregister it.
        """
        self.destroy()

    def process(self, input, **kwargs):
        """
        Abstract function that plugin instances should implement to process incoming data.
        Don't call this function externally unless ``threaded=False``, because
        otherwise the plugin's internal thread dispatches from the queue.

        Args:
        
          input: input data to process from the previous plugin in the pipeline
          kwargs: optional processing arguments that accompany this data
          
        Returns:
        
          Plugins should return their output data to be sent to downstream plugins.
          You can also call :func:`output()` as opposed to returning it.
        """
        logging.warning(f"plugin {self.name} did not implement process() - dropping input")
    
    def connect(self, plugin, channel=0, **kwargs):
        """
        Connect the output queue from this plugin with the input queue of another plugin,
        so that this plugin sends its output data to the other one.
        
        Args:
        
          plugin (Plugin|callable): either the plugin to link to, or a callback function.
          channel (int) -- the output channel of this plugin to link the other plugin to.
                        
        Returns:

          A reference to this plugin instance (self)
        """
        from nano_llm.plugins import Callback
        
        if not isinstance(plugin, Plugin):
            if not callable(plugin):
                raise TypeError(f"{type(self)}.connect() expects either a Plugin instance or a callable function (was {type(plugin)})")
            plugin = Callback(plugin, **kwargs)
            
        self.outputs[channel].append(plugin)
        
        if isinstance(plugin, Callback):
            logging.debug(f"connected {self.name} to {plugin.function.__name__} on channel={channel}")  # TODO https://stackoverflow.com/a/25959545
        else:
            logging.debug(f"connected {self.name} to {plugin.name} on channel={channel}")
            
        return self
        
    def add(self, plugin, channel=0, **kwargs):
        """
        @deprecated Please use :func:``Plugin.connect``
        """
        return self.connect(plugin, channel=channel, **kwargs)
    
    def __call__(self, input=None, **kwargs):
        """
        Callable () operator alias for the :func:`input()` function.
        This is provided for a more intuitive way of processing data 
        like ``plugin(data)`` instead of ``plugin.input(data)``
        
        Args:
        
          input: input data sent to the plugin's :func:`process()` function.
          kwargs: additional arguments forwarded to the plugin's :func:`process()` function.
          
        Returns:
        
          None if the plugin is threaded, otherwise returns any outputs.
        """
        return self.input(input, **kwargs)
        
    def input(self, input=None, **kwargs):
        """
        Add data to the plugin's processing queue and return immediately,
        or process it now and return the results if ``threaded=False``.
        
        Args:
        
          input: input data sent to the plugin's :func:`process()` function.
          kwargs: additional arguments forwarded to the plugin's :func:`process()` function.
          
        Returns:
        
          None if the plugin is threaded, otherwise returns any outputs.
        """
        if self.threaded:
            #self.start() # thread may not be started if plugin only called from a callback
            if self.drop_inputs:
                configs = []
                while True:
                    try:
                        config_input, config_kwargs = self.input_queue.get(block=False)
                        if config_input is None and len(config_kwargs) > 0:  # still apply config
                            configs.append((config_input, config_kwargs))
                        #else:
                        #    logging.debug(f"{self.name} dropping inputs")
                    except queue.Empty:
                        break
                for config in configs:
                    self.input_queue.put(config)
                    self.input_event.set()

            self.input_queue.put((input,kwargs))
            self.input_event.set()
        else:
            self.dispatch(input, **kwargs)
            
    def output(self, output, channel=0, **kwargs):
        """
        Output data to the next plugin(s) on the specified channel (-1 for all channels)
        """
        #if output is None:
        #    return

        if channel >= 0:
            kwargs.update(dict(sender=self, input_channel=channel))
            for output_plugin in self.outputs[channel]:
                output_plugin.input(output, **kwargs)
        else:
            for output_channel in self.outputs:
                kwargs.update(dict(sender=self, input_channel=output_channel))
                for output_plugin in output_channel:
                    output_plugin.input(output, **kwargs)
                    
        return output
     
    @property
    def num_outputs(self):
        """
        Return the total number of output connections across all channels
        """
        count = 0
        for output_channel in self.outputs:
            count += len(output_channel) 
        return count
        
    def start(self):
        """
        Start threads for all plugins in the graph that have threading enabled.
        """
        if self.threaded:
            if not self.is_alive():
                super().start()
            
        for output_channel in self.outputs:
            for output in output_channel:
                output.start()
                
        return self
     
    def stop(self):
        """
        Flag the plugin to stop processing and exit the run() thread.
        """
        self.stop_flag = True
        logging.debug(f"stopping plugin {self.name} (thread {self.native_id})")
     
    def destroy(self):
        """
        Stop a plugin thread's running, and unregister it from the global instances.
        """ 
        self.stop()
        
        try:
            Plugin.Instances.remove(self)
        except ValueError:
            logging.warning(f"Plugin {getattr(self, 'name', '')} wasn't in global instances list when being deleted")
            
    def run(self):
        """
        Processes the queue forever and automatically run when created with ``threaded=True``
        """
        while not self.stop_flag:
            try:
                self.process_inputs(timeout=0.25)
            except Exception as error:
                logging.error(f"Exception occurred during processing of {self.name}\n\n{traceback.format_exc()}")

        logging.debug(f"{self.name} plugin stopped (thread {self.native_id})")

    def process_inputs(self, timeout=0):
        """
        Waits for inputs up to the timeout in seconds (or ``None`` to wait forever)
        """
        if self.input_queue.empty():
            if timeout is not None and timeout <= 0:
                return
                
            self.input_event.wait(timeout=timeout)
            self.input_event.clear()
            
        while not self.stop_flag:
            try:
                input, kwargs = self.input_queue.get(block=False)
                self.dispatch(input, **kwargs)
            except queue.Empty:
                break
                            
    def dispatch(self, input, **kwargs):
        """
        Invoke the process() function on incoming data
        """
        if self.interrupted:
            #logging.debug(f"{type(self)} resetting interrupted flag to false")
            self.interrupted = False
          
        self.processing = True
        outputs = self.process(input, **kwargs)
        self.processing = False

        if outputs is not None:
            self.output(outputs)
        
        if self.relay:
            self.output(input)
            
        return outputs
   
    def interrupt(self, clear_inputs=True, recursive=True, block=None):
        """
        Interrupt any ongoing/pending processing, and optionally clear the input queue
        along with any downstream queues, and optionally wait for processing of those 
        requests to have finished.
        
        Args:
        
          clear_inputs (bool):  if True, clear any remaining inputs in this plugin's queue.
          recursive (bool):  if True, then any downstream plugins will also be interrupted.
          block (bool):  is true, this function will wait until any ongoing processing has finished.
                         This is done so that any lingering outputs don't cascade downstream in the pipeline.
                         If block is None, it will automatically be set to true if this plugin has outputs.
        """
        #logging.debug(f"interrupting plugin {type(self)}  clear_inputs={clear_inputs} recursive={recursive} block={block}")
        
        if clear_inputs:
            self.clear_inputs()
          
        self.interrupted = True
        
        num_outputs = self.num_outputs
        block_other = block
        
        if block is None and num_outputs > 0:
            block = True
            
        while block and self.processing:
            #logging.debug(f"interrupt waiting for {type(self)} to complete processing")
            time.sleep(0.01) # TODO use an event for this?
        
        if recursive and num_outputs > 0:
            for output_channel in self.outputs:
                for output in output_channel:
                    output.interrupt(clear_inputs=clear_inputs, recursive=recursive, block=block_other)
                    
    def clear_inputs(self):
        """
        Clear the input queue, dropping any data.
        """
        while True:
            try:
                self.input_queue.get(block=False)
            except queue.Empty:
                return         

    def find(self, type):
        """
        Return the plugin with the specified type by searching for it among
        the pipeline graph of inputs and output connections to other plugins.
        """
        if isinstance(self, type):
            return self
            
        for output_channel in self.outputs:
            for output in output_channel:
                if isinstance(output, type):
                    return output
                plugin = output.find(type)
                if plugin is not None:
                    return plugin
            
        return None

    def add_tool(self, function, enabled=True, **kwargs): 
        """
        Register a function that is able to be called by function-calling models.
        
        Args:
          func (callable|str): The function or name of the function.
          doc_templates (dict): Substitute the keys with their values in the help docs.
        """
        if not callable:
            if isinstance(function, str):
                name = function
                function = getattr(self, name, None)
                
                if function is None:
                    raise ValueError(f"class method {self.__class__.__name__}.{name}() does not exist")
                elif not callable(function):
                    raise ValueError(f"{self.__class__.__name__}.{name} was not a callable function")
            else:
                raise ValueError(f"expected either a string or callable function (was {type(function)})")
        else:
            name = function.__name__
                    
        self.tools[name] = AttributeDict(
            name=name, class_name=self.name,
            function=function, enabled=True,
            signature=inspect_function(function),
            openai=inspect_function(function, return_spec='openai'),
            docs=f"`{name}()` - {function.__doc__.strip()}",
        )
        
        return self.tools[name]
        
    def add_parameter(self, attribute: str, name=None, type=None, default=None,
                      read_only=False, hidden=False, help=None, kwarg=None, end=None, **kwargs):
        """
        Make an attribute that is shared in the state_dict and can be accessed/modified by clients.
        These will automatically show up in the studio web UI and can be sync'd over websockets.
        If there is an __init__ parameter by the same name, its help docs will be taken from that.
        """
        if not kwarg:
            kwarg = attribute
            
        init = inspect_function(self.__init__)['parameters'].get(kwarg, {})
        
        if not read_only: #if not hasattr(self, attribute):
            setattr(self, attribute, default)
            
        if name is None:
            name = attribute.replace('_', ' ').title()
        
        if type is None:
            type = init.get('type')
        else:
            type = json_type(type)
            
        param = {
            'display_name': name,
            'type': type,
            'read_only': read_only,
            'hidden': hidden,
        }
        
        if hasattr(self, 'type_hints'):
            for key, value in self.type_hints().items():
                if key == attribute:
                    param.update(value)
                    
        #if kwarg:
        #    param['kwarg'] = kwarg
        
        if not help:
            help = init.get('help')
        
        if help:
            param['help'] = help
            
        if default:
            param['default'] = default
   
        if end:
            param['end'] = end

        param.update(**kwargs)
        self.parameters[attribute] = param
        return param
    
    def add_parameters(self, **kwargs):
        """
        Add parameters from kwargs of the form ``Plugin.add_parameters(x=x, y=y)`` 
        where the keys are the attribute names and the values are the default values.
        """
        for key, value in kwargs.items():
            self.add_parameter(key, default=value)
            
    def set_parameters(self, **kwargs):
        """
        Set a state dict of parameters. Only entries in the dict matching a parameter will be set.
        """
        for attr, value in kwargs.items():
            if attr not in self.parameters:
                if attr != 'name' and attr != 'type' and attr != 'connections':
                    logging.warning(f"attempted to set unknown parameter {self.name}.{attr}={value} (skipping)")
                continue
            logging.debug(f"{self.name} setting parameter '{attr}' to {value}")
            if self.parameters[attr]['type'] == 'boolean' and isinstance(value, str):
                value = value.lower()
                if value == 'true' or value == '1':
                    value = True
                else:
                    value = False
            setattr(self, attr, value)
    
    def reorder_parameters(self):
        """
        Move some parameters to the end for display purposes (if end=True)
        """
        if hasattr(self, '_reordered_parameters') and self._reordered_parameters:
            return
            
        params = self.parameters.copy()
        
        for param_name, param in params.items():
            if 'end' in param:
                del param['end']
                del self.parameters[param_name]
                self.parameters[param_name] = param
        
        self._reordered_parameters = True
                    
    def state_dict(self, config=False, connections=False, hidden=False, **kwargs):
        """
        Return a configuration dict with plugin state that gets shared with clients. 
        Subclasses can reimplement this to add custom state for each type of plugin.
        """
        state = {
            'name': self.name,
            'type': self.__class__.__name__,
        }
        
        if config:
            connections = True
            
        if connections:
            connections = []
            
            for c, output_channel in enumerate(self.outputs):
                for output in output_channel:
                    connections.append({
                        'to': output.name,
                        'input': 0,
                        'output': c
                     })
            
            state['connections'] = connections
   
        if config:
            self.reorder_parameters()
            
            state.update({
                'title': self.title if self.title else self.name,
                'inputs': self.input_names,
                'outputs': self.output_names,
                'parameters': self.parameters,
            })
        
        for attr, param in self.parameters.items():
            if hidden or not param['hidden'] or config:
                state[attr] = getattr(self, attr)
            
        return state
               
    def send_state(self, state_dict=None, **kwargs):
        """
        Send the state dict message over the websocket.
        """
        if not WebServer.Instance or not WebServer.Instance.connected:
            logging.warning(f"plugin {self.name} had no webserver or connected clients to send state_dict")
            return
            
        if state_dict is None:
            state_dict = self.state_dict(**kwargs)
            
        WebServer.Instance.send_message({
            'state_dict': {self.name: state_dict}
        })
     
    def send_stats(self, stats={}, **kwargs):
        """
        Send performance stats over the websocket.
        """
        if not WebServer.Instance or not WebServer.Instance.connected or (not stats and not kwargs):
            return
       
        stats.update(kwargs)

        WebServer.Instance.send_message({
            'stats': {self.name: stats}
        })
           
    def send_alert(self, message, **kwargs):
        """
        Send an alert message to the webserver (see WebServer.send_alert() for kwargs)
        """
        if not WebServer.Instance or not WebServer.Instance.connected:
            return
            
        return WebServer.Instance.send_alert(message, **kwargs)
     
    def send_client_output(self, channel):
        """
        Subscribe clients to recieving plugin output over websockets.
        """
        from nano_llm.plugins import WebClient
        
        for plugin in self.outputs[channel]:
            if isinstance(plugin, WebClient):
                return
               
        web_client = WebClient()
        web_client.start()
         
        self.connect(web_client, channel=channel)

    def apply_substitutions(self, text):
        """
        Perform variable substitution on a string of text by looking up values from other plugins.
        
        References can be scoped to a plugin's name:  "The date is ${Clock.date}"
        Or if left unscoped, find the first plugin with it:  "The date is ${date}"
        Both plugins and attributes are case-insensitive:  "The date is ${DATE}"
        
        These can also refer to getter functions or properties that require no positional arguments,
        and if found the associate function will be called and its return value substituted instead.
        """
        def find_closing_bracket(s : str):
            for i, c in enumerate(s):
                if c == '}':
                    return i
                elif c.isalnum() or c == '_' or c == '.':
                    continue
         
        def read_param(var : str):
            period = var.find('.')
            
            if period > 0 and period < len(var) - 1:
                plugin_name = var[:period].lower()
                plugin_attr = var[period+1:]
            else:
                plugin_name = None
                plugin_attr = var
            
            bot_functions = self.BotFunctions()

            for plugin in [self] + Plugin.Instances + self.BotFunctions():  # resolve unclassed references to this plugin first
                if plugin_name and plugin_name != plugin.name.lower():
                    continue
                    
                if not hasattr(plugin, plugin_attr):
                    plugin_attr_lower = plugin_attr.lower()
                    if hasattr(plugin, plugin_attr_lower):
                        plugin_attr = plugin_attr_lower
                    elif hasattr(plugin, 'function') and getattr(plugin, 'name', '').lower() == plugin_attr_lower:
                        return str(plugin.function())
                    else:
                        continue
                        
                value = getattr(plugin, plugin_attr)
                
                if callable(value):
                    value = value()
                    
                return str(value)
          
            logging.warning(f"{self.name} could not find variable ${{{var}}} for substitution")
            return f"${{{var}}}"
            
        #while True:
        splits = text.split('${')
        string = ''
    
        if len(splits) <= 1:
            return text
        
        for split in splits:
            if not split:
                continue
                
            end = find_closing_bracket(split)
            
            if end is None:
                string = string + split
                continue
                
            var = split[:end].strip()
            string = string + read_param(var) 
        
            if end < len(split)-1:
                string = string + split[end+1:]
                
        return string
        '''                
            if text != string:
                text = string
                continue
            else:        
                return string  
        '''   
