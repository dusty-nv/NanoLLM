#!/usr/bin/env python3
import time
import queue
import threading
import logging
import traceback


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
    
    Args:
    
      output_channels (int): the number of sets of output connections the plugin has
      relay (bool): if true, will relay any inputs as outputs after processing
      drop_inputs (bool): if true, only the most recent input in the queue will be used
      threaded (bool): if true, will spawn independent thread for processing the queue.
    """
    def __init__(self, output_channels=1, relay=False, drop_inputs=False, threaded=True, **kwargs):
        """
        Initialize plugin
        """
        super().__init__(daemon=True)

        self.relay = relay
        self.drop_inputs = drop_inputs
        self.threaded = threaded
        self.interrupted = False
        self.processing = False
        
        self.outputs = [[] for i in range(output_channels)]
        self.output_channels = output_channels
        
        if threaded:
            self.input_queue = queue.Queue()
            self.input_event = threading.Event()

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
        raise NotImplementedError(f"plugin {type(self)} has not implemented process()")
    
    def add(self, plugin, channel=0, **kwargs):
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
                raise TypeError(f"{type(self)}.add() expects either a Plugin instance or a callable function (was {type(plugin)})")
            plugin = Callback(plugin, **kwargs)
            
        self.outputs[channel].append(plugin)
        
        if isinstance(plugin, Callback):
            logging.debug(f"connected {type(self).__name__} to {plugin.function.__name__} on channel={channel}")  # TODO https://stackoverflow.com/a/25959545
        else:
            logging.debug(f"connected {type(self).__name__} to {type(plugin).__name__} on channel={channel}")
            
        return self
    
    def __call__(self, input=None, **kwargs):
        """
        Callable () operator alias for the :func:`input()` function.
        This is provided for a more intuitive way of processing data 
        like ``plugin(data)`` instead of ``plugin.input(data)``
        """
        self.input(input, **kwargs)
        
    def input(self, input=None, **kwargs):
        """
        Add data to the plugin's processing queue (or process it now if ``threaded=False``)
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
        if output is None:
            return
            
        if channel >= 0:
            for output_plugin in self.outputs[channel]:
                output_plugin.input(output, **kwargs)
        else:
            for output_channel in self.outputs:
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
            
    def run(self):
        """
        Processes the queue forever and automatically run when created with ``threaded=True``
        """
        while True:
            try:
                self.input_event.wait()
                self.input_event.clear()
                
                while True:
                    try:
                        input, kwargs = self.input_queue.get(block=False)
                        self.dispatch(input, **kwargs)
                    except queue.Empty:
                        break
            except Exception as error:
                logging.error(f"Exception occurred during processing of {type(self)}\n\n{''.join(traceback.format_exception(error))}")

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

        self.output(outputs)
        
        if self.relay:
            self.output(input)
   
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
    
    '''
    def __getitem__(self, type):
        """
        Subscript indexing [] operator alias for find()
        """
        return self.find(type)
    '''       