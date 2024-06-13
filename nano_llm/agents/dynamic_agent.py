#!/usr/bin/env python3
import time
import pprint
import logging
import threading

from nano_llm import Agent
from nano_llm.web import WebServer
from nano_llm.plugins import DynamicPlugin, TerminalPlugin, Tegrastats


class DynamicAgent(Agent):
    """
    Agent that is dynamically configured at runtime by adding/removing plugins and their connections.
    It also provides a websocket API that automatically routes messages to plugin functions and attributes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        DynamicPlugin.register_all()
        
        self.plugins = []
        self.tegrastats = Tegrastats()
        self.terminal = TerminalPlugin()
        
        self.global_states = {
            'GraphEditor': {'web_grid': {'x': 0, 'y': 0, 'w': 8, 'h': 14}},
            'TerminalPlugin': {'web_grid': {'x': 0, 'y': 4, 'w': 8, 'h': 6}},
        }

        self.webserver = WebServer(
            title='Agent Studio', 
            msg_callback=self.on_websocket, 
            mounts={
                #'/data/uploads': '/uploads',
                '/data': '/data',
            },
            **kwargs
        )

    def add_plugin(self, type='', **kwargs):
        if not kwargs.get('__thread__'):
            threading.Thread(target=self.add_plugin, kwargs={**kwargs, 'type': type, '__thread__': True}).run()
            return
            
        load_begin = time.perf_counter()
        plugin = DynamicPlugin(type, **kwargs)
            
        plugin_name = plugin.name
        plugin_idx = 1
        
        while any([x.name == plugin.name for x in self.plugins]):
            plugin.name = f"{plugin_name}_{plugin_idx}"
            plugin_idx += 1
        
        self.plugins.append(plugin)     
        plugin.start()
        self.webserver.send_message({'plugin_added': [plugin.state_dict(config=True)]})

        load_time = time.perf_counter() - load_begin
        
        if load_time > 1:  # don't show alert if model was cached
            args_str = pprint.pformat(kwargs, indent=2, width=80).replace('\n', '<br/>')
            self.webserver.send_alert(f"Created {plugin.name} in {load_time:.1f} seconds<br/>&nbsp;&nbsp;{args_str}", level='success')
    
    def config_plugin(self, name='', **kwargs):
        plugin = self.find_plugin(name)
            
        if plugin is not None:
            plugin.set_parameters(**kwargs)
        else:
            self.global_states[name] = kwargs

    def remove_plugin(self, name):
        plugin = self.find_plugin(name);
        
        if plugin is None:
            return
            
        self.plugins.remove(plugin)
        plugin.stop()
        del plugin
                        
    def find_plugin(self, name):
        if not name:
            return
            
        for plugin in self.plugins:
            if name == plugin.name:
                return plugin
   
    def add_connection(self, input='', input_channel=0, output='', output_channel=0):
        input_plugin = self.find_plugin(input)
        output_plugin = self.find_plugin(output)

        if input_plugin is None or output_plugin is None:
            return
        
        output_plugin.add(input_plugin, channel=output_channel)

    def remove_connection(self, input='', input_channel=0, output='', output_channel=0):
        input_plugin = self.find_plugin(input)
        output_plugin = self.find_plugin(output)

        if input_plugin is None or output_plugin is None:
            return
        
        output_plugin.outputs[output_channel].remove(input_plugin)

    def get_state_dict(self, name):
        plugin = self.find_plugin(name)
        
        if plugin is not None:
            state_dict = plugin.state_dict()
        else:
            state_dict = self.global_states.get(name)
           
        if state_dict is not None: 
            return {'state_dict': {name: state_dict}}
    
    def on_client_connected(self):
        init_msg = {
            'modules': DynamicPlugin.modules(),
            'plugin_types': DynamicPlugin.TypeInfo,
        }
        
        init_msg['plugin_added'] = [{'name': name, 'global': True, **state} for name, state in self.global_states.items()]
        init_msg['plugin_added'].extend([plugin.state_dict(config=True) for plugin in self.plugins])
        
        return init_msg 
            
    def on_client_state(self, state):
        if state == 'connected':
            return self.on_client_connected()
                
    def on_websocket(self, message, msg_type=0, metadata='', **kwargs): 
        if msg_type != WebServer.MESSAGE_JSON:
            return

        def invoke_handler(obj, attr, msg):
            if obj is None:
                return False
             
            if not hasattr(obj, attr):
                attr = 'on_' + attr
                if not hasattr(obj, attr):
                    return False
                
            func = getattr(obj, attr)

            if callable(func):
                logging.debug(f"websocket call:  {obj.__class__.__name__}.{attr}({msg})")
                
                if isinstance(msg, dict):
                    response = func(**msg)
                else:
                    response = func(msg)
                    
                if isinstance(response, dict): #if response is not None:
                    self.webserver.send_message(response)
            else:
               logging.debug(f"websocket setting:  {obj.__class__.__name__}.{attr} = {msg}")
               setattr(obj, attr, msg)
    
            return True
             
        def on_message(obj, message):
            if not isinstance(message, dict):
                logging.warning(f"recieved invalid websocket {plugin.name} message (expected dict, was {type(msg)}  {msg}")
                return

            for key, msg in message.items():
                if invoke_handler(obj, key, msg):
                    continue

                plugin = self.find_plugin(key)
            
                if plugin is not None:
                    on_message(plugin, msg)
                else:
                    logging.warning(f"recieved unrecognized websocket json message '{key}'\n{pprint.pformat(msg, indent=2)}")

        on_message(self, message)

        '''
        if 'init_plugin' in msg:
            self.add(msg['init_plugin']['name'], **msg['init_plugin']['args'])
            
        if 'config_plugin' in msg:
            config = msg['config_plugin']
            plugin = self.find_plugin(config['name'])
            
            if plugin is not None:
                plugin.set_parameters(**config['args'])
            else:
                self.global_states[config['name']] = config['args']
         
        if 'remove_plugin' in msg:
            name = msg['remove_plugin'];
            plugin = self.find_plugin(name);
            self.plugins.remove(plugin)
            logging.info(f"removed plugin {plugin.name}")
            plugin.stop()
            del plugin
   
        if 'add_connection' in msg:
            conn = msg['add_connection']
            self.find_plugin(conn['output']['name']).add(self.find_plugin(conn['input']['name']), conn['output']['channel'])
        
        if 'remove_connection' in msg:
            conn = msg['remove_connection']
            self.find_plugin(conn['output']['name']).outputs[conn['output']['channel']].remove(self.find_plugin(conn['input']['name']))
         
        if 'get_state_dict' in msg:
            plugin_name = msg['get_state_dict']
            plugin = self.find_plugin(plugin_name)
            self.webserver.send_message({
                'state_dict': {
                    plugin_name: plugin.state_dict() if plugin else self.global_states.get(plugin_name)
                }
            })
        '''
                                  
    def start(self):
        self.terminal.start()
        self.tegrastats.start()
        self.webserver.start()
        return self
        
    def run(self, timeout=None):
        self.start()
        self.webserver.web_thread.join(timeout)
        return self           


if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--index", "--page", type=str, default="studio.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    agent = DynamicAgent(**vars(args)).run()

    
