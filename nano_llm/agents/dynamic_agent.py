#!/usr/bin/env python3
import os
import json
import yaml
import glob
import time
import pprint
import logging
import threading
import traceback

import nano_llm

from nano_llm import Agent, Plugin
from nano_llm.web import WebServer
from nano_llm.plugins import DynamicPlugin, TerminalPlugin, Tegrastats


class DynamicAgent(Agent):
    """
    Agent that is dynamically configured at runtime by adding/removing plugins and their connections.
    It also provides a websocket API that automatically routes messages to plugin functions and attributes.
    """
    def __init__(self, load=None, preset_dir='/data/nano_llm/presets', **kwargs):
        super().__init__(**kwargs)
        
        DynamicPlugin.register_all()

        self.plugins = []
        self.preset_dir = preset_dir
        
        os.makedirs(preset_dir, exist_ok=True)
        
        if not kwargs.get('web_trace', False):
            self.tegrastats = Tegrastats()
            self.terminal = None #TerminalPlugin()
        else:
            self.tegrastats = None
            self.terminal = None
            
        self.reset()
        
        self.webserver = WebServer(
            title='Agent Studio', 
            msg_callback=self.on_websocket, 
            mounts={
                #'/data/uploads': '/uploads',
                '/data': '/data',
            },
            **kwargs
        )

        if load:
            self.load(load)
            
    def add_plugin(self, type='', wait=False, start=True, state_dict={}, layout_node=None, **kwargs):          
        if not wait:
            threading.Thread(target=self.add_plugin, kwargs={'type': type, 'wait': True, 'state_dict': state_dict, 'layout_node': layout_node, **kwargs}).run()
            return
            
        load_begin = time.perf_counter()
        init_kwargs = {**state_dict.get('init_kwargs', {}), **kwargs}
        
        # create the desired plugin type with the specified init args
        plugin = DynamicPlugin(type, **init_kwargs)
        
        # rename the plugin if one already exists by that name
        plugin_name = plugin.name
        plugin_idx = 1
        
        while any([x.name == plugin.name for x in self.plugins]):
            plugin.name = f"{plugin_name}_{plugin_idx}"
            plugin_idx += 1
        
        # update with runtime params and start threads
        self.plugins.append(plugin)
        plugin.set_parameters(**state_dict)  
         
        if layout_node:
            plugin.layout_node['x'] = plugin.layout_node.get('x', 0) + layout_node['x']
            plugin.layout_node['y'] = plugin.layout_node.get('y', 0) + layout_node['y']
            logging.debug(f"offset {plugin.name} node layout by {layout_node} to {plugin.layout_node}")
            
        if start:  
            plugin.start()
            
        self.webserver.send_message({'plugin_added': [plugin.state_dict(config=True)]})

        load_time = time.perf_counter() - load_begin
        
        if load_time > 1:  # don't show alert if model was cached
            args_str = pprint.pformat(init_kwargs, indent=2, width=80).replace('\n', '<br/>')
            self.webserver.send_alert(f"Loaded {plugin.name} in {load_time:.1f} seconds<br/>&nbsp;&nbsp;{args_str}", level='success')
        
        return plugin
        
    def config_plugin(self, name='', **kwargs):
        plugin = self.find_plugin(name)
            
        if plugin is not None:
            plugin.set_parameters(**kwargs)
        else:
            self.global_states[name] = kwargs
                        
    def find_plugin(self, name):
        if not name:
            return
            
        for plugin in self.plugins:
            if name == plugin.name:
                return plugin

    def remove_plugin(self, name):
        if isinstance(name, Plugin):
            plugin = name
        else:
            plugin = self.find_plugin(name);
        
        if plugin is None:
            return
        
        for output_plugin in self.plugins:
            for output_channel in output_plugin.outputs:
                for output in output_channel.copy():
                    if plugin == output:
                        output_channel.remove(plugin)
         
        self.webserver.send_message({'plugin_removed': plugin.name});              
        self.plugins.remove(plugin)
        
        plugin.destroy()
        del plugin
     
    def reset(self, plugins=True, globals=True):
        if plugins:
            for plugin in self.plugins.copy():
                self.remove_plugin(plugin)
        
        if globals:    
            self.global_states = {
                'GraphEditor': {'layout_grid': {'x': 0, 'y': 0, 'w': 8, 'h': 10}},
            }
            
            if self.terminal is not None:
                self.global_states['TerminalPlugin'] = {'layout_grid': {'x': 0, 'y': 4, 'w': 8, 'h': 6}}
        
        logging.debug(f"{self.__class__.__name__} issued reset (plugins={plugins}, globals={globals})")
    
    def clear_cache(self):
        DynamicPlugin.clear_cache()
            
    def add_connection(self, input='', input_channel=0, output='', output_channel=0):
        input_plugin = self.find_plugin(input)
        output_plugin = self.find_plugin(output)

        if input_plugin is None or output_plugin is None:
            return
        
        output_plugin.connect(input_plugin, channel=output_channel)

    def remove_connection(self, input='', input_channel=0, output='', output_channel=0):
        input_plugin = self.find_plugin(input)
        output_plugin = self.find_plugin(output)

        if input_plugin is None or output_plugin is None:
            return
        
        output_plugin.disconnect(input_plugin, channel=output_channel)
 
    def get_state_dict(self, name=''):
        if not name or name == 'agent':
            state_dict = {
                'version': nano_llm.__version__,
                'globals': self.global_states,
                'plugins': [],
            }
            
            for plugin in self.plugins:
                plugin_state = plugin.state_dict(connections=True, hidden=True)

                if hasattr(plugin, 'init_kwargs'): 
                    plugin_state['init_kwargs'] = plugin.init_kwargs
    
                state_dict['plugins'].append(plugin_state)
            
            return state_dict    

        # get the state of a specific plugin
        plugin = self.find_plugin(name)
        
        if plugin is not None:
            state_dict = plugin.state_dict()
        else:
            state_dict = self.global_states.get(name)
           
        if state_dict is not None: 
            return {'state_dict': {name: state_dict}}
    
    def set_state_dict(self, state_dict={}, name='', wait=None, reset=False, layout_node=None, **kwargs):
        if wait is None:
            wait = True if name else False

        if not wait:
            threading.Thread(
                target=self.set_state_dict, 
                kwargs={**kwargs, 'name': name, 'state_dict': state_dict, 'layout_node': layout_node, 'wait': True}
            ).run()
            return
        
        #print('SET STATE DICT', state_dict, name, wait, reset, kwargs)
           
        state_dict.update(kwargs)
        
        if name and name != 'agent':
            self.config_plugin(name, **state_dict)
            return
        
        if reset:
            self.reset()
                
        self.global_states.update(state_dict.get('globals', {}))
        
        plugins = state_dict.get('plugins', [])
        instances = []
        
        layout_node_origin = {'x': 999999, 'y': 999999}

        for plugin in plugins:
            layout_node_origin['x'] = min(layout_node_origin['x'], plugin.get('layout_node', {}).get('x', 0))
            layout_node_origin['y'] = min(layout_node_origin['y'], plugin.get('layout_node', {}).get('y', 0))
         
        if layout_node:
            layout_node['x'] -= layout_node_origin['x']
            layout_node['y'] -= layout_node_origin['y']
            
        logging.debug(f"{name} agent node layout origin:  {layout_node_origin}")
        insert = (len(self.plugins) > 0)
        
        for plugin in plugins:
            try:
                instance = self.add_plugin(
                    type=plugin['type'], 
                    wait=True, start=False, 
                    state_dict=plugin, 
                    layout_node=layout_node
                )
                if insert:
                    plugin['original_name'] = plugin['name']
                    plugin['name'] = instance.name
                else:
                    instance.name = plugin['name']

                instances.append(instance)
            except Exception as error:
                logging.error(f"Exception occurred during adding plugin:\n{pprint.pformat(plugin, indent=2)}\n\n{traceback.format_exc()}")
        
        for plugin in plugins:
            output_plugin = self.find_plugin(plugin['name'])
            
            if output_plugin is None:
                logging.warning(f"could not find plugin '{plugin['name']}' when loading config (skipping connections)")
                continue
            
            for connection in plugin['connections']:
                for p in plugins:
                    if 'original_name' in p:
                        connection['to'] = connection['to'].replace(p['original_name'], p['name'])
                    
                input_plugin = self.find_plugin(connection['to'])
                
                if input_plugin is None:
                    logging.warning(f"could not find plugin '{connection['to']}' when loading config (skipping connections)")
                    continue  
                    
                output_plugin.connect(input_plugin, channel=connection['output'])
                
                self.webserver.send_message({
                    'plugin_connected': {
                        'from': output_plugin.name,
                        **connection
                    }
                })
    
        for plugin in instances:
            plugin.start()
            
    def save(self, path):
        if not path:
            return
            
        ext = os.path.splitext(path)[1].lower()
        
        if not ext:
            ext = '.json'
            path = path + ext
            
        if not os.path.dirname(path):
            path = os.path.join(self.preset_dir, path)

        state_dict = self.get_state_dict()
        
        with open(path, 'w') as f:
            if ext == '.json':
                json.dump(state_dict, f, indent=2)
            elif ext == '.yaml' or ext == '.yml':
                yaml.safe_dump(state_dict, f, indent=2)
            else:
                raise ValueError(f"supported extensions are .json, .yml, .yaml (was {ext})")
                
        self.webserver.send_alert(f"saved preset to {path}", level='success')
        self.webserver.send_message({'presets': self.list_presets()})
    
    def load(self, path, reset=True, layout_node=None):
        if not path:
            return

        logging.info(f"loading {path}   (reset={reset}, layout_node={layout_node})")
        
        found_path = None
        previous_nodes = len(self.plugins)
        possible_files = [
            path,
            path + '.json',
            path + '.yaml',
            path + '.yml',
        ]
        
        possible_files += [os.path.join(self.preset_dir, x) for x in possible_files]

        for possible_file in possible_files:
            if os.path.isfile(possible_file):
                found_path = possible_file
                break
                
        if not found_path:
           self.webserver.send_alert(f"Couldn't find preset '{path}' on server", level='error')
           return

        try:   
            ext = os.path.splitext(found_path)[1].lower()
            
            with open(found_path, 'r') as f:
                if ext == '.json':
                    state_dict = json.load(f)
                elif ext == '.yaml' or ext == '.yml':
                    state_dict = yaml.safe_load(f)
                else:
                    raise ValueError(f"supported extensions are .json, .yml, .yaml (was {ext})")

            self.set_state_dict(state_dict, reset=reset, layout_node=layout_node)
        except Exception as error:
            self.webserver.send_alert(f"Exception occurred loading preset '{path}'\n\n{traceback.format_exc()}", level='error')
        else:
            self.webserver.send_alert(f"Loaded preset {path} ({len(self.plugins)-previous_nodes} nodes)", level='success')

    def insert(self, path=None, layout_node=None):
        return self.load(path, reset=False, layout_node=layout_node)
        
    def list_presets(self, preset_dir=None, remove_extensions=True):
        if not preset_dir:
            preset_dir = self.preset_dir

        files = []
        extensions = ['.json', '.yaml', '.yml']
        
        for ext in extensions:
            files.extend([
                x.replace(preset_dir + ('/' if preset_dir[-1] != '/' else ''), '')
                for x in sorted(glob.glob(os.path.join(preset_dir, f'**{ext}')))
            ])
            
        if remove_extensions:
            for ext in extensions:
                for i, file in enumerate(files):
                    files[i] = file.replace(ext, '')
                    
        return files
              
    def on_client_connected(self):
        init_msg = {
            'modules': DynamicPlugin.modules(),
            'plugin_types': DynamicPlugin.TypeInfo,
            'presets': self.list_presets(),
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
                logging.debug(f"websocket | calling {obj.__class__.__name__}.{attr}({msg})")
                
                if isinstance(msg, dict):
                    response = func(**msg)
                else:
                    response = func(msg)
                    
                if isinstance(response, dict): #if response is not None:
                    self.webserver.send_message(response)
            else:
               logging.debug(f"websocket | setting {obj.__class__.__name__}.{attr} = {msg}")
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
        if self.terminal is not None:
            self.terminal.start()
            
        if self.tegrastats is not None:
            self.tegrastats.start()
            
        self.webserver.start()
        return self
        
    def run(self, timeout=None):
        self.start()
        self.webserver.web_thread.join(timeout)
        return self           


if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--load", type=str, default=None, help="load an agent from .json or .yaml")
    parser.add_argument("--agent-dir", type=str, default="/data/nano_llm/agents", help="change the agent load/save directory")
    
    parser.add_argument("--index", "--page", type=str, default="studio.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    agent = DynamicAgent(**vars(args)).run()

    
