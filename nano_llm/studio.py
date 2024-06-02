#!/usr/bin/env python3
from pprint import pprint

from nano_llm import Agent, Plugin
from nano_llm.web import WebServer
from nano_llm.plugins import ChatSession, VideoSource, VideoOutput
from nano_llm.utils import ArgParser, inspect_function



class DynamicPlugin(Plugin):
    types = {}
    type_info = {}
    
    def __new__(cls, plugin, *args, **kwargs):
        if isinstance(plugin, str):
            if plugin in cls.types:
                plugin = cls.types[plugin]
            else:
                raise ValueError(f"unregistered plugin type: {plugin}")
        return plugin(*args, **kwargs)
        
    @classmethod
    def register(cls, plugin, **kwargs):
        info = {
            'name': plugin.__name__,
            'flags': kwargs,
            'init': inspect_function(plugin.__init__),
        }

        if hasattr(plugin, 'apply_config'):
            info['config'] = inspect_function(plugin.apply_config)

        cls.type_info[info['name']] = info
        cls.types[info['name']] = plugin
    
    @classmethod
    def state_dict(cls, plugin):
        links = []
        
        for c, output_channel in enumerate(plugin.outputs):
            for output in output_channel:
                links.append({
                    'to': output.__class__.__name__,
                    'input': 0,
                    'output': c
                 })
        
        flags = cls.type_info[plugin.__class__.__name__]['flags']
              
        return {
            'name': plugin.__class__.__name__,
            'type': plugin.__class__.__name__,
            'inputs': [] if 'source' in flags else [0],
            'outputs': [] if 'sink' in flags else [x for x in range(plugin.output_channels)],
            'links': links,
        }   
   
DynamicPlugin.register(ChatSession)     
DynamicPlugin.register(VideoSource, source=True)
DynamicPlugin.register(VideoOutput, sink=True)

pprint(DynamicPlugin.type_info)
 
 
class DynamicAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.webserver = WebServer(title='Agent Studio', msg_callback=self.on_websocket, **kwargs)
        self.plugins = []
    
    def add(self, plugin, **kwargs):
        if not isinstance(plugin, Plugin):
            plugin = DynamicPlugin(plugin, **kwargs)
        plugin.start()
        self.plugins.append(plugin)
        self.webserver.send_message({
            'plugin_added': [DynamicPlugin.state_dict(plugin)],
        })
        return plugin
    
    def find(self, name):
        for plugin in self.plugins:
            if name == plugin.__class__.__name__:
                return plugin
   
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs): 
        print('on_websocket()', msg)
        if msg_type == WebServer.MESSAGE_JSON:
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    init_msg = {
                        'plugin_types': DynamicPlugin.type_info,
                    }
                    
                    if self.plugins:
                        init_msg['plugin_added'] = [DynamicPlugin.state_dict(plugin) for plugin in self.plugins]
                        
                    self.webserver.send_message(init_msg)
                    
            if 'init_plugin' in msg:
                self.add(msg['init_plugin']['type'], **msg['init_plugin']['args'])
                
            if 'config_plugin' in msg:
                self.find(msg['config_plugin']['type']).apply_config(**msg['config_plugin']['args'])
                
            if 'add_connection' in msg:
                conn = msg['add_connection']
                self.find(conn['output']['name']).add(self.find(conn['input']['name']), conn['output']['channel'])
             
            if 'get_state_dict' in msg:
                plugin_name = msg['get_state_dict']
                self.webserver.send_message({
                    'state_dict': {
                        plugin_name: self.find(plugin_name).state_dict()
                    }
                })
                                       
    def start(self):
        self.webserver.start()
        return self
        
    def run(self, timeout=None):
        self.start()
        self.webserver.web_thread.join(timeout)
        return self           




'''
def get_plugin_types():
    plugins = [VideoSource]
    types = {}
    
    for cls in plugins:
        desc = inspect_function(cls.__init__)
        desc['name'] = cls.__name__
        
        for param_name, param in desc['parameters'].items():
            param['display_name'] = param_name.replace('_', ' ').title()
            
        types[desc['name']] = desc
        
    pprint(types)
    return types
'''    
    


if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--index", "--page", type=str, default="studio.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    agent = DynamicAgent(**vars(args)).run()

    
