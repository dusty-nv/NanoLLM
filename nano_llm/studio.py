#!/usr/bin/env python3
from pprint import pprint

from nano_llm import Agent, Plugin
from nano_llm.web import WebServer
from nano_llm.plugins import ChatSession, VideoSource, VideoOutput, UserPrompt
from nano_llm.utils import ArgParser, inspect_function



class DynamicPlugin(Plugin):
    Types = {}
    TypeInfo = {}

    def __new__(cls, plugin, *args, **kwargs):
        if isinstance(plugin, str):
            if plugin in cls.Types:
                plugin = cls.Types[plugin]
            else:
                raise ValueError(f"unregistered plugin type: {plugin}")
        return plugin(*args, **kwargs)
        
    @classmethod
    def register(cls, plugin, **kwargs):
        info = {
            'name': plugin.__name__, # __class__ type
            'flags': kwargs,
            'init': inspect_function(plugin.__init__),
        }

        if hasattr(plugin, 'apply_config'):
            info['config'] = inspect_function(plugin.apply_config)

        cls.TypeInfo[info['name']] = info
        cls.Types[info['name']] = plugin

   
DynamicPlugin.register(ChatSession)   
DynamicPlugin.register(UserPrompt)  
DynamicPlugin.register(VideoSource)
DynamicPlugin.register(VideoOutput)

pprint(DynamicPlugin.TypeInfo)
 
 
class DynamicAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.webserver = WebServer(title='Agent Studio', msg_callback=self.on_websocket, **kwargs)
        self.plugins = []
    
    def add(self, plugin, **kwargs):
        if not isinstance(plugin, Plugin):
            plugin = DynamicPlugin(plugin, **kwargs)
            
        plugin_name = plugin.name
        plugin_idx = 1
        
        while any([x.name == plugin.name for x in self.plugins]):
            plugin.name = f"{plugin_name}_{plugin_idx}"
            plugin_idx += 1
             
        plugin.start()
        
        self.plugins.append(plugin)
        self.webserver.send_message({
            'plugin_added': [DynamicPlugin.state_dict(plugin)],
        })
        
        return plugin
    
    def find(self, name):
        for plugin in self.plugins:
            if name == plugin.name:
                return plugin
   
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs): 
        print('on_websocket()', msg)
        if msg_type == WebServer.MESSAGE_JSON:
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    init_msg = {
                        'plugin_types': DynamicPlugin.TypeInfo,
                    }
                    
                    if self.plugins:
                        init_msg['plugin_added'] = [plugin.state_dict() for plugin in self.plugins]
                        
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


if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--index", "--page", type=str, default="studio.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    agent = DynamicAgent(**vars(args)).run()

    
