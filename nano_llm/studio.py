#!/usr/bin/env python3
import logging
import pprint

from nano_llm import Agent, Plugin
from nano_llm.web import WebServer

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
                
        for key, value in kwargs.items():
            if value == 'false':
                kwargs[key] = False
            elif value == 'true':
                kwargs[key] = True
                
        return plugin(*args, **kwargs)
        
    @classmethod
    def register(cls, plugin, **kwargs):
        info = {
            'name': plugin.__name__, # __class__ type
            'flags': kwargs,
            'init': inspect_function(plugin.__init__),
        }

        cls.TypeInfo[info['name']] = info
        cls.Types[info['name']] = plugin

    @classmethod
    def register_all(cls):
        from nano_llm.plugins import (
            ChatSession, VideoSource, VideoOutput, 
            UserPrompt, VADFilter, PrintStream,
            AudioInputDevice, AudioOutputDevice, AudioRecorder
        )
        
        from nano_llm.plugins.audio.riva_asr import RivaASR
        from nano_llm.plugins.audio.riva_tts import RivaTTS
        from nano_llm.plugins.audio.piper_tts import PiperTTS
        from nano_llm.plugins.audio.whisper_asr import WhisperASR
        from nano_llm.plugins.audio.web_audio import WebAudioIn, WebAudioOut
        
        DynamicPlugin.register(ChatSession)   
        DynamicPlugin.register(UserPrompt)  
        DynamicPlugin.register(PrintStream)
        DynamicPlugin.register(VideoSource)
        DynamicPlugin.register(VideoOutput)
        DynamicPlugin.register(VADFilter)
        DynamicPlugin.register(WhisperASR)
        DynamicPlugin.register(RivaASR)
        DynamicPlugin.register(RivaTTS)
        DynamicPlugin.register(PiperTTS)
        DynamicPlugin.register(AudioInputDevice)
        DynamicPlugin.register(AudioOutputDevice)
        DynamicPlugin.register(AudioRecorder)
        DynamicPlugin.register(WebAudioIn)
        DynamicPlugin.register(WebAudioOut)
        
        logging.info(f"Registered dynamic plugin types:\n\n{pprint.pformat(DynamicPlugin.TypeInfo, indent=2)}")

 
class DynamicAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        DynamicPlugin.register_all()
        
        self.plugins = []
        self.webserver = WebServer(title='Agent Studio', msg_callback=self.on_websocket, **kwargs)

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
            'plugin_added': [plugin.state_dict(config=True)],
        })
        
        return plugin
    
    def find(self, name):
        for plugin in self.plugins:
            if name == plugin.name:
                return plugin
   
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs): 
        if msg_type == WebServer.MESSAGE_JSON:
            print('on_websocket()', msg)
            
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    init_msg = {
                        'plugin_types': DynamicPlugin.TypeInfo,
                    }
                    
                    if self.plugins:
                        init_msg['plugin_added'] = [plugin.state_dict(config=True) for plugin in self.plugins]
                        
                    self.webserver.send_message(init_msg)
                    
            if 'init_plugin' in msg:
                self.add(msg['init_plugin']['name'], **msg['init_plugin']['args'])
                
            if 'config_plugin' in msg:
                self.find(msg['config_plugin']['name']).set_parameters(**msg['config_plugin']['args'])
                
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

    
