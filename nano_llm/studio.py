#!/usr/bin/env python3
import time
import pprint
import logging

from nano_llm import Agent, Plugin
from nano_llm.web import WebServer
from nano_llm.plugins import Tegrastats
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
        
        if hasattr(plugin, 'type_hints'):
            type_hints = plugin.type_hints()
            for key, value in type_hints.items():
                if key in info['init']['parameters']:
                    info['init']['parameters'][key].update(value)

        cls.TypeInfo[info['name']] = info
        cls.Types[info['name']] = plugin

    @classmethod
    def register_all(cls):
        from nano_llm.plugins import (
            ChatSession, VideoSource, VideoOutput,
            UserPrompt, AutoPrompt, VADFilter,
            TextStream, TextOverlay, RateLimit,
            AudioInputDevice, AudioOutputDevice, AudioRecorder,
        )
        
        from nano_llm.plugins.audio.riva_asr import RivaASR
        from nano_llm.plugins.audio.riva_tts import RivaTTS
        from nano_llm.plugins.audio.piper_tts import PiperTTS
        from nano_llm.plugins.audio.whisper_asr import WhisperASR
        from nano_llm.plugins.audio.web_audio import WebAudioIn, WebAudioOut
        
        DynamicPlugin.register(ChatSession)   
        DynamicPlugin.register(UserPrompt)
        DynamicPlugin.register(AutoPrompt)   
        DynamicPlugin.register(TextStream)
        DynamicPlugin.register(TextOverlay)
        DynamicPlugin.register(VideoSource)
        DynamicPlugin.register(VideoOutput)
        DynamicPlugin.register(RateLimit)
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
        self.global_states = {'GraphEditor': {'web_grid': {'x': 0, 'y': 0, 'w': 8, 'h': 14}}}
        
        self.tegrastats = Tegrastats()
        self.webserver = WebServer(title='Agent Studio', msg_callback=self.on_websocket, **kwargs)

    def add(self, plugin, **kwargs):
        load_begin = time.perf_counter()
        
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
        
        load_time = time.perf_counter() - load_begin
        
        if load_time > 1:  # don't show if model was cached
            args_str = pprint.pformat(kwargs, indent=2, width=80).replace('\n', '<br/>')
            self.webserver.send_alert(f"Created {plugin.name} in {load_time:.1f} seconds<br/>&nbsp;&nbsp;{args_str}", level='success')
        
        return plugin
    
    def find(self, name):
        for plugin in self.plugins:
            if name == plugin.name:
                return plugin
   
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs): 
        if msg_type != WebServer.MESSAGE_JSON:
            return
            
        print('on_websocket()', msg)
        
        if 'client_state' in msg:
            if msg['client_state'] == 'connected':
                init_msg = {
                    'plugin_types': DynamicPlugin.TypeInfo,
                }
                
                init_msg['plugin_added'] = [{'name': name, 'global': True, **state} for name, state in self.global_states.items()]
                init_msg['plugin_added'].extend([plugin.state_dict(config=True) for plugin in self.plugins])
                    
                self.webserver.send_message(init_msg)
                
        if 'init_plugin' in msg:
            self.add(msg['init_plugin']['name'], **msg['init_plugin']['args'])
            
        if 'config_plugin' in msg:
            config = msg['config_plugin']
            plugin = self.find(config['name'])
            
            if plugin is not None:
                plugin.set_parameters(**config['args'])
            else:
                self.global_states[config['name']] = config['args']
         
        if 'remove_plugin' in msg:
            name = msg['remove_plugin'];
            plugin = self.find(name);
            self.plugins.remove(plugin)
            logging.info(f"removed plugin {plugin.name}")
            del plugin
   
        if 'add_connection' in msg:
            conn = msg['add_connection']
            self.find(conn['output']['name']).add(self.find(conn['input']['name']), conn['output']['channel'])
        
        if 'remove_connection' in msg:
            conn = msg['remove_connection']
            self.find(conn['output']['name']).outputs[conn['output']['channel']].remove(self.find(conn['input']['name']))
         
        if 'get_state_dict' in msg:
            plugin_name = msg['get_state_dict']
            plugin = self.find(plugin_name)
            self.webserver.send_message({
                'state_dict': {
                    plugin_name: plugin.state_dict() if plugin else self.global_states.get(plugin_name)
                }
            })
                                   
    def start(self):
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

    
