#!/usr/bin/env python3
import pprint
import logging

from nano_llm import Plugin
from nano_llm.utils import inspect_function


class DynamicPlugin(Plugin):
    """
    This meta-plugin dynamically instantiates other plugins at runtime with settings
    either from the user or that were saved/loaded.  It overloads the __new__ operator
    and returns the actual plugin instance when you create it, along with the plugin's args.
    
    It also inspects the plugins for type information used for populating the webUI
    along with descriptors for the bot tools & function calling.
    """
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
                
        instance = plugin(*args, **kwargs)
        
        if instance is not None:
            instance.init_kwargs = kwargs
            
        return instance
       
    @classmethod
    def modules(cls):
        modules = {}
        
        for plugin in cls.TypeInfo.values():
            module = plugin['module']
            if module not in modules:
                modules[module] = {}
            modules[plugin['module']][plugin['name']] = plugin
 
        return modules
         
    @classmethod
    def register(cls, plugin, **kwargs):
        info = {
            'name': plugin.__name__, # __class__ type
            'module': str(plugin.__module__).split('.')[-2],
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
            ChatModel, VideoSource, VideoOutput,
            UserPrompt, AutoPrompt, VADFilter,
            TextStream, VideoOverlay, RateLimit,
            AudioInputDevice, AudioOutputDevice, AudioRecorder, 
            NanoDB, DataTable, DataLogger, EventFilter,
        )

        from nano_llm.plugins.speech.riva_asr import RivaASR
        from nano_llm.plugins.speech.riva_tts import RivaTTS
        from nano_llm.plugins.speech.piper_tts import PiperTTS
        from nano_llm.plugins.speech.whisper_asr import WhisperASR

        from nano_llm.plugins.audio.web_audio import WebAudioIn, WebAudioOut
        
        # LLM
        DynamicPlugin.register(ChatModel)   
        DynamicPlugin.register(UserPrompt)
        DynamicPlugin.register(AutoPrompt)   
        DynamicPlugin.register(TextStream)
        
        # speech
        DynamicPlugin.register(VADFilter)
        DynamicPlugin.register(WhisperASR)
        DynamicPlugin.register(RivaASR)
        DynamicPlugin.register(RivaTTS)
        DynamicPlugin.register(PiperTTS)
        
        # audio
        DynamicPlugin.register(AudioInputDevice)
        DynamicPlugin.register(AudioOutputDevice)
        DynamicPlugin.register(AudioRecorder)
        DynamicPlugin.register(WebAudioIn)
        DynamicPlugin.register(WebAudioOut)

        # video
        DynamicPlugin.register(VideoSource)
        DynamicPlugin.register(VideoOutput)
        DynamicPlugin.register(VideoOverlay)
        DynamicPlugin.register(RateLimit)
        
        # database
        DynamicPlugin.register(NanoDB)
        DynamicPlugin.register(DataTable)
        DynamicPlugin.register(DataLogger)
        DynamicPlugin.register(EventFilter)
        
        logging.info(f"Registered dynamic plugin types:\n\n{pprint.pformat(DynamicPlugin.TypeInfo, indent=2)}")

    
