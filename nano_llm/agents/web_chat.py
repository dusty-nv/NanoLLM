#!/usr/bin/env python3
import os
import re
import logging
import natsort
import threading
import numpy as np

from nano_llm import StopTokens, BotFunctions, bot_function
from nano_llm.web import WebServer
from nano_llm.utils import ArgParser, KeyboardInterrupt
from nano_llm.plugins import AutoASR

from .voice_chat import VoiceChat


class WebChat(VoiceChat):
    """
    Adds webserver hooks to ASR/TTS voice chat agent and provide web UI.
    When a multimodal model is loaded, the user can drag & drop images
    to chat about into the UI.  Also supports streaming the client's
    microphone and output speakers using WebAudio.
    """
    def __init__(self, **kwargs):
        """
        Args:
        
          upload_dir (str): the path to save files uploaded from the client
          
        See :class:`VoiceChat` and :class:`WebServer` for inherited arguments.
        """
        super().__init__(**kwargs)
        
        # temp singleton instance until bot_function closures are fixed
        WebChat.Instance = self
        
        # add additional hooks to the voice components
        if self.asr:
            self.asr.add(self.on_asr_partial, AutoASR.OutputPartial, threaded=True)
            #self.asr.add(self.on_asr_final, AutoASR.OutputFinal)
        
        self.llm.add(self.on_llm_reply, threaded=True)
        
        if self.tts:
            self.tts_output.add(self.on_tts_samples, threaded=True)
        
        # configure system prompt and function calling
        self._system_instruct = self.llm.history.system_prompt
        
        self.enable_autodoc = True
        self.enable_profile = True
        
        self.user_profile = []  # stores info from SAVE()
        
        self.llm.functions = BotFunctions()
        self.generate_system_prompt()
        
        # filters for sanitizing chat HTML
        self.web_regex = [
            (re.compile(r'`(.*?)`'), r'<code>\1</code>'),  # code blocks
            (re.compile(r'\*(.*?)\*'), r'*<i>\1</i>*'),    # emotives inside asterisks
        ]

        for function in self.llm.functions:
            regex = re.compile(f"({function.name}\(.*?\))")
            self.web_regex.append((regex, r'<code>\1</code>'))
            
            if self.tts:
                self.tts.filter_regex.append((regex, function.name.lower()))
            
        # create webserver / websocket
        web_title = kwargs.get('web_title')
        web_title = web_title if web_title else 'llamaspeak'
        
        self.server = WebServer(
            msg_callback=self.on_message, 
            model_name=os.path.basename(self.llm.model.config.name),
            title=web_title,
            **kwargs
        )
             
    def on_message(self, msg, msg_type=0, metadata='', **kwargs):
        """
        Websocket message handler from the client.
        """
        if msg_type == WebServer.MESSAGE_JSON:
            if 'chat_history_reset' in msg:
                #self.llm('/reset')
                self.generate_system_prompt(force_reset=True)
                
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    client_init_msg = {
                        'system_prompt': self.llm.history.system_prompt,
                        'bot_functions': BotFunctions.generate_docs(prologue=False),
                        'user_profile': '\n'.join(self.user_profile),
                    }
                    
                    if self.tts:
                        voices = self.tts.voices
                        
                        if len(voices) > 20:
                            voices = natsort.natsorted(voices)
                  
                        speakers = self.tts.speakers
                        
                        if len(speakers) > 20:
                            speakers = natsort.natsorted(speakers)
                            
                        client_init_msg.update({
                            'tts_voice': self.tts.voice, 
                            'tts_voices': voices, 
                            'tts_speaker': self.tts.speaker, 
                            'tts_speakers': speakers, 
                            'tts_rate': self.tts.rate
                        })

                    self.server.send_message(client_init_msg)
                    threading.Timer(1.0, lambda: self.send_chat_history()).start()
                    
            if 'system_prompt' in msg:
                self.generate_system_prompt(msg['system_prompt'])
            if 'function_calling' in msg:
                self.llm.history.functions = BotFunctions() if msg['function_calling'] else None
                self.generate_system_prompt()
            if 'function_autodoc' in msg:
                self.enable_autodoc = msg['function_autodoc']
                self.generate_system_prompt()
            if 'user_profile' in msg:
                self.user_profile = [x.strip() for x in msg['user_profile'].split('\n')]
                self.generate_system_prompt()
            if 'enable_profile' in msg:
                self.enable_profile = msg['enable_profile']
                self.generate_system_prompt()
            if 'tts_voice' in msg and self.tts:
                self.tts.voice = msg['tts_voice']
                self.server.send_message({'tts_speaker': self.tts.speaker, 'tts_speakers': self.tts.speakers})
            if 'tts_speaker' in msg and self.tts:
                self.tts.speaker = msg['tts_speaker']
            if 'tts_rate' in msg and self.tts:
                self.tts.rate = float(msg['tts_rate'])
        elif msg_type == WebServer.MESSAGE_TEXT:  # chat input
            self.on_interrupt()
            self.prompt(msg.strip('"'))
        elif msg_type == WebServer.MESSAGE_AUDIO:  # web audio (mic)
            if self.asr:
                self.asr(msg)
        elif msg_type == WebServer.MESSAGE_IMAGE:
            logging.info(f"recieved {metadata} image message {msg.size} -> {msg.filename}")
            self.llm(['/reset', msg.filename])
            threading.Timer(0.1, self.send_chat_history).start()
        else:
            logging.warning(f"WebChat agent ignoring websocket message with unknown type={msg_type}")

    @bot_function(docs='nosig')
    def SAVE(text=None):
        """
        SAVE("<insert info here>") - save information about the user, for example SAVE("Mary likes to garden")
        """
        self = WebChat.Instance
        
        if text:
            text = text.strip()
            
        if text and text.lower() != "info":  # sometimes the bot likes to call it like an example
            self.user_profile.append(text)
            log_msg = f"Saved to user profile: '{text}'"
            logging.warning(log_msg)
            self.server.send_message({'user_profile': '\n'.join(self.user_profile)})
            self.server.send_alert(log_msg, category='user_profile', level='success')
            
    @property
    def system_prompt(self):
        """
        Get the instruction prologue of the system prompt, before functions or RAG are added.
        """
        return self._system_instruct
        
    @system_prompt.setter
    def system_prompt(self, instruction):
        """
        Set the instruction prologue of the system prompt, before functions or RAG are added.
        """
        self.generate_system_prompt(instruction)
     
    def generate_system_prompt(self, instruct=None, enable_autodoc=None, enable_profile=None, force_reset=False):
        """
        Assemble the system prompt from the instruction prologue, function docs, and user profile.
        """
        if instruct is None:
            instruct = self._system_instruct
        else:
            self._system_instruct = instruct
            
        if enable_autodoc is None:
            enable_autodoc = self.enable_autodoc
        
        if enable_profile is None:
            enable_profile = self.enable_profile
            
        system_prompt = [instruct]
        
        if enable_autodoc and self.llm.functions:
            system_prompt.append("\n" + BotFunctions.generate_docs())
            
        if enable_profile and self.user_profile:
            system_prompt.append(
                "\n".join(["\nHere are the things you previously saved about the user:\n"] + \
                ["* " + x for x in self.user_profile if x]
            ))
        
        system_prompt = "\n".join(system_prompt)
            
        if force_reset or system_prompt != self.llm.history.system_prompt:
            self.llm.history.system_prompt = system_prompt
            self.llm.history.reset()
            if hasattr(self, 'server'): # server may not be created yet
                threading.Timer(0.1, self.send_chat_history).start()
                
        return system_prompt
           
    def on_asr_partial(self, text):
        """
        Update the web chat history when a partial ASR transcript arrives.
        """
        self.send_chat_history()
        threading.Timer(1.5, self.on_asr_waiting, args=[text]).start()
        
    def on_asr_waiting(self, transcript):
        """
        If the ASR partial transcript hasn't changed, probably a misrecognized sound or echo (cancel it)
        """
        if self.asr_history == transcript:
            logging.warning(f"ASR partial transcript has stagnated, dropping from chat ({self.asr_history})")
            self.asr_history = None
            self.send_chat_history() # drop the rejected ASR from the client

    def on_llm_reply(self, text):
        """
        Update the web chat history when the latest LLM response arrives.
        """
        self.send_chat_history()
        
    def on_tts_samples(self, audio):
        """
        Send audio samples to the client when they arrive.
        """
        self.server.send_message(audio, type=WebServer.MESSAGE_AUDIO)
        
    def send_chat_history(self):
        """
        Sanitize the chat history for HTML and send it to the client.
        """
        history, num_tokens, max_context_len = self.llm.chat_state
            
        if self.asr and self.asr_history:
            history.append({'role': 'user', 'text': self.asr_history})
            
        def web_text(text):
            text = text.strip()
            text = text.strip('\n')
            text = text.replace('\n', '<br/>')
            text = text.replace('<s>', '')
            
            for stop_token in StopTokens:
                text = text.replace(stop_token, '')
                
            for regex, replace in self.web_regex:
                text = regex.sub(replace, text)
                
            return text
          
        def web_image(image):
            if not isinstance(image, str):
                if not hasattr(image, 'filename'):
                    return None
                image = image.filename
            return os.path.join(self.server.mounts.get(os.path.dirname(image), ''), os.path.basename(image))
            
        for entry in history:
            if 'text' in entry:
                entry['text'] = web_text(entry['text'])
            if 'image' in entry:
                entry['image'] = web_image(entry['image'])
                
        self.server.send_message({
            'chat_history': history,
            'chat_stats': {
                'num_tokens': num_tokens,
                'max_context_len': max_context_len,
            }
        })
 
    def start(self):
        """
        Start the webserver & websocket listening in other threads.
        """
        super().start()
        self.server.start()
        return self

    Instance = None  # singleton instance
    

if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_output', 'web'])
    args = parser.parse_args()
    
    agent = WebChat(**vars(args))
    interrupt = KeyboardInterrupt()
    
    agent.run() 
    
