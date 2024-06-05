#!/usr/bin/env python3
import torch

from nano_llm import Plugin
from nano_llm.web import WebServer
from nano_llm.utils import convert_audio, resample_audio, audio_db


WEBAUDIO_SAMPLE_RATE=48000
WEBAUDIO_STATS_RATE=4800


class WebAudioIn(Plugin):
    """
    Recieves audio samples over the websocket from client microphone.
    """
    def __init__(self, **kwargs):
        super().__init__(inputs=0, outputs='audio', **kwargs)
        self.sample_counter = 0
        WebServer.add_listener(self.on_websocket)

    def on_websocket(self, samples, msg_type=0, **kwargs): 
        if msg_type != WebServer.MESSAGE_AUDIO:
            return
            
        samples = convert_audio(samples, torch.float32)
        
        self.output(samples, sample_rate=WEBAUDIO_SAMPLE_RATE)
        self.sample_counter += len(samples)
        
        if self.sample_counter >= WEBAUDIO_STATS_RATE:
            db = audio_db(samples)
            self.send_stats(audio_db=db, summary=f"{db:.1f}dB")
            self.sample_counter = 0

        
class WebAudioOut(Plugin):
    """
    Sends audio samples over websocket to the client speakers.
    """
    def __init__(self, **kwargs):
        super().__init__(inputs='audio', outputs=0, **kwargs)
        self.sample_counter = 0
        
    def process(self, samples, sample_rate=None, **kwargs): 
        if not WebServer.Instance:
            return
            
        if sample_rate is not None and sample_rate != WEBAUDIO_SAMPLE_RATE:
            samples = resample_audio(samples, sample_rate, WEBAUDIO_SAMPLE_RATE, warn=self)
            
        WebServer.Instance.send_message(samples, type=WebServer.MESSAGE_AUDIO)
        
        self.sample_counter += len(samples)
        
        if self.sample_counter >= WEBAUDIO_STATS_RATE:
            db = audio_db(samples)
            self.send_stats(audio_db=db, summary=f"{db:.1f}dB")
            self.sample_counter = 0
