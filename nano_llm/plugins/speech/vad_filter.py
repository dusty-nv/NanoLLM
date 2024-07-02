#!/usr/bin/env python3
import time
import logging

import torch
import numpy as np

from nano_llm import Plugin
from nano_llm.utils import convert_tensor, convert_audio, resample_audio, update_default

try:
    from whisper_trt.vad import load_vad
    HAS_WHISPER_TRT=True
except ImportError as error:
    HAS_WHISPER_TRT=False
    logging.warning(f"whisper_trt not installed (minimum BSP version JetPack 6 / L4T R36) - VADFilter plugins will fail to initialize")


class VADFilter(Plugin):
    """
    Voice Activity Detection (VAD) model that filters/drops audio when there is no speaking.
    This is typically used before ASR to reduce erroneous transcripts from background noise.
    
    Inputs:  incoming audio samples (prefer @ 16KHz)
    Output:  audio samples that have voice activity
    """
    ModelCache = {}
    
    def __init__(self, vad_threshold: float=0.5, vad_window: float=0.5, 
                 interrupt_after: float=0.5, audio_chunk: float=0.1, 
                 use_cache: bool=True, **kwargs):
        """
        Voice Activity Detection (VAD) model that filters/drops audio when there is no speaking.
        This is typically used before ASR to reduce erroneous transcripts from background noise.
    
        Args:
          vad_threshold (float): If any of the audio chunks in the window are above this confidence (0,1) for voice activity,
                                 then the audio will be forwarded to the next plugin (otherwise dropped).                 
          vad_window (float): The duration of time (in seconds) that the VAD filter processes over.
                              If speaking isn't detected for this long, it will be considered silent. 
          interrupt_after (float): Send an interruption signal to mute/silence the bot after this many
                                   seconds of sustained audio activity.                 
          audio_chunk (float): The duration of time or number of audio samples processed per batch.
          use_cache (bool): If true, reuse the model if it's already in memory (and cache it if it needs to be loaded)                     
        """
        super().__init__(outputs=['audio', 'interrupt'], **kwargs)
        
        if not HAS_WHISPER_TRT:
            raise ImportError("whisper_trt not installed (minimum BSP version JetPack 6 / L4T R36)")
            
        if use_cache and self.ModelCache:
            self.vad = self.ModelCache['silero']
        else:
            self.vad = load_vad()
            if use_cache:
                self.ModelCache['silero'] = self.vad

        self.vad_filter = []
        
        self.buffers = []
        self.buffered_samples = 0
        
        self.speaking = False     # true when voice is detected
        self.speaking_start = 0   # time when voice first detected
        self.sample_rate = 16000  # what the Silero VAD model uses
        self.sent_interrupt = False
        
        self.add_parameter('vad_threshold', name='Voice Threshold', type=float, range=(0,1), default=vad_threshold)
        self.add_parameter('vad_window', name='Window Length', type=float, default=vad_window)
        self.add_parameter('interrupt_after', type=float, default=interrupt_after)
        self.add_parameter('audio_chunk', type=float, default=audio_chunk)
        
        #self.apply_config(vad_threshold=vad_threshold, vad_window=vad_window, audio_chunk=audio_chunk)
        
    def process(self, samples, sample_rate=None, **kwargs):
        """
        Apply VAD filtering to incoming audio, only allowing it to pass through when speaking is detected.
        At the end of each sequence, None will be output once to mark EOS, until audio from speaking resumes.
        """
        if self.vad_threshold <= 0:
            self.output(samples, sample_rate=sample_rate)

        samples = convert_audio(samples, dtype=torch.float32)
           
        if sample_rate is not None and sample_rate != self.sample_rate:
            samples = resample_audio(samples, sample_rate, self.sample_rate, warn=self)

        self.buffers.append(samples)
        self.buffered_samples += len(samples)
        
        if self.buffered_samples < self.audio_chunk:
            return
            
        samples = torch.cat(self.buffers)
        
        self.buffers = []
        self.buffered_samples = 0
        
        vad_prob = float(self.vad(samples.cpu(), self.sample_rate).flatten()[0])
        
        if len(samples) / self.sample_rate * len(self.vad_filter) < self.vad_window:
            self.vad_filter.append(vad_prob)
        else:
            self.vad_filter[0:-1] = self.vad_filter[1:]
            self.vad_filter[-1] = vad_prob

        speaking = any(x > self.vad_threshold for x in self.vad_filter)
        curr_time = time.perf_counter()
            
        if speaking and not self.speaking:
            logging.info(f"voice activity detected (conf={self.vad_filter[-1]:.3f})")
            self.speaking_start = curr_time

        if speaking:
            self.output(samples, sample_rate=self.sample_rate, vad_confidence=vad_prob)
            
            if not self.sent_interrupt and (curr_time - self.speaking_start >= self.interrupt_after):
                self.sent_interrupt = True
                for plugin in self.outputs[1]:
                    plugin.interrupt(block=False)
                    
        elif self.speaking:
            self.output(None, vad_confidence=vad_prob) # EOS   
            self.send_interrupt = False
            logging.info(f"voice activity ended (duration={curr_time-self.speaking_start:.2f}s, conf={self.vad_filter[-1]:.3f})")
            
        self.speaking = speaking
        
        stats = {
            'speaking': speaking,
            'confidence': vad_prob,
            'summary': [f"{vad_prob*100:.1f}%"],
        }
        
        if speaking:
            stats['summary'].append(f"{curr_time-self.speaking_start:.1f}s")
        
        self.send_stats(**stats)

    @property
    def audio_chunk(self):
        return self._audio_chunk
        
    @audio_chunk.setter
    def audio_chunk(self, value):
        if value < 16:
            self._audio_chunk = int(value * self.sample_rate)
        else:
            self._audio_chunk = int(value)
    
    '''                
    def apply_config(self, vad_threshold : float = None, vad_window : float = None, audio_chunk : float = None, **kwargs):
        """
        Update VAD settings.
        
        Args:
          vad_threshold (float): If any of the audio chunks in the window are above this confidence,
                                 then the audio will be forwarded to the next plugin (otherwise dropped).
                                 This should be between 0 and 1, with a threshold of 0 emitting all audio.            
          vad_window (float): The duration of time (in seconds) that the VAD filter processes over.
                              If speaking isn't detected for this long, it will be considered silent.               
          audio_chunk (float): The duration of time or number of audio samples processed per batch. 
        """   
        self.vad_threshold = update_default(vad_threshold, self.vad_threshold, float)
        self.vad_window = update_default(vad_window, self.vad_window, float)
    
            

    def state_dict(self):
        return {
            **super().state_dict(),
            'vad_threshold': self.vad_threshold,
            'vad_window': self.vad_window,
            'audio_chunk': self.audio_chunk,
       }
    '''           
