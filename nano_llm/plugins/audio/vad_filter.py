#!/usr/bin/env python3
import time
import logging

import torch
import numpy as np

from nano_llm import Plugin
from nano_llm.utils import convert_tensor, convert_audio, resample_audio

from whisper_trt.vad import load_vad


class VADFilter(Plugin):
    """
    Voice Activity Detection (VAD) model that filters/drops audio when there is no speaking.
    This is typically used before ASR to reduce erroneous transcripts from background noise.
    
    Inputs:  incoming audio samples (prefer @ 16KHz)
    Output:  audio samples that have voice activity
    """
    def __init__(self, vad_threshold=0.5, vad_window=0.5, audio_chunk=0.1, **kwargs):
        """
        Parameters:
        
          vad_threshold (float): If any of the audio chunks in the window are above this confidence,
                                 then the audio will be forwarded to the next plugin (otherwise dropped).
                                 This should be between 0 and 1, with a threshold of 0 emitting all audio.
                                 
          vad_window (float): The duration of time (in seconds) that the VAD filter processes over.
                              If speaking isn't detected for this long, it will be considered silent.                      
        """
        super().__init__(output_channels=1, **kwargs)
        
        self.vad = load_vad()
        
        self.vad_threshold = vad_threshold
        self.vad_window = vad_window
        self.vad_filter = []
        
        self.speaking = False     # true when voice is detected
        self.speaking_start = 0   # time when voice first detected
        self.sample_rate = 16000  # what the Silero VAD model uses

        if audio_chunk < 16:
            self.audio_chunk = int(audio_chunk * self.sample_rate)
        else:
            self.audio_chunk = int(audio_chunk)
        
        self.buffers = []
        self.buffered_samples = 0
        
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
            
        if speaking and not self.speaking:
            logging.info(f"voice activity detected (conf={self.vad_filter[-1]:.3f})")
            self.speaking_start = time.perf_counter()

        if speaking:
            self.output(samples, sample_rate=self.sample_rate, vad_confidence=vad_prob)
        elif self.speaking:
            self.output(None, vad_confidence=vad_prob) # EOS   
            logging.info(f"voice activity ended (duration={time.perf_counter()-self.speaking_start:.2f}s, conf={self.vad_filter[-1]:.3f})")
            
        self.speaking = speaking
         
