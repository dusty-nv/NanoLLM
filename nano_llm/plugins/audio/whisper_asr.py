#!/usr/bin/env python3
import time
import queue
import threading
import logging

import torch
import numpy as np

from nano_llm.plugins import AutoASR
from nano_llm.utils import convert_tensor, convert_audio, resample_audio

from whisper_trt.vad import load_vad
from whisper_trt.model import load_trt_model


class WhisperASR(AutoASR):
    """
    Whisper ASR with TensorRT (github.com/NVIDIA-AI-IOT/whisper_trt)
    
    Inputs:  incoming audio samples coming from another audio plugin

    Output:  two channels, the first for word-by-word 'partial' transcript strings
             the second is for the full/final sentences
    """
    def __init__(self, model='small', language_code='en_US', partial_transcripts=0.25, **kwargs):
        """
        Parameters:
        
          asr (str): The Whisper model to load - 'tiny' (39M), 'base' (74M), 'small' (244M)
        """
        super().__init__(output_channels=2, **kwargs)
        
        self.language = language_code.lower().replace('-', '_').split('_')[0]  # ignore the country
        
        if self.language != 'en':
            raise ValueError(f"only the Whisper models for English are currently supported (requested {language_code})")
            
        model = model.lower().replace('-', '_').replace('whisper_', '')
        
        if model == 'whisper':
            model = 'base'
            
        self.model_name = f"{model}.{self.language}"
        self.sample_rate = 16000  # what the Whisper models use
        self.chunks = []
        
        self.partial_transcripts = partial_transcripts
        self.last_partial = 0
        
        logging.info(f"loading Whisper model '{self.model_name}' with TensorRT")
        
        self.model = load_trt_model(self.model_name, verbose=True)
        self.model.transcribe(np.zeros(1536, dtype=np.float32)) # warmup
    
    def transcribe(self, chunks):
        """
        Transcribe a list of audio chunks, returning the text.
        """
        if not isinstance(chunks, list):
            chunks = [chunks]
        
        time_begin = time.perf_counter()    
        samples = torch.cat(chunks).cuda()
        transcript = self.model.transcribe(samples)['text'].strip()
        
        time_elapsed = time.perf_counter() - time_begin
        time_audio = len(samples) / self.sample_rate
        logging.debug(f"Whisper {self.model_name} - transcribed {time_audio:.2f} sec of audio in {time_elapsed:.2f} sec (RTFX={time_audio/time_elapsed:2.2f}x)")
        
        return transcript
        
    def process(self, samples, sample_rate=None, **kwargs):
        """
        Buffer incoming audio samples, waiting for them to end before generating the transcript.
        """
        if samples is None: # EOS
            if len(self.chunks) == 0:
                return

            transcript = self.transcribe(self.chunks)
            
            if transcript:
                self.output(transcript, channel=AutoASR.OutputFinal)
                
            self.chunks = []
            return
        
        samples = convert_audio(samples, dtype=torch.float32)
            
        if sample_rate is not None and sample_rate != self.sample_rate:
            samples = resample_audio(samples, sample_rate, self.sample_rate, warn=self)
        
        current_time = time.perf_counter()

        self.chunks.append(samples)
        
        if (self.partial_transcripts > 0 and 
                current_time - self.last_partial >= self.partial_transcripts and
                len(self.chunks) > 0 and self.input_queue.empty()
            ):
            self.last_partial = current_time
            partial_transcript = self.transcribe(self.chunks)
            
            if partial_transcript:
                self.output(partial_transcript, channel=AutoASR.OutputPartial, partial=True)
                

