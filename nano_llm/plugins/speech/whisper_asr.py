#!/usr/bin/env python3
import time
import queue
import threading
import logging

import torch
import numpy as np

from nano_llm.plugins import AutoASR
from nano_llm.utils import convert_tensor, convert_audio, resample_audio

try:
    from whisper_trt.model import load_trt_model
    HAS_WHISPER_TRT=True
except ImportError as error:
    HAS_WHISPER_TRT=False
    logging.warning(f"whisper_trt not installed (requires JetPack 6 / L4T R36) - WhisperASR plugins will fail to initialize")


class WhisperASR(AutoASR):
    """
    Whisper ASR with TensorRT (github.com/NVIDIA-AI-IOT/whisper_trt)
    
    Inputs:  incoming audio samples coming from another audio plugin

    Output:  two channels, the first for word-by-word 'partial' transcript strings
             the second is for the full/final sentences
    """
    ModelCache = {}
    
    def __init__(self, model: str='base', language_code: str='en_US', 
                 partial_transcripts: float=0.25, use_cache: bool=True, **kwargs):
        """
        Whisper streaming voice transcription with TensorRT.
        
        Args:
          model (str): The Whisper model to load - 'tiny' (39M), 'base' (74M), 'small' (244M)
          language_code (str): The language to load the models for (currently 'en_US')
          partial_transcripts (float): The update rate for streaming partial ASR results (in seconds, <=0 to disable)
          use_cache (bool): If true, reuse the model if it's already in memory (and cache it if it needs to be loaded)
        """
        super().__init__(outputs=['final', 'partial'], **kwargs)
        
        if not HAS_WHISPER_TRT:
            raise ImportError("whisper_trt not installed (requires JetPack 6 / L4T R36)")
            
        self.language = language_code.lower().replace('-', '_').split('_')[0]  # ignore the country
        
        if self.language != 'en':
            raise ValueError(f"only the Whisper models for English are currently supported (requested {language_code})")
            
        model = model.lower().replace('-', '_').replace('whisper_', '')
        
        if model == 'whisper':
            model = 'base'
            
        self.model_name = f"{model}.{self.language}"
        self.sample_rate = 16000  # what the Whisper models use
        self.last_partial = 0
        self.chunks = []
        
        self.add_parameter('partial_transcripts', type=float, default=partial_transcripts)
        
        logging.info(f"loading Whisper model '{self.model_name}' with TensorRT")
        
        if use_cache and self.model_name in self.ModelCache:
            self.model = self.ModelCache[self.model_name]
        else:
            self.model = load_trt_model(self.model_name, verbose=True)
            if use_cache:
                self.ModelCache[self.model_name] = self.model
                
        self.model.transcribe(np.zeros(1536, dtype=np.float32)) # warmup
    
    @classmethod
    def type_hints(cls):
        return {'model': {'options': ['tiny', 'base', 'small']}}
        
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
        rtfx = time_audio / time_elapsed
        
        logging.debug(f"Whisper {self.model_name} - transcribed {time_audio:.2f} sec of audio in {time_elapsed:.2f} sec (RTFX={rtfx:2.2f}x)")
        self.send_stats(rtfx=rtfx, summary=f"RTFX={rtfx:2.2f}x")
        
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
                len(self.chunks) > 0 and self.input_queue.empty() and
                len(self.outputs[AutoASR.OutputPartial]) > 0
            ):
            self.last_partial = current_time
            partial_transcript = self.transcribe(self.chunks)
            
            if partial_transcript:
                self.output(partial_transcript, channel=AutoASR.OutputPartial, partial=True)     

