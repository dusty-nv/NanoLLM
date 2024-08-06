#!/usr/bin/env python3
import os
import time
import natsort
import logging

import numpy as np
import torch
import torchaudio

from .auto_tts import AutoTTS
from nano_llm.utils import convert_tensor, convert_audio, resample_audio

from piper import PiperVoice
from piper.download import ensure_voice_exists, find_voice, get_voices

class PiperTTS(AutoTTS):
    """
    TTS service using Piper models through onnxruntime (https://github.com/rhasspy/piper)

    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    
    You can get the list of voices with tts.voices, and list of languages with tts.languages
    The speed can be set with tts.rate (1.0 = normal). The default voice is '...' with rate 1.0
    """
    ModelCache = {}
    
    def __init__(self, voice: str='en_US-libritts-high', voice_speaker: str='p339',
                 voice_rate: float=1.0, sample_rate_hz: int=22050, 
                 model_cache: str=os.environ.get('PIPER_CACHE'),
                 use_cache: bool=True, **kwargs):
        """
        Load Piper TTS model with ONNX Runtime using CUDA.
        
        Args:
          voice (str):  Name of the Piper model to use.
          voice_speaker (str):  Name or ID of the speaker to use for multi-voice models.
          voice_rate (float):  The speed of the voice (1.0 = 100%)
          sample_rate_hz (int):  Piper generates 16000 KHz for 'low' quality models and 22050 KHz for 'medium' and 'high' quality models.
          model_cache (str):  The directory on the server to save the models that get downloaded.
          use_cache (bool): If true, reuse the model if it's already in memory (and cache it if it needs to be loaded)
        """
        super().__init__(outputs='audio', **kwargs)
        
        if not voice:
            voice = 'en_US-libritts-high' #'en_GB-cori-high' #'en_US-lessac-high'

        self.sample_rate = sample_rate_hz
        self.cache_path = model_cache  
        self.use_cache = use_cache
        self.resampler = None
        self.languages = []
        self.voices = []
        self._voice = None
        
        self.voices_info = get_voices(self.cache_path, update_voices=True)

        for key, model_info in self.voices_info.items():
            if model_info['language']['code'] not in self.languages:
                self.languages.append(model_info['language']['code'])
        
        #self.language = language_code

        self.add_parameter('voice', default=voice)
        self.add_parameter('speaker', default=voice_speaker, options=self.speakers, kwarg='voice_speaker')
        self.add_parameter('rate', default=voice_rate, kwarg='voice_rate')
        
        logging.debug(f"running Piper TTS model warm-up for {self.voice}")
        self.process("This is a test of the text to speech.")
          
    @property
    def voice(self):
        return self._voice
        
    @voice.setter
    def voice(self, voice):
        if self._voice == voice:
            return
            
        if self.use_cache and voice in self.ModelCache:
            self.model = self.ModelCache[voice]
        else:
            try:
                model_path, config_path = find_voice(voice, [self.cache_path])
            except Exception as error:
                ensure_voice_exists(voice, self.cache_path, self.cache_path, self.voices_info)
                model_path, config_path = find_voice(voice, [self.cache_path])
                
            logging.info(f"loading Piper TTS model from {model_path}")
            self.model = PiperVoice.load(model_path, config_path=config_path, use_cuda=True)
            
            if self.use_cache:
                self.ModelCache[voice] = self.model
                
        self.model_sample_rate = self.model.config.sample_rate
        
        if self.sample_rate is None:
            self.sample_rate = self.model_sample_rate
            
        self._voice = voice
        self._speaker_id_map = self.voices_info[self._voice]['speaker_id_map']
        
        if not self._speaker_id_map:
            self._speaker_id_map = {'Default': 0}
        
        self._speaker_list = list(self._speaker_id_map.keys())
                  
        if len(self._speaker_list) > 20:
            self._speaker_list = natsort.natsorted(self._speaker_list)
            
        self.speaker = self.speakers[0]

    @property
    def speakers(self):
        return self._speaker_list
        
    @property
    def speaker(self):
        return self._speaker
        
    @speaker.setter
    def speaker(self, speaker):
        try:
            self._speaker_id = self._speaker_id_map[speaker]
            self._speaker = speaker
        except Exception as error:
            logging.warning(f"Piper TTS failed to set speaker to '{speaker}', ignoring... ({error})")
     
    '''       
    @property
    def language(self):
        return self._language
        
    @language.setter
    def language(self, language):
        self._language = language.lower().replace('-', '_').split('_')[0]  # drop the country code (e.g. 'en_US')
        self.voices = []
        
        for key, model_info in self.voices_info.items():
            if model_info['language']['code'].lower().startswith(self._language):
                self.voices.append(key)
    '''
    
    def process(self, text, final=None, partial=None, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect before being passed to the TTS for generation.
        """
        text = self.buffer_text(text, final=final, partial=partial)    
        text = self.filter_text(text)

        if not text or self.interrupted:
            logging.debug(f"Piper TTS {self.voice} waiting for more input text (buffering={self.buffering} interrupted={self.interrupted})")
            return
            
        logging.debug(f"generating Piper TTS with {self.voice} for '{text}'")

        time_begin = time.perf_counter()
        num_samples = 0
        
        synthesize_args = {
            "speaker_id": self._speaker_id,
            "length_scale": 1.0 / self.rate,
            "noise_scale": 0.667,     # noise added to the generator
            "noise_w": 0.8,           # phoneme width variation
            "sentence_silence": 0.2,  # seconds of silence after each sentence
        }
    
        stream = self.model.synthesize_stream_raw(text, **synthesize_args)

        for samples in stream:
            if self.interrupted:
                logging.debug(f"TTS interrupted, terminating request early:  {text}")
                return
            
            samples = convert_audio(samples, dtype=np.int16)
            
            if self.sample_rate != self.model_sample_rate:
                samples = resample_audio(samples, self.model_sample_rate, self.sample_rate, warn=self)

            num_samples += len(samples)
            self.output(samples, sample_rate=self.sample_rate)

        time_elapsed = time.perf_counter() - time_begin
        time_audio = num_samples / self.sample_rate
        rtfx = time_audio / time_elapsed
        
        logging.debug(f"finished TTS request, streamed {num_samples} samples at {self.sample_rate/1000:.1f}KHz - {time_audio:.2f} sec of audio in {time_elapsed:.2f} sec (RTFX={rtfx:.4f})")
        self.send_stats(rtfx=rtfx, summary=f"RTFX={rtfx:2.2f}x")
        
