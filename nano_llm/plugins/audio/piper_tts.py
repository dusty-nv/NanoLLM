#!/usr/bin/env python3
import os
import time
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
    def __init__(self, voice='en_US-libritts-high', voice_speaker=None,
                 language_code='en_US', sample_rate_hz=22050, voice_rate=1.0, 
                 model_cache=os.environ.get('PIPER_CACHE'), **kwargs):
        """
        Load Piper TTS model and set default options (many of which can be changed at runtime)
        """
        super().__init__(**kwargs)
        
        if not voice:
            voice = 'en_US-libritts-high' #'en_GB-cori-high' #'en_US-lessac-high'
            
        self.rate = voice_rate
        self.sample_rate = sample_rate_hz
        self.cache_path = model_cache  
        self.resampler = None
        self.languages = []
        self.voices = []
        
        self.voices_info = get_voices(self.cache_path, update_voices=True)

        for key, model_info in self.voices_info.items():
            if model_info['language']['code'] not in self.languages:
                self.languages.append(model_info['language']['code'])
        
        self.language = language_code
        self.voice = voice
        
        if voice_speaker:
            self.speaker = voice_speaker

        logging.debug(f"running Piper TTS model warm-up for {self.voice}")
        self.process("This is a test of the text to speech.")
        
    @property
    def voice(self):
        return self._voice
        
    @voice.setter
    def voice(self, voice):
        try:
            model_path, config_path = find_voice(voice, [self.cache_path])
        except Exception as error:
            ensure_voice_exists(voice, self.cache_path, self.cache_path, self.voices_info)
            model_path, config_path = find_voice(voice, [self.cache_path])
            
        logging.info(f"loading Piper TTS model from {model_path}")
        self.model = PiperVoice.load(model_path, config_path=config_path, use_cuda=True)
        self.model_sample_rate = self.model.config.sample_rate
        
        self._voice = voice
        self._speaker_id_map = self.voices_info[self._voice]['speaker_id_map']
        
        if not self._speaker_id_map:
            self._speaker_id_map = {'Default': 0}
        
        self.speaker = self.speakers[0]

    @property
    def speakers(self):
        return list(self._speaker_id_map.keys())
        
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

    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect before being passed to the TTS for generation.
        """
        text = self.buffer_text(text)    
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
                samples = resample_audio(samples, self.model_sample_rate, self.sample_rate)

            num_samples += len(samples)
            self.output(samples)

        time_elapsed = time.perf_counter() - time_begin
        logging.debug(f"finished TTS request, streamed {num_samples} samples at {self.sample_rate/1000:.1f}KHz - {num_samples/self.sample_rate:.2f} sec of audio in {time_elapsed:.2f} sec (RTFX={num_samples/self.sample_rate/time_elapsed:.4f})")
