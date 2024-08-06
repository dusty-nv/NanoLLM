#!/usr/bin/env python3
import time
import queue
import logging
import numpy as np

import riva.client
import riva.client.audio_io

from .auto_tts import AutoTTS
from nano_llm.utils import validate


class RivaTTS(AutoTTS):
    """
    Streaming TTS service using NVIDIA Riva
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    The available voices are from:
          https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html#voices
        
    Rate, pitch, and volume are dynamic SSML tags from:
      https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-basics-customize-ssml.html#customizing-rate-pitch-and-volume-with-the-prosody-tag
      
    Inputs:  words to speak (str)
    Output:  audio samples (np.ndarray, int16)
    """
    def __init__(self, riva_server: str = 'localhost:50051', voice: str = 'English-US.Female-1', 
                 language_code: str = 'en-US', sample_rate_hz: int = 48000, 
                 voice_rate: str = '100%', voice_pitch='default', voice_volume='default',
                 tts_buffering : str = 'punctuation',
                 **kwargs):
        """
        Streaming TTS using NVIDIA Riva. You need to have the Riva container running first:
        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
        
        Args:
          riva_server (str): URL and port of the Riva GRPC server that should be running.
          voice (str):  Name of the voice to use (e.g. 'English-US.Female-1', 'English-US.Male-1')
          language_code (str): The language to use (see Riva docs for models in other languages)
          sample_rate_hz (int):  The desired sample rate to generate audio in (defaults to 48KHz)
          voice_rate (str):  The speed of the voice (between '25%' and '250%')
          voice_pitch (str): Pitch shift to apply (between [-3,3] or [-150Hz, 150Hz])
          voice_volume (str): Increase or decrease the volume by [-13dB, 8dB]
          tts_buffering (str):  If 'punctuation', TTS will wait to generate until the end of sentences for better dynamics.  If 'time', TTS will wait until audio gap-out approaches.  If 'time,punctuation', will wait for both.
        """
        super().__init__(tts_buffering=tts_buffering, **kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)
        self.tts_service = riva.client.SpeechSynthesisService(self.auth)
        
        self.voice = voice   # these voice settings be changed at runtime
        self.rate = voice_rate
        self.pitch = voice_pitch
        self.volume = voice_volume

        self.language = language_code
        self.sample_rate = sample_rate_hz
        
        # find out how to query these for non-English models
        self.voices = [
            "English-US.Female-1",
            "English-US.Male-1"
        ]
        
        self.languages = ["en-US"]
        self.speakers = []
        self.speaker = ''
        
        if not self.voice:
            self.voice = "English-US.Female-1"
            
        self.process("This is a test of Riva text to speech.", flush=True)

    def apply_config(self, voice: str = None, voice_rate: str = None, voice_pitch: str = None, voice_volume: str = None, tts_buffering: str = None, **kwargs):
        """
        Streaming TTS using NVIDIA Riva.
        
        Args:
          voice (str):  Name of the voice to use (e.g. 'English-US.Female-1', 'English-US.Male-1')
          voice_rate (str):  The speed of the voice (between '25%' and '250%')
          voice_pitch (str): Pitch shift to apply (between [-3,3] or [-150Hz, 150Hz])
          voice_volume (str): Increase or decrease the volume by [-13dB, 8dB]
          tts_buffering (str):  If 'punctuation', TTS will wait to generate until the end of sentences for better dynamics.  If 'time', TTS will wait until audio gap-out approaches.  If 'time,punctuation', will wait for both.
        """
        self.voice = validate(voice, self.voice, str)
        self.rate = validate(voice_rate, self.rate, str)
        self.pitch = validate(voice_pitch, self.pitch, str)
        self.volume = validate(voice_volume, self.volume, str)
        self.buffering = validate(tts_buffering, self.buffering, str)

    def state_dict(self):
        return {
            **super().state_dict(),
            'voice': self.voice,
            'voice_rate': self.rate,
            'voice_pitch': self.pitch,
            'voice_volume': self.volume,
            'tts_buffering': self.buffering,
       }
       
    def process(self, text, **kwargs):
        """
        Inputs text, outputs stream of audio samples (np.ndarray, np.int16)
        
        The input text is buffered by punctuation/phrases as it sounds better,
        and filtered for emojis/ect, and has SSML tags applied (if enabled) 
        """
        if len(self.outputs[0]) == 0:
            #logging.debug(f"TTS has no output connections, skipping generation")
            return
            
        text = self.buffer_text(text)
        text = self.filter_text(text)
        text = self.apply_ssml(text)
        
        if not text or self.interrupted:
            return
            
        logging.debug(f"generating TTS for '{text}'")

        responses = self.tts_service.synthesize_online(
            text, self.voice, self.language, sample_rate_hz=self.sample_rate
        )

        for response in responses:
            if self.interrupted:
                logging.debug(f"TTS interrupted, terminating request early:  {text}")
                break
                
            samples = np.frombuffer(response.audio, dtype=np.int16)
            #logging.debug(f"TTS outputting {len(samples)} audio samples")
            self.output(samples)
            
        #logging.debug(f"done with TTS request '{text}'")
