#!/usr/bin/env python3
import time
import queue
import threading
import logging

from nano_llm import Plugin
from nano_llm.utils import audio_silent


class AutoASR(Plugin):
    """
    Base class for ASR model plugins, supporting live transcription of audio streams.
    """
    OutputFinal=0    # output full transcripts (channel 0)
    OutputPartial=1  # output partial transcripts (channel 1)
    
    @staticmethod
    def from_pretrained(asr=None, **kwargs):
        """
        Factory function for automatically creating different types of ASR models.
        The `tts` param should either be 'riva' or 'xtts' (or name/path of XTTS model)
        The kwargs are forwarded to the TTS plugin implementing the model.
        """
        if not asr:
            return None
            
        asrl = asr.lower()
        
        if asrl == 'none' or asrl.startswith('disable'):
            return None
        
        if asrl.startswith('riva'):
            from nano_llm.plugins.audio.riva_asr import RivaASR
            return RivaASR(**kwargs)
        elif asrl.startswith('whisper'):
            from nano_llm.plugins.audio.whisper_asr import WhisperASR
            return WhisperASR(**{**kwargs, 'model' : asr})
        else:
            raise ValueError(f"ASR model type should be 'riva'")
    
    def add_punctuation(self, text):
        """
        Make sure that the transcript ends in some kind of punctuation
        """
        x = text.strip()
        
        if not any([x[-1] == y for y in ('.', ',', '?', '!', ':')]):
            return text + '.'
            
        return text
