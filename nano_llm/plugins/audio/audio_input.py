#!/usr/bin/env python3
import time
import wave
import logging
import pyaudio

import numpy as np
import torch
import torchaudio

from nano_llm import Plugin
from nano_llm.utils import convert_audio, resample_audio, find_audio_device, pyaudio_dtype


class AudioInputDevice(Plugin):
    """
    Capture audio from a microphone or soundcard device attached to the server.
    Outputs audio samples as an np.ndarray with dtype=int16
    """
    def __init__(self, audio_input_device: int = None, audio_input_channels: int = 1, 
                 sample_rate_hz: int = None, audio_chunk: float = 0.1, **kwargs):
        """
        Capture audio from a microphone or soundcard device attached to the machine.
        
        Args:
          audio_input_device (int):  Audio input device number (PyAudio / PortAudio)
          audio_input_channels (int):  1 for mono, 2 for stereo.
          sample_rate_hz (int):  Sample rate to open the device with (typically 16000, 44100, 48000),
                                 or None to use the device's default sampling rate.
          audio_chunk (float): The duration of time or number of audio samples captured per batch.
        """
        super().__init__(input_channels=0, **kwargs)
        
        self.pa = pyaudio.PyAudio()
        
        self.device = None
        self.device_info = find_audio_device(audio_input_device, self.pa)
        self.device_id = self.device_info['index']

        self.channels = audio_input_channels
        self.format = pyaudio.paFloat32 #pyaudio.paInt16
        
        if sample_rate_hz is None:
            self.sample_rate = int(self.device_info['defaultSampleRate'])
        else:
            self.sample_rate = sample_rate_hz
            
        if audio_chunk < 16:
            self.chunk_size = int(audio_chunk * self.sample_rate)
        else:
            self.chunk_size = int(audio_chunk)
            
        self.device_sample_rate = self.sample_rate
        self.device_chunk_size = self.chunk_size

    def __del__(self):
        self.close()
        self.pa.terminate()
    
    def open(self):
        if self.device:
            return
        
        sample_rates = [self.sample_rate, int(self.device_info['defaultSampleRate']), 16000, 22050, 32000, 44100, 48000] 
        chunk_sizes = []
        
        for sample_rate in sample_rates:
            chunk_sizes.append(int(self.chunk_size * sample_rate / self.sample_rate))
            
        for sample_rate, chunk_size in zip(sample_rates, chunk_sizes):
            try:    
                logging.info(f"trying to open audio input {self.device_id} '{self.device_info['name']}' (sample_rate={sample_rate}, chunk_size={chunk_size}, channels={self.channels})")
                
                self.device = self.pa.open(format=self.format,
                                channels=self.channels,
                                rate=sample_rate,
                                input=True,
                                input_device_index=self.device_id,
                                frames_per_buffer=chunk_size)
                                
                self.device_sample_rate = sample_rate
                self.device_chunk_size = chunk_size
                break
                
            except OSError as err:
                logging.warning(f'failed to open audio input {self.device_id} with sample_rate={sample_rate} ({err})')
                self.device = None
                
        if self.device is None:
            raise ValueError(f"failed to open audio input device {self.device_id} with any of these sample rates: {str(sample_rates)}")

        logging.success(f"opened audio input device {self.device_id}  '{self.device_info['name']}' (sample_rate_in={self.device_sample_rate}, sample_rate_out={self.sample_rate}, chunk_size={chunk_size}, channels={self.channels})")
      
    def close(self):
        if self.device is not None:
            self.device.stop_stream()
            self.device.close()
            self.device = None
     
    def reset(self):
        self.close()
        self.open()
        
    def capture(self):
        self.open()
            
        samples = self.device.read(self.device_chunk_size, exception_on_overflow=False)
        samples = torch.frombuffer(samples, dtype=pyaudio_dtype(self.format, to='pt'))  # samples = np.frombuffer(samples, dtype=pyaudio_dtype(self.format, to='np'))

        if self.sample_rate != self.device_sample_rate:
            samples = resample_audio(samples, self.device_sample_rate, self.sample_rate)
            expected_samples = self.chunk_size * self.channels
            
            if len(samples) != expected_samples and not hasattr(self, '_resample_warning'):
                logging.warning(f"resampled input audio from device {self.device_id} has {len(samples)} samples, but expected {expected_samples} samples")
                self._resample_warning = True

        #logging.debug(f"captured {len(samples)} audio samples from audio device {self.device_id} (dtype={samples.dtype})")
        return samples

    def run(self):
        self.open()
        while True:
            self.output(self.capture(), sample_rate=self.sample_rate)   
    
