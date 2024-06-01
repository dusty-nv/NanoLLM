#!/usr/bin/env python3
import time
import queue
import threading
import logging
import numpy as np

import riva.client
import riva.client.audio_io

from nano_llm.plugins import AutoASR
from nano_llm.utils import convert_tensor, convert_audio, resample_audio


class RivaASR(AutoASR):
    """
    Streaming ASR service using NVIDIA Riva
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    Inputs:  incoming audio samples coming from another audio plugin
             RivaASR can also open an audio device connected to this machine
             
    Output:  two channels, the first for word-by-word 'partial' transcript strings
             the second is for the full/final sentences
    """
    def __init__(self, riva_server='localhost:50051',
                 language_code='en-US', sample_rate_hz=48000, 
                 asr_threshold=-2.5, audio_chunk=0.1,
                 automatic_punctuation=True, inverse_text_normalization=False, 
                 profanity_filter=False, boosted_lm_words=None, boosted_lm_score=4.0, 
                 **kwargs):
        """
        Parameters:
        
          riva_server (str) -- URL of the Riva GRPC server that should be running
          audio_input (int) -- audio input device number for locally-connected microphone
          sample_rate_hz (int) -- sample rate of any incoming audio or device (typically 16000, 44100, 48000)
          audio_chunk (int) -- the audio input buffer length (in samples) to use for input devices
          audio_input_channels (int) -- 1 for mono, 2 for stereo
          inverse_text_normalization (bool) -- https://developer.nvidia.com/blog/text-normalization-and-inverse-text-normalization-with-nvidia-nemo/
        """
        super().__init__(output_channels=2, **kwargs)
        
        self.server = riva_server
        self.auth = riva.client.Auth(uri=riva_server)

        self.audio_queue = AudioQueue(self)
        self.input_device = None #audio_input_device
        self.language_code = language_code
        self.confidence_threshold = asr_threshold
        self.keep_alive_timeout = 5  # requests timeout after 1000 seconds
        
        if sample_rate_hz is None:
            self.sample_rate = 16000
        else:
            self.sample_rate = sample_rate_hz    

        if audio_chunk < 16:
            self.audio_chunk = int(audio_chunk * self.sample_rate)
        else:
            self.audio_chunk = int(audio_chunk)
        
        self.asr_service = riva.client.ASRService(self.auth)
        self.asr_request = None

        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                max_alternatives=1,
                profanity_filter=profanity_filter,
                enable_automatic_punctuation=automatic_punctuation,
                verbatim_transcripts=not inverse_text_normalization,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        
        riva.client.add_word_boosting_to_config(self.asr_config, boosted_lm_words, boosted_lm_score)

    def run(self):
        if self.input_device is not None:
            self.mic_thread = threading.Thread(target=self.run_mic, daemon=True)
            self.mic_thread.start()
    
        self.generate(self.audio_queue)
        
    def run_mic(self):
        logging.info(f"opening audio input device ({self.input_device})")
        self.generate(riva.client.audio_io.MicrophoneStream(
            self.sample_rate,
            self.audio_chunk,
            device=self.input_device,
        ))
        
    def generate(self, audio_generator):
        while True:
            with audio_generator:
                responses = self.asr_service.streaming_response_generator(
                    audio_chunks=audio_generator, streaming_config=self.asr_config
                )
            
                for response in responses:
                    if not response.results:
                        continue

                    for result in response.results:
                        transcript = result.alternatives[0].transcript.strip()
                        if result.is_final:
                            score = result.alternatives[0].confidence
                            if score >= self.confidence_threshold:
                                logging.debug(f"Riva submitting ASR transcript (confidence={score:.3f}) -> '{transcript}'")
                                self.output(self.add_punctuation(transcript), AutoASR.OutputFinal)
                            else:
                                logging.warning(f"Riva dropping ASR transcript (confidence={score:.3f} < {self.confidence_threshold:.3f}) -> '{transcript}'")
                        else:
                            self.output(transcript, AutoASR.OutputPartial, partial=True)
        

class AudioQueue:
    """
    Implement same context manager/iterator interfaces as Riva's MicrophoneStream
    for ingesting ASR audio samples from external sources via the plugin's input queue.
    """
    def __init__(self, asr):
        self.asr = asr

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
        
    def __next__(self) -> bytes:
        data = []
        size = 0
        chunk_size = self.asr.audio_chunk * 2  # 2 bytes per int16 sample
        time_begin = time.perf_counter()
        
        while size <= chunk_size:  
            try:
                samples, kwargs = self.asr.input_queue.get(timeout=self.asr.keep_alive_timeout) 
            except queue.Empty:
                logging.debug(f"sending ASR keep-alive silence (idle for {self.asr.keep_alive_timeout} seconds)")
                return bytes(chunk_size)

            if samples is None:
                raise StopIteration
                
            sample_rate = kwargs.get('sample_rate')
            
            if sample_rate is not None and sample_rate != self.asr.sample_rate:
                samples = resample_audio(samples, sample_rate, self.asr.sample_rate, warn=self)

            data.append(convert_audio(samples, dtype=np.int16).tobytes())
            size += len(data[-1])

        return b''.join(data)
    
    def __iter__(self):
        return self

    
