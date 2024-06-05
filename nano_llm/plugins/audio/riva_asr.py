#!/usr/bin/env python3
import time
import queue
import threading
import logging
import numpy as np

import riva.client
import riva.client.audio_io

from nano_llm.plugins import AutoASR
from nano_llm.utils import convert_tensor, convert_audio, resample_audio, update_default


class RivaASR(AutoASR):
    """
    Streaming ASR service using NVIDIA Riva:
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html
    
    You need to have the Riva server running first:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
    
    Riva pre-trained ASR models:
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html#pretrained-asr-models
    
    What is Inverse Text Normalization?
    https://developer.nvidia.com/blog/text-normalization-and-inverse-text-normalization-with-nvidia-nemo/
    
    Inputs:  incoming audio samples coming from another audio plugin
             RivaASR can also open an audio device connected to this machine
             
    Output:  two channels, the first for word-by-word 'partial' transcript strings
             the second is for the full/final sentences
    """
    def __init__(self, riva_server : str = 'localhost:50051',
                 language_code : str = 'en-US', asr_threshold : float = -2.5, 
                 sample_rate_hz : int = 16000, audio_chunk : float = 0.1,
                 automatic_punctuation : bool = True, inverse_text_normalization : bool = False, 
                 profanity_filter : bool = False, boosted_words : str = None, boosted_score : float = 4.0, 
                 **kwargs):
        """
        Streaming ASR using NVIDIA Riva. You need to have the Riva container running first:
        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64
        
        Args:
          riva_server (str): URL and port of the Riva GRPC server that should be running.
          language_code (str): The language to use (see Riva docs for multilingual models)
          asr_threshold (float): Minimum confidence for the output to be kept (only applies to 'final' transcripts)
          sample_rate_hz (int): Sample rate of the incoming audio (typically 16000, 44100, 48000)
          audio_chunk (int): The duration of time or number of audio samples captured per batch.
          automatic_punctuation (bool): Enable periods, question marks, and exclamations added at the end of sentences.
          inverse_text_normalization (bool): Convert numbers and symbols to words instead of digits.
          profanity_filter (bool): Remove derogatory language from the transcripts.
          boosted_words (str): Words to boost when decoding the transcript (hotword, wakeword)
          boosted_score (float): The amount by which to boost the scores of the words above.  
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
        
        riva.client.add_word_boosting_to_config(self.asr_config, boosted_words, boosted_score)

    def apply_config(self, asr_threshold : float = None, **kwargs):
        """
        Streaming ASR using NVIDIA Riva.
        
        Args:
          asr_threshold (float): Minimum confidence for the output to be kept (only applies to 'final' transcripts)
        """   
        self.confidence_threshold = update_default(asr_threshold, self.confidence_threshold, float)

    def state_dict(self):
        return {
            **super().state_dict(),
            'asr_confidence': self.confidence_threshold,
       }
       
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

    
