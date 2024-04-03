#!/usr/bin/env python3
from nano_llm import Agent, Pipeline
from nano_llm.utils import ArgParser, print_table

from nano_llm.plugins import (
    UserPrompt, ChatQuery, PrintStream, 
    AutoASR, AutoTTS, RateLimit, ProcessProxy, 
    AudioOutputDevice, AudioOutputFile
)


class VoiceChat(Agent):
    """
    Agent for ASR → LLM → TTS pipeline.
    """
    def __init__(self, asr=None, llm=None, tts=None, **kwargs):
        """
        Args:
          asr (NanoLLM.plugins.AutoASR|str): the ASR plugin instance or model name to connect with the LLM.
          llm (NanoLLM.Plugin|str): The LLM model plugin instance (like ChatQuery) or model name.
          tts (NanoLLM.plugins.AutoTTS|str): the TTS plugin instance (or model name)- if None, will be loaded from kwargs.
        """
        super().__init__(**kwargs)

        #: The LLM model plugin (like ChatQuery)
        if isinstance(llm, str):
            kwargs['model'] = llm
            
        if not llm or isinstance(llm, str):
            self.llm = ProcessProxy('ChatQuery', **kwargs)  
        else:
            self.llm = llm
            
        self.llm.add(PrintStream(color='green'))
        
        #: The ASR plugin whose output is connected to the LLM.
        if not asr or isinstance(asr, str):
            self.asr = AutoASR.from_pretrained(asr=asr, **kwargs) 
        else:
            self.asr = asr
            
        if self.asr:
            self.asr.add(PrintStream(partial=False, prefix='## ', color='blue'), AutoASR.OutputFinal)
            self.asr.add(PrintStream(partial=False, prefix='>> ', color='magenta'), AutoASR.OutputPartial)
            
            self.asr.add(self.asr_partial, AutoASR.OutputPartial) # pause output when user is speaking
            self.asr.add(self.asr_final, AutoASR.OutputFinal)     # clear queues on final ASR transcript
            self.asr.add(self.llm, AutoASR.OutputFinal)  # runs after asr_final() and any interruptions occur
            
            self.asr_history = None  # store the partial ASR transcript

        #: The TTS plugin that speaks the LLM output.
        if not tts or isinstance(tts, str):
            self.tts = AutoTTS.from_pretrained(tts=tts, **kwargs) 
        else:
            self.tts = tts
            
        if self.tts:
            self.tts_output = RateLimit(kwargs['sample_rate_hz'], chunk=9600) # slow down TTS to realtime and be able to pause it
            self.tts.add(self.tts_output)
            self.llm.add(self.tts, ChatQuery.OutputWords)

            self.audio_output_device = kwargs.get('audio_output_device')
            self.audio_output_file = kwargs.get('audio_output_file')
            
            if self.audio_output_device is not None:
                self.audio_output_device = AudioOutputDevice(**kwargs)
                self.tts_output.add(self.audio_output_device)
            
            if self.audio_output_file is not None:
                self.audio_output_file = AudioOutputFile(**kwargs)
                self.tts_output.add(self.audio_output_file)
        
        #: Text prompts from web UI or CLI.
        self.prompt = UserPrompt(interactive=True, **kwargs)
        self.prompt.add(self.llm)
        
        # setup pipeline with two entry nodes
        self.pipeline = [self.prompt]

        if self.asr:
            self.pipeline.append(self.asr)
            
    def asr_partial(self, text):
        """
        Callback that occurs when the ASR has a partial transcript (while the user is speaking).
        These partial transcripts get revised mid-stream until the user finishes their phrase.
        This is also used for pausing/interrupting the bot output for when the user starts speaking.
        """
        self.asr_history = text
        if len(text.split(' ')) < 2:
            return
        if self.tts:
            self.tts_output.pause(1.0)

    def asr_final(self, text):
        """
        Callback that occurs when the ASR outputs when there is a pause in the user talking,
        like at the end of a sentence or paragraph.  This will interrupt/cancel any ongoing bot output.
        """
        self.asr_history = None
        self.on_interrupt()
        
    def on_interrupt(self):
        """
        Interrupt/cancel the bot output when the user submits (or speaks) a full query.
        """
        self.llm.interrupt(recursive=False)
        if self.tts:
            self.tts.interrupt(recursive=False)
            self.tts_output.interrupt(block=False, recursive=False) # might be paused/asleep
 
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_output'])
    args = parser.parse_args()
    
    agent = VoiceChat(**vars(args)).run() 
    