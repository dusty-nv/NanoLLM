#!/usr/bin/env python3
import time
import logging

from nano_llm import Plugin, NanoLLM, ChatHistory 
from nano_llm.web import WebServer
from nano_llm.utils import ImageTypes, print_table, update_default


class ChatSession(Plugin):
    """
    Plugin that feeds incoming text or ChatHistory to LLM and generates the reply.
    It can either internally manage the ChatHistory, or that can be done externally.
    
    Inputs:  (str or list[str]) -- one or more text prompts
             (dict) -- an existing ChatEntry dict
             (ChatHistory) -- use latest entry from chat history
     
    Outputs:  channel 0 (str) -- the partially-generated output text, token-by-token
              channel 1 (str) -- the partially-generated output text, word-by-word
              channel 2 (str) -- the entire final output text, after generation is complete
              channel 3 (StreamingReponse) -- the stream iterator from generate()    
              channel 4 (ndarray) -- the last CLIP image features/embeddings
    """
    OutputToken = 0
    OutputWords = 1
    OutputFinal = 2
    OutputStream = 3
    OutputImageEmbedding = 4
    
    def __init__(self, model : str = "princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT", 
                 api : str = "mlc", quantization : str = "q4f16_ft", 
                 max_context_len : int = None, chat_template : str = None,
                 **kwargs):
        """
        Load an LLM and run generation on chat requests.
        
        Args:
          model (str): Either the path to the model, or HuggingFace model repo/name.
          api (str): The model backend API to use:  'mlc', 'awq', or 'hf' (by default, it will attempt to be automatically determined)
          quantization (str): For MLC, 'q4f16_ft', 'q4f16_1', 'q8f16_ft', 'q8f16_1'. For AWQ, the path to the fully-quantized AWQ weights.
          max_context_len (str): The maximum chat length in tokens (by default, inherited from the model)
          chat_template (str|dict): The chat template to use like 'llama-2', 'vicuna-v1' (by default, will attempt to determine model type)              
        """
        super().__init__(outputs=['delta', 'words', 'final', 'stream', 'embed'], **kwargs)

        load_time = time.perf_counter()
        
        if isinstance(model, str):
            self.model_name = model
            self.model = NanoLLM.from_pretrained(
                model, api=api, 
                quantization=quantization, 
                max_context_len=max_context_len, 
                **kwargs
            )
        else:
            self.model = model
            self.model_name = self.config.name

        load_time = time.perf_counter() - load_time
        
        self.history = ChatHistory(self.model, chat_template=chat_template, **kwargs)
        self.functions = None
        self.stream = None
        
        self.max_context_len = self.model.config.max_length
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.min_new_tokens = kwargs.get('min_new_tokens', -1)
        self.wrap_tokens = kwargs.get('wrap_tokens', 512)
        
        self.do_sample = kwargs.get('do_sample', False)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.95)
            
        self.print_stats = kwargs.get('print_stats', kwargs.get('debug', False))
        
        warmup = True
        
        if warmup:
            warmup = warmup if isinstance(warmup, str) else 'What is 2+2?'
            self.history.append(role='user', text=warmup)
            logging.info(f"Warming up LLM with query '{warmup}'")
            logging.info(f"Warmup response:  '{self.model.generate(self.history.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
            self.history.reset()

        if load_time > 1:  # don't show if model was cached
            self.send_alert(f"Loaded {self.model_name} in {load_time:.1f} seconds", level='success')
    
    def apply_config(self, system_prompt : str = None, 
                     max_new_tokens : int = None, min_new_tokens : int = None,
                     do_sample : bool = None, temperature : float = None,
                     top_p : float = None, repetition_penalty : float = None, 
                     **kwargs):
        """
        Change LLM generation settings.
        
        Args:
          system_prompt (str):  Set the system prompt (changing this will reset the chat)
          max_new_tokens (int): The number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int): Force the model to generate a set number of output tokens (default: -1)
          do_sample (bool): If true, temperature/top_p sampling will be used over the logits.
          temperature (float): Randomness token sampling parameter (default=0.7, only used if do_sample=True)
          top_p (float): If set to < 1 and do_sample=True, only the smallest set of most probable tokens
                         with probabilities that add up to top_p or higher are kept for generation (default 0.95) 
          repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty (default: 1.0) 
        """   
        if system_prompt is not None:
            self.history.system_prompt = system_prompt

        self.max_new_tokens = update_default(max_new_tokens, self.max_new_tokens, int)
        self.min_new_tokens = update_default(min_new_tokens, self.min_new_tokens, int)
        self.do_sample = update_default(do_sample, self.do_sample, bool)
        self.temperature = update_default(temperature, self.temperature, float)
        self.top_p = update_default(top_p, self.top_p, float)

    def state_dict(self, html=True, **kwargs):
        return {
            **super().state_dict(),
            'model': self.model_name,
            'history': self.history.to_list(html=html),
            'num_tokens': self.history.num_tokens,
            'max_context_len': self.max_context_len,
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'system_prompt': self.history.system_prompt,
       }
                      
    @property
    def chat_history(self):
        return self.history.to_list()
        
    @property
    def chat_tokens(self):
        return self.history.num_tokens
        
    @property
    def chat_state(self):
        return self.chat_history, self.chat_tokens, self.max_context_len
    
    @property
    def config(self):
        return self.model.config

    def process(self, input, **kwargs):
        """
        Generate the reply to a prompt or the latest ChatHistory.
        
        Parameters:
        
          input (str|dict|ChatHistory) -- either a string prompt from the user,
                                          a ChatEntry dict, or ChatHistory dict.
                                          
        Returns:
        
          The generated text (token by token), if input was a string or dict.
          If input was a ChatHistory, returns the streaming iterator/generator.
        """
        self.apply_config(**kwargs)
        
        if input is None:
            return
            
        if isinstance(input, list):
            for x in input:
                self.process(x, **kwargs)
            return

        if self.interrupted:
            return
            
        # handle some special commands
        if isinstance(input, str):
            x = input.lower()
            if any([x == y for y in ('/reset', '/clear', 'reset', 'clear')]):
                self.history.reset()
                return
        
        # add prompt to chat history
        if isinstance(input, str) or isinstance(input, dict) or isinstance(input, ImageTypes):
            self.history.append(role='user', msg=input)
            chat_history = self.history
        elif isinstance(input, ChatHistory):
            chat_history = input  # TODO also recieve chat history as list for cross-process
        else:
            raise TypeError(f"ChatSession plugin expects inputs of type str, dict, image, or ChatHistory (was {type(input)})")

        # images should be followed by text prompts
        if chat_history[-1].is_type('image'):
            logging.debug("image message, waiting for user prompt")
            return
        
        # support both inline and multi-generation tools
        if chat_history.tool_style == 'openai':
            tool_functions = self.functions
            inline_functions = None
        else:
            tool_functions = None
            inline_functions = self.functions

        while True:
            # get the latest chat embeddings
            embedding, position = chat_history.embed_chat(
                max_tokens=self.model.config.max_length - self.max_new_tokens,
                wrap_tokens=self.wrap_tokens,
                use_cache=self.model.has_embed and chat_history.kv_cache,
            )
            
            # output vision features
            if chat_history.image_embedding is not None:
                self.output(chat_history.image_embedding, ChatSession.OutputImageEmbedding)
                
            # start generating output
            self.stream = self.model.generate(
                embedding, 
                streaming=True, 
                functions=inline_functions,
                kv_cache=chat_history.kv_cache,
                cache_position=position,
                stop_tokens=chat_history.template.stop,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                **kwargs
            )
            
            # output the stream iterator on channel 3
            self.output(self.stream, ChatSession.OutputStream)

            # output the generated tokens on channel 0
            bot_reply = chat_history.append(role='bot', text='', cached=True)
            words = ''
            
            for token in self.stream:
                if self.interrupted:
                    logging.debug(f"LLM interrupted, terminating request early")
                    self.stream.stop()
                    
                # sync the reply with the entire text, so that multi-token
                # unicode unicode sequences are detokenized and decoded together
                bot_reply.content = self.stream.text
                bot_reply.tokens = self.stream.tokens
                
                # output stream of raw tokens
                self.output(token, ChatSession.OutputToken)
                self.send_state()
                
                # if a space was added, emit new word(s)
                words += token
                last_space = words.rfind(' ')
                
                if last_space >= 0:
                    self.output(words[:last_space+1], ChatSession.OutputWords)
                    if last_space < len(words) - 1:
                        words = words[last_space+1:]
                    else:
                        words = ''
                
            if len(words) > 0:
                self.output(words, ChatSession.OutputWords)
                
            bot_reply.content = self.stream.text
            bot_reply.tokens = self.stream.tokens
            chat_history.kv_cache = self.stream.kv_cache
            self.stream = None
            
            # output the final generated text on channel 2
            self.output(bot_reply.content, ChatSession.OutputFinal)
            self.send_state()
            
            if self.print_stats:
                print_table(self.model.stats)
                
            # run bot functions
            if tool_functions is None:
                return
            
            from nano_llm import BotFunctions
               
            tool_response = BotFunctions.run(
                bot_reply.content, template=chat_history.template, functions=tool_functions
            )
            
            if tool_response:
                chat_history.append('tool_response', tool_response)
            else:
                return
                
            
