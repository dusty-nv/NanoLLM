#!/usr/bin/env python3
import os
import time
import logging

from nano_llm import Plugin, ChatHistory, ChatTemplates, NanoLLM as NanoModel
from nano_llm.utils import ImageTypes, load_prompts, print_table


class NanoLLM(Plugin):
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
              channel 4 (list[dict]) -- the list of messages in the chat history
              channel 5 (None) -- plugins that provide tool functions that the model can call
    """
    OutputDelta = 0
    OutputPartial = 1
    OutputFinal = 2
    OutputWords = 3
    OutputHistory = 4
    OutputTools = 5
    
    def __init__(self, model: str="princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT", 
                 api: str="mlc", quantization: str="q4f16_ft", 
                 max_context_len: int=None, drop_inputs: bool=False,
                 chat_template: str=None, system_prompt: str=None, **kwargs):
        """
        Load an LLM and run generation on chat requests.
        
        Args:
          model (str): Either the path to the model, or HuggingFace model repo/name.
          api (str): The model backend to use (MLC - fastest, AWQ - accurate quantization, HF - compatability)
          quantization (str): For MLC: recommend q4f16_ft or 8f16_ft. For AWQ: the path to the quantized weights.
          max_context_len (str): The maximum chat length in tokens (by default, inherited from the model)  
          drop_inputs (bool): If true, only the latest message from the input queue will be used (older messages dropped)
          chat_template (str|dict): The chat template (by default, will attempt to determine from model type)
          system_prompt (str):  Set the system prompt (changing this will reset the chat)          
        """
        super().__init__(
            outputs=kwargs.pop('outputs', ['delta', 'partial', 'final', 'words', 'history', 'tools']), 
            drop_inputs=drop_inputs, 
            **kwargs
        )

        if max_context_len is not None and max_context_len <= 0:
            max_context_len = None
            
        load_time = time.perf_counter()
        
        if isinstance(model, str):
            self.model_name = model
            self.model = NanoModel.from_pretrained(
                model, api=api, 
                quantization=quantization, 
                max_context_len=max_context_len, 
                use_cache=True,
                **kwargs
            )
        else:
            self.model = model
            self.model_name = self.config.name

        try:
            system_prompt = load_prompts(system_prompt, concat=True)
        except Exception as error:
            self.send_alert(f"Failed to load system prompt from {system_prompt} ({error})", level="error")
            
        self.title = os.path.basename(self.model_name)
        self.stream = None

        self.history = ChatHistory(self.model, chat_template=chat_template, system_prompt=system_prompt, **kwargs)

        self.add_parameter('max_new_tokens', type=int, default=kwargs.get('max_new_tokens', 128), help="The number of tokens to output in addition to the prompt.")
        self.add_parameter('min_new_tokens', type=int, default=kwargs.get('min_new_tokens', -1), help="Force the model to generate a set number of output tokens (<0 to disable)")
        self.add_parameter('do_sample', type=bool, default=kwargs.get('do_sample', False), help="If true, temperature/top_p sampling will be used over the logits.")
        self.add_parameter('temperature', type=float, default=kwargs.get('temperature', 0.7), help="Randomness token sampling parameter (only used if do_sample=true)")
        self.add_parameter('top_p', type=float, default=kwargs.get('top_p', 0.95), help="If set to < 1 and do_sample=True, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.")
        self.add_parameter('repetition_penalty', type=float, default=kwargs.get('repetition_penalty', 1.0), help="The parameter for repetition penalty. 1.0 means no penalty")
        self.add_parameter('drop_inputs', default=drop_inputs)
        self.add_parameter('system_prompt', default=self.history.system_prompt)

        #self.add_parameter('tool_docs', type=str, read_only=True, controls=False)

        self.max_context_len = self.model.config.max_length
        self.wrap_tokens = kwargs.get('wrap_tokens', 512)
        self.print_stats = kwargs.get('print_stats', True) #kwargs.get('debug', False))
        
        warmup = True
        
        if warmup:
            warmup = warmup if isinstance(warmup, str) else 'What is 2+2?'
            self.history.append(role='user', text=warmup)
            logging.info(f"Warming up LLM with query '{warmup}'")
            logging.info(f"Warmup response:  '{self.model.generate(self.history.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
            
        self.reset()
                      
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

    @property
    def system_prompt(self):
        return self._system_prompt
        
    @system_prompt.setter
    def system_prompt(self, value):
        try:
            value = load_prompts(value, concat=True)
        except Exception as error:
            self.send_alert(f"Failed to load system prompt from {value} ({error})", level="error")
            
        self._system_prompt = value
        self.reset()

    @property
    def toolset(self):
        tools = {}
        
        for output in self.outputs[NanoLLM.OutputTools]:
            tools.update(output.tools)
        
        #for function in BotFunctions():
        #    tools[function.name] = function
          
        return tools

    @property
    def tool_docs(self):
        if self.tool_spec == 'openai':
            return str([x.openai for x in self.toolset.values() if x.enabled])
        elif self.tool_spec == 'python':
            return '\n'.join(['* ' + x.docs for x in self.toolset.values() if x.enabled])
        else:
            return ''
            
    @property
    def tool_spec(self):
        return self.history.template.tool_spec
            
    @classmethod
    def type_hints(cls):
        return {
            'model': {
                'suggestions': [
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    "NousResearch/Hermes-2-Pro-Llama-3-8B",
                    "princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT",
                    "Efficient-Large-Model/VILA1.5-3b",
                    "Efficient-Large-Model/Llama-3-VILA1.5-8B",
                ]
            },
            
            'api': {
                'display_name': 'API',
                'options': ['MLC', 'AWQ', 'HF'],
            },
            
            'quantization': {
                'suggestions': ['q3f16_0', 'q3f16_1', 'q4f16_0', 'q4f16_1', 'q4f16_2', 'q4f16_ft', 'q4f16_ft_group', 'q4f32_0', 'q4f32_1', 'q8f16_ft', 'q8f16_ft_group', 'q8f16_1'],
            },
            
            'chat_template': {
                'suggestions': ['auto'] + list(ChatTemplates.keys())
            },
            
            'system_prompt': {
                'multiline': 3,
            },
            
            'max_context_len': {
                'display_name': 'Max Context Length',
            },
        }

    def state_dict(self, **kwargs):
        return {**super().state_dict(**kwargs), 'model': self.model_name}
     
    def process(self, input, partial=False, **kwargs):
        """
        Generate the reply to a prompt or the latest ChatHistory.
        
        Parameters:
        
          input (str|dict|ChatHistory) -- either a string prompt from the user,
                                          a ChatEntry dict, or ChatHistory dict.
                                          
        Returns:
        
          The generated text (token by token), if input was a string or dict.
          If input was a ChatHistory, returns the streaming iterator/generator.
        """
        #self.apply_config(**kwargs)
        
        if input is None:
            return
            
        if isinstance(input, list):
            for i, x in enumerate(input):
                self.process(x, queue_depth=(len(input)-i-1), **kwargs)
            return

        if self.interrupted:
            return

        # handle some special commands
        if isinstance(input, str):
            x = input.lower()
            if any([x == y for y in ('/reset', '/clear', '<reset>', '<clear>')]):
                self.reset(send_history=(kwargs.get('queue_depth', 0) == 0))
                return
            input = self.apply_substitutions(input)
            if any([x == y for y in ('/refresh', '<refresh>')]):
                self.send_history()
                return
                
        # add prompt to chat history
        if isinstance(input, str) or isinstance(input, dict) or isinstance(input, ImageTypes):
            if partial:
                self.send_history(append=dict(role='user', text=str(input), partial=True))
                return
            else:
                self.history.append(role='user', msg=input)
        else:
            raise TypeError(f"ChatModel plugin expects inputs of type str, dict, image, or ChatHistory (was {type(input)})")

        # images should be followed by text prompts
        if self.history[-1].is_type('image'):
            logging.debug("image message, waiting for user prompt")
            return
        
        # support both inline and multi-generation tools
        if self.tool_spec == 'openai':
            tool_functions, inline_functions = self.toolset, None
        elif self.tool_spec == 'python':
            tool_functions, inline_functions = None, self.toolset
        else:
            tool_functions, inline_functions = None, None

        while True:
            # get the latest chat embeddings
            embedding, position = self.history.embed_chat(
                max_tokens=self.model.config.max_length - self.max_new_tokens,
                wrap_tokens=self.wrap_tokens,
                use_cache=self.model.has_embed and self.history.kv_cache,
            )

            # start generating output
            self.stream = self.model.generate(
                embedding, 
                streaming=True, 
                functions=inline_functions,
                kv_cache=self.history.kv_cache,
                cache_position=position,
                stop_tokens=self.history.template.stop,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                **kwargs
            )

            # output the generated tokens on channel 0
            bot_reply = self.history.append(role='bot', text='', cached=True)
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
                self.output(token, NanoLLM.OutputDelta, delta=True, partial=True)
                self.output(bot_reply.content, NanoLLM.OutputPartial, partial=True)
                self.send_history(partial=True)

                # if a space was added, emit new word(s)
                words += token
                last_space = words.rfind(' ')
                
                if last_space >= 0:
                    self.output(words[:last_space+1], NanoLLM.OutputWords, delta=True, partial=True)
                    if last_space < len(words) - 1:
                        words = words[last_space+1:]
                    else:
                        words = ''
                
                # update the web stats
                num_chat_tokens = self.history.num_tokens
                
                self.send_stats(
                    chat_tokens=num_chat_tokens, 
                    prefill_time=self.model.stats.prefill_time, 
                    decode_rate=self.model.stats.decode_rate,
                    summary=[f"{num_chat_tokens} tokens", f"{self.model.stats.decode_rate:.1f} tps"],
                )
                
            if len(words) > 0:
                self.output(words, NanoLLM.OutputWords, delta=True, partial=True)
                
            bot_reply.content = self.stream.text
            bot_reply.tokens = self.stream.tokens
            self.history.kv_cache = self.stream.kv_cache
            self.stream = None

            # output the final generated text on channel 2
            self.output(bot_reply.content, NanoLLM.OutputFinal) 
            self.send_history()
            
            if self.print_stats:
                print_table(self.model.stats)
                
            # run bot functions
            if not tool_functions:
                return
                
            tool_response = self.history.run_tools(bot_reply, tools=tool_functions)
            
            if tool_response is None:
                return

            if self.outputs[NanoLLM.OutputHistory]:
                self.output(self.history.to_list(html=True), NanoLLM.OutputHistory)
                

    def start(self):
        """
        Reset the chat on plugin start so references to other plugins are resolved
        for variable substitution (this gets called after all plugins have been created)
        """
        super().start()
        self.reset()

    def reset(self, send_history=True):
        """
        Reset the chat history and re-apply variable substitution to the system prompt
        """
        self.history.reset(system_prompt=self.apply_substitutions(self._system_prompt))
        if send_history:
            self.send_history()
        
    def send_history(self, html=True, append=None, **kwargs):
        """
        Output the chat history (typically with HTML formatting applied for web clients)
        """
        if len(self.outputs) <= NanoLLM.OutputHistory:
             return
             
        if len(self.outputs[NanoLLM.OutputHistory]) == 0:
            return
            
        history = self.history.to_list(html=html)
        
        if append:
            if not isinstance(append, list):
                append = [append]
            history.extend(append)
            
        self.output(history, NanoLLM.OutputHistory, **kwargs)
                       
            
