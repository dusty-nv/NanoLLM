#!/usr/bin/env python3
import threading

import torch
import numpy as np

from nano_llm.utils import ends_with_token


class StreamingResponse():
    """
    Asynchronous output iterator returned from :meth:`NanoLLM.generate`.
    Use it to stream the reply from the LLM as they are decoded token-by-token::
    
        response = model.generate("Once upon a time,")
        
        for token in response:
            print(token, end='', flush=True)
    
    The entire response generated so far is also stored in :attr:`StreamingResponse.tokens`
    and :attr:`StreamingResponse.text`. To terminate processing prematurely, call :meth:`StreamingResponse.stop`,
    which will signal the model to stop from generating additional output tokens.
    """
    def __init__(self, model, input, **kwargs):
        super().__init__()
        
        #: accumulated output tokens generated so far (for the whole reply)
        self.tokens = []  
        
        #: detokenize and return text if true, otherwise token ID's
        self.detokenize = kwargs.get('detokenize', True)
        
        #: detokenized output text generated so far (for the whole reply)
        self.text = ''    

        #: how many tokens or characters have been read by the iterator so far
        self.read = 0
        
        #: the original input query from the user
        self.input = input
        
        #: the :class:`NanoLLM` model instance being used to generate the output
        self.model = model
        
        #: the :class:`KVCache` used by this request
        self.kv_cache = kwargs.get('kv_cache', None)

        #: set if the user requested early termination
        self.stopping = False  
        
        #: set when generation has actually stopped
        self.stopped = False   

        #: event that triggers when new tokens are added
        self.event = threading.Event()
        
        #: the original arguments the generation request was created with
        self.kwargs = kwargs
        
    def __iter__(self):
        return self

    def __next__(self):
        """
        Wait until the model generates more output, and return the new text (only the delta message)
        If :attr:``StreamingResponse.detokenize`` is False, then the list of tokens is returned.
        """
        def decode():
            delta = self.decode()
            
            if delta is not None:
                return delta
            else:
                raise StopIteration
                
        if self.stopped:
            return decode()
            
        self.event.wait()
        self.event.clear()

        return decode()
     
    @property
    def eos(self):
        """
        Returns true if End of Sequence (EOS) and generation has stopped.
        """
        return self.stopped
        
    def stop(self):
        """
        Signal the model to halt output generation before the end of the reply.
        """
        self.stopping = True

    def wait(self):
        """
        Wait for the generation to be over and the full response complete.
        """
        while True:
            if self.stopped:
                return self.decode()
            self.event.wait()
            self.event.clear()
            
    def add_tokens(self, tokens, event=False):
        """
        Add an output token, detokenize the reply, and accumulate the delta message.
        This function is only used by the model APIs when they generate a new token.
        """
        if isinstance(tokens, (torch.Tensor, np.ndarray)):
            tokens = tokens.squeeze().tolist()
        
        if not isinstance(tokens, list):
            tokens = [tokens]
            
        self.tokens.extend(tokens)    

        if event:
            self.event.set()

    def decode(self):
        """
        Retrieve the message text or tokens that have accumulated since the iterator was last read, 
        and detokenize them. The entire reply is detokenized on each new output token, because 
        multiple tokens can combine, changing the previous text (like with long words and unicode)
        If :attr:``StreamingResponse.detokenize`` is False, then that's skipped and tokens are returned.
        If this stream originates from a :class:`VLAModel`, then the actions are decoded and returned.
        """  
        read = self.read
        self.read = len(self.tokens)
        
        if self.read == read:
            return None

        action_space = self.kwargs.get('action_space')
        
        if self.model.vla and action_space:
            delta = self.actions = self.model.vla.decode_action(self.tokens, action_space=action_space)
        elif self.detokenize:
            read = len(self.text)
            delta = self.text = self.model.tokenizer.decode(self.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            delta = self.tokens
        
        if len(delta) == 0 or read == 0:
            return delta

        return delta[read:]
        
