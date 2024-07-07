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
        
        #: the new text or tokens added since the iterator was last read
        self.delta = '' if self.detokenize else []
        
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
        
        self.event = threading.Event()
        self.kwargs = kwargs
        
    def __iter__(self):
        return self

    def __next__(self):
        """
        Wait until the model generates more output, and return the new text (only the delta)
        """
        if self.stopped:
            '''
            # early-stop EOS token is now added inside LLM APIs
            stop_tokens = self.kwargs.get('stop_tokens', [self.model.tokenizer.eos_token_id])
            if not ends_with_token(self.tokens, stop_tokens, self.model.tokenizer):
                self.add_tokens(self.model.tokenizer.eos_token_id) # add EOS if necessary
                return self._pop_delta()
            '''
            delta = self._pop_delta()
            
            if delta:
                return delta
            else:
                raise StopIteration
            
        self.event.wait()
        self.event.clear()

        return self._pop_delta()
     
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
                return self
            self.event.wait()
            self.event.clear()
            
    def add_tokens(self, tokens, detokenize=True, event=False):
        """
        Add an output token, detokenize the reply, and accumulate the delta message.
        This function is only used by the model APIs when they generate a new token.
        """
        if isinstance(tokens, (torch.Tensor, np.ndarray)):
            tokens = tokens.squeeze().tolist()
        
        if not isinstance(tokens, list):
            tokens = [tokens]
            
        self.tokens.extend(tokens)    

        if not detokenize:
            return
            
        if self.detokenize:
            # detokenize the entire reply on each new output token, because multiple tokens can
            # combine with each other, changing the previous text (like with long words and unicode)
            message = self.model.tokenizer.decode(self.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            self.delta = self.delta + message[len(self.text):]
            self.text = message
        else:
            self.delta.extend(tokens)
            
        if event:
            self.event.set()

    def _pop_delta(self, reset=True):
        """
        Get the tokens that have accumulated since the iterator was last read, and reset it.
        """
        delta = self.delta
        
        if reset:
            self.delta = '' if self.detokenize else []
            
        return delta
