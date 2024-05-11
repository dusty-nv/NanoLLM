#!/usr/bin/env python3

class KVCache():
    """
    Abstract interface for storing & manipulating the KV cache,
    which encodes all the context and model state in the chat.
    These are implemented by different LLM backends and are
    backed by CUDA memory for each layer in the model, which
    these functions provide some modifications.  
    
    It gets returned in the :class:`StreamingResponse` iterator
    from :meth:`NanoLLM.generate()` and as an optional argument
    can be re-used during the next generation to grow the cache
    instead of having to refill the chat context each request.
    
    For example, :meth:`KVCache.pop` will drop the most recent
    N tokens off the end of the cache, while :meth:`KVCache.remove`
    while remove a range of tokens from anywhere in the cache.
    
    The :class:`ChatHistory` object provides a higher-level way
    of maintaining consistency for removing messages from the chat
    by keeping track of their token counts and positions in the chat.
    It also keeps the KV cache between requests, so that only the
    new tokens need to be added (and the model only processes those).
    """
    def __init__(self):
        super().__init__()
        
        #: The current length of the KV cache
        self.num_tokens = 0
  
    def __len__(self):
        """
        Return the current length of the cache in terms of tokens or embedding positions.
        """
        return self.num_tokens
    
    def pop(self, tokens):
        """
        Remove the given number of tokens from the end of the cache.
        """
        raise NotImplementedError(f"{type(self)} did not implement pop()")

    def remove(self, start, stop, sync=True):
        """
        Remove a range of tokens from the cache, from the start index (inclusive) to the stop index (exclusive)
        """
        raise NotImplementedError(f"{type(self)} did not implement pop()")
               
   
