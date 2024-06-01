#!/usr/bin/env python3
from nano_llm import Plugin
from nano_llm.utils import function_has_kwargs

class Callback(Plugin):
    """
    Wrapper for calling a function with the same signature as Plugin.process()
    This is automatically used by Plugin.add() so it's typically not needed.
    Callbacks are threaded by default and will be run asynchronously.
    If it's a lightweight non-blocking function, you can set threaded=False
    """
    def __init__(self, function, threaded=False, **kwargs):
        """
        Parameters:
          function (callable) -- function for processing data like Plugin.process() would
        """
        super().__init__(threaded=threaded, **kwargs)
        
        self.function = function
        self.has_kwargs = function_has_kwargs(function)
        
    def process(self, input, **kwargs):
        if self.has_kwargs:
            self.function(input, **kwargs)
        else:
            self.function(input)
