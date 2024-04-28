#!/usr/bin/env python3
import re
import logging

import functools
import traceback

from decorator_args import decorator_args


@decorator_args(optional=True)        
def bot_function(func, name=None, docs='pydoc', code='python', enabled=True):
    """
    Decorator for exposing a function to be callable by the LLM.
    This will create wrapper functions that do the parsing to
    determine if this function was called in the output text,
    and then interpret it to invoke the function call.
    
    Args:
    
      func (Callable):  The function to be called by the model.
      
      name (str):  The function name that the model should refer to.
                   By default, it will be the actual Python function name.
                   
      docs (str):  Description of the function that is added to the model's
                   system prompt.  By default, the Python docstring is used
                   from the function's code comment block (`''' docs here '''`)
                   
      code (str):  Language that the model is expected to write code in.
                   By default this is Python, but JSON will be added also.
         
      enabled (bool):  Boolean that toggles whether this function is added
                       to the system prompt and able to be called or not.
    """                   
    return BotFunctions.register(func, name=name, docs=docs, code=code, enabled=enabled)


class BotFunctions:
    """
    Manager of functions able to be called by the LLM that have been registered
    with the :func:`bot_function` decorator or :meth:`BotFunction.register`.
    
    This is a singleton class that is mostly supposed to be used like a function,
    where `BotFunction()` returns the list of currently active/enabled functions.
    """
    functions = []
    builtins = []
        
    def __new__(cls, all=False, load=True, test=True):
        """
        Return the list of enabled functions whenever `BotFunctions()` is called,
        making it seem like you are just calling a function that returns a list::
        
           for func in BotFunctions():
              func("SQRT(64)")
              
        If `all=True`, then even the disabled functions will be included.
        If `load=True`, then the built-in functions will be loaded (if they haven't yet been).
        If `test=True`, then the built-in functions will be tested (if they haven't yet been).
        """
        if load and not cls.builtins:
            cls.load(test=test)
            
        return cls.list(all=all)
        
    def __class_getitem__(cls, index):
        """
        Return the N-th registered function, like `BotFunctions[N]`
        """
        return cls.functions[index]
    
    @classmethod
    def len(cls):
        """
        Returns the number of all registered bot functions.
        """
        return len(cls.functions)
       
    @classmethod
    def list(cls, all=False):
        """
        Return the list of all enabled functions available to the bot.
        If `all=True`, then even the disabled functions will be included.
        """
        return cls.functions if all else [x for x in cls.functions if x.enabled]
       
    @classmethod
    def filter(cls, filters, mode='enable'):
        """
        Apply filters to the registered functions, either enabling or disabling them
        if their names are matched against the filter list.
        """
        for filter in filters:
            for function in cls.functions:
                if function.name == filter.lower():
                    function.enabled == mode.startswith('enable')
        return cls.list()
        
    @classmethod
    def generate_docs(cls):
        """
        Collate the documentation strings from all the enabled functions
        """
        docs = "You are able to call the Python functions defined below, "
        docs = docs + "and the returned values will be added to the chat.\n"
        docs = docs + '\n'.join([x.docs for x in cls.functions if x.enabled])
        
        return docs

    @classmethod
    def register(cls, func, name=None, docs='pydoc', code='python', enabled=True):
        """
        See the docs for :func:`bot_function`
        """
        name = name if name else func.__name__
        regex = re.compile(f"`{name}\(.*?\)`")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # this gets called from NanoLLM.generate() with `text` as the only arg,
            # which represents the entire response of the bot so far
            text = args[0]
            found_match = False
            for match in regex.finditer(text):
                # only evaluate what was just called (i.e. it is at the end of the reply)
                # also sometimes there are commas after the call inside the same token
                if match.end() >= len(text)-2:
                    found_match=True
                    break
                    
            if not found_match:
                return None
            
            code_str = args[0][match.start():match.end()].strip('`')

            try:
                logging.debug(f"Running generated code `{code_str}`")
                return eval(code_str, {name : func})
            except Exception as error:
                logging.error(f"Exception occurred executing generated code {code_str}\n\n{''.join(traceback.format_exception(error))}")
                return None

        wrapper.name = name if name else wrapper.__name__
        wrapper.docs = ''
        wrapper.enabled = enabled
        wrapper.function = func
        wrapper.regex = re.compile(f"`{wrapper.name}\(.*?\)`")
        
        if docs == 'nosig':
            docs = 'pydoc_nosig'
            
        if docs.startswith('pydoc'):
            if wrapper.__doc__:
                if docs == 'pydoc':
                    wrapper.docs = f"`{name}`() - " + wrapper.__doc__.strip()
                elif docs == 'pydoc_nosig':
                    wrapper.docs = wrapper.__doc__.strip()
            else:
                wrapper.docs = name + '() '
        elif docs:
            wrapper.docs = docs
            
        cls.functions.append(wrapper)
        func._bot_function = wrapper
        
        return func
        
    @classmethod
    def load(cls, test=True):
        """
        Load the built-in functions by importing their modules.
        """
        if cls.builtins:
            return True
            
        # TODO: automate this
        from . import clock
        from . import location
        from . import weather
        
        assert(cls.functions)
        cls.builtins = True
        return cls.test()
       
    @classmethod
    def test(cls, disable_on_error=True):
        """
        Test that the functions are able to be run, and disable them if not.
        Returns true if all tests passed, otherwise false.
        """
        logging.info(f"Bot function descriptions:\n{cls.generate_docs()}")
        logging.info("Testing bot functions:")
        
        had_error = False
        
        for function in cls.functions:
            try:
                logging.info(f"  * {function.name}() => '{function.function()}'  ({function.docs})")
            except Exception as error:
                logging.error(f"Exception occurred testing bot function {function.name}()\n\n{''.join(traceback.format_exception(error))}")
                had_error = True
                if disable_on_error:
                    function.enabled = False
        
        return had_error
                
                    
