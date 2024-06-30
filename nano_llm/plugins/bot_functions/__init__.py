#!/usr/bin/env python3
import re
import json
import logging

import functools
import traceback

from decorator_args import decorator_args
from ...utils import inspect_function


@decorator_args(optional=True)        
def bot_function(func, name=None, docs=None, enabled=True):
    """
    Decorator for exposing a function to be callable by the LLM.
    This will create wrapper functions that do the parsing to
    determine if this function was called in the output text,
    and then interpret it to invoke the function call.
    Text returned from these functions will be added to the chat.
    
    For example, this definition will expose the ``TIME()`` function to the bot::
    
        @bot_function
        def TIME():
            ''' Returns the current time. '''
            return datetime.now().strftime("%-I:%M %p")
    
    You should then add instructions for calling it to the system prompt so that
    the bot knows it's available. :meth:`BotFunctions.generate_docs` can automatically
    generate the function descriptions for you from their Python docstrings, which
    you can then add to the chat history.
            
    Args:
      func (Callable):  The function to be called by the model.
      name (str):  The function name that the model should refer to.
                   By default, it will be the actual Python function name.
      docs (str):  Description of the function that overrides its pydoc string.
      enabled (bool):  Boolean that toggles whether this function is added
                       to the system prompt and able to be called or not.
    """                   
    return BotFunctions.register(func, name=name, docs=docs, enabled=enabled)


class BotFunctions:
    """
    Manager of functions able to be called by the LLM that have been registered
    with the :func:`bot_function` decorator or :meth:`BotFunctions.register`.
    This is a singleton that is mostly intended to be used like a list,
    where ``BotFunction()`` returns the currently enabled functions.
    
    You can pass these to :meth:`NanoLLM.generate`, and they will be called inline 
    with the generation::
    
        model.generate(
            BotFunctions().generate_docs() + "What is the date?", 
            functions=BotFunctions()
        )
        
    :meth:`BotFunctions.generate_docs` will automatically generate function descriptions
    from their Python docstrings.  You can filter and disable functions with :meth:`BotFunctions.filter`
    """
    functions = []
    builtins = []
        
    def __new__(cls, all=False, load=True, test=False):
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
    def find(cls, name, functions=None):
        """
        Find a function by name, or return None if not found
        """
        if functions is None:
            functions = cls.functions
        
        if isinstance(functions, dict):
            return functions.get(name)
                
        for function in functions:
            if function.name == name:
                return function
    
    @classmethod
    def generate_docs(cls, prologue=True, epilogue=True, spec='python', functions=None):
        """
        Collate the documentation strings from all the enabled functions
        """
        if functions is None:
            functions = BotFunctions()
            
        if not isinstance(prologue, str):
            if prologue is None or prologue == False:
                prologue = ''
            elif prologue == True:
                if spec == 'python':
                    prologue = "You are able to call the Python functions defined below, and the returned values will be added to the chat:\n\n"
                else:
                    prologue = ''
                    
        if not isinstance(epilogue, str):
            if epilogue is None or epilogue == False:
                epilogue = ''
            elif epilogue == True:
                if spec == 'python':
                    epilogue = "\n\nFor example, if the user asks for the temperature, call the WEATHER() function."
                else:
                    epilogue = ''
                    
        if spec == 'python':          
            docs = '\n'.join(['* ' + x.docs for x in functions if x.enabled])
        elif spec == 'openai':
            docs = str([x.docs for x in functions if x.enabled])
        else:
            raise ValueError(f"supported function-calling tool specifications are 'openai' and 'python' (was spec={spec})")
                
        if prologue:
            docs = prologue + docs
            
        if epilogue:
            docs = docs + epilogue
            
        return docs

    @classmethod
    def register(cls, func, name=None, docs=None, enabled=True):
        """
        See the docs for :func:`bot_function`
        """
        name = name if name else func.__name__
        regex = re.compile(f"{name}\(.*?\)")
        
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
                logging.error(f"Exception occurred executing generated code {code_str}\n\n{traceback.format_exc()}")
                return None

        if not docs:
            if wrapper.__doc__:
                docs = f"`{name}()` - " + wrapper.__doc__.strip()
            else:
                docs = name + '() '
                
        wrapper.name = name
        wrapper.docs = docs
        wrapper.spec = None
        wrapper.enabled = enabled
        wrapper.function = func
        wrapper.regex = regex
        wrapper.openai = inspect_function(func, return_spec='openai')

        cls.functions.append(wrapper)
        func._bot_function = wrapper

        return func
    
    @classmethod
    def run(cls, text, template=None, functions=None):
        """
        Invoke any function calls in the output text and return the results.
        """
        if not text:
            return None
            
        if functions is None:
            functions = BotFunctions()
            
        if template is None:
            # inline-style python/JSON
            function_text = ''
            for function in functions:
                output = function(text)
                if not output:
                    continue
                function_text = ' ' + output
            return function_text if function_text else None

        if template.tool_spec != 'openai':
            raise RuntimeError("to run tools, the chat template needs to have keys for 'tool_call' and 'tool_response'")
         
        if 'tool_regex' not in template:
            template.tool_regex = re.compile(template.tool_call, flags=re.DOTALL) # allow for newlines in matches
                  
        def parse_tools(text):
            try:
                for match in template.tool_regex.finditer(text):
                    return json.loads(match.group(1)) # extract what is inside any prefix/postfix tags
            except Exception as error:
                logging.warning(f"Exception occurred trying to parse tool calls from bot reply:\n\n```{text}```\n\n{traceback.format_exc()}")             

        call = parse_tools(text)
        
        if call is None:
            return None
             
        logging.debug(f'invoking tool call {call}')
               
        if template.tool_spec == 'openai':
            func_name = call['name']
            func_args = call.get('arguments', {})
        else:
            raise ValueError("Tool calling is only currently supported with openai format")
         
        func = cls.find(func_name, functions=functions)
            
        if not func or not func.enabled:
            error = f"Error: could not find tool named '{func_name}'"
            logging.error(error)
            return error
                
        try:
            response = func.function(**func_args)
        except Exception as error:
            error = f"Exception occurred running tool {func_name}({func_args}) - {error}"
            logging.error(f"{error}\n\n{traceback.format_exc()}")
            return error
        
        if response is None:
            return None
                    
        if template.tool_spec == 'openai':  
            response = json.dumps({"name": func_name, "content": response})

        return response
            
    @classmethod
    def load(cls, test=True):
        """
        Load the built-in functions by importing their modules.
        """
        if cls.builtins:
            return True
            
        # TODO: automate this
        from . import alert
        from . import clock
        from . import location
        #from . import weather
        #from . import home_assistant
        
        assert(cls.functions)
        cls.builtins = True
        
        if test:
            cls.test()
            
        return cls.functions
       
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
                logging.error(f"Exception occurred testing bot function {function.name}()\n\n{traceback.format_exc()}")
                had_error = True
                if disable_on_error:
                    function.enabled = False
        
        return had_error
                
                    
