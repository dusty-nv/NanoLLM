#!/usr/bin/env python3
import re
import json
import requests
import logging
import traceback

from nano_llm import NanoLLM, ChatHistory, Plugin
from nano_llm.utils import ArgParser, load_prompts, print_table

from termcolor import cprint

from functools import wraps  
from datetime import datetime
from decorator_args import decorator_args

def TIME(text):
    if text.endswith('`TIME()`'):
        return datetime.now().strftime("%-I:%M %p")


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
    with the :func:`bot_function` decorator or :meth:`BotFunction.register`
    """
    functions = []
        
    def __new__(cls, all=False):
        """
        Return the list of enabled functions whenever `BotFunctions()` is called,
        making it seem like you are just calling a function that returns a list::
        
           for func in BotFunctions():
              func("SQRT(64)")
              
        If `all=True`, then even the disabled functions will be included.
        """
        return cls.functions if all else [x for x in cls.functions if x.enabled]
        
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
    def list(cls):
        """
        Return the list of all registered functions available.
        """
        return cls.functions
        
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
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # this gets called from NanoLLM.generate() with `text` as the only arg,
            # which represents the entire response of the bot so far
            text = args[0]
            found_match = False
            for match in regex.finditer(text):
                if match.end() == len(text): # only evaluate what was just called
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
        
        if docs == 'pydoc':
            wrapper.docs = name + '() - ' + wrapper.__doc__
        elif docs == 'pydoc_nosig':
            wrapper.docs = wrapper.__doc__
            
        cls.functions.append(wrapper)
        func._bot_function = wrapper
        
        return func
        

@bot_function(docs="returns the current date in MM/DD/YYYY format")
def DATE():
    return datetime.now().strftime("%-m/%d/%Y")

@bot_function
def LOCATION():
    """ Returns the current location, like the name of the city. """
    x = requests.get(f'http://ip-api.com/json').json()  # zip, lat/long, timezone, query (IP)
    #return json.dumps(x, indent=2)
    #return x.get('city', x.get('regionName', x.get('country')))
    return f"{x.get('city')}, {x.get('regionName')}"
    
print('LOCATION', LOCATION())
    
def inspect_func(func):
    print('Function __name__', func.__name__)
    print('Function __docs__', func.__doc__)
    print('Function name:', func.name)
    print('Function docs:', func.docs)
    print('Function enabled:', func.enabled)
    print('Function inner:', func.function)

print('num BotFunctions', len(BotFunctions()))#len(BotFunctions))

for func in BotFunctions():
    inspect_func(func)
    
print('\n\n' + BotFunctions.generate_docs())

print([TIME] + BotFunctions())


# parse args and set some defaults
parser = ArgParser(extras=ArgParser.Defaults + ['prompt'])
parser.add_argument('--disable-plugins', action='store_true')
args = parser.parse_args()

prompts = load_prompts(args.prompt)

if not prompts:
    prompts = [
        "You have some functions available to you like `TIME()`, `DATE()`, and `LOCATION()` which will add those to the chat - please call them when required.  `LOCATION()` will return the closest city and state.  \nHi, how are you today?", 
        "What month is it?",
        "What time is it?",
        #"Where are we?",
        "What city are we in?",
    ]
    
if not args.model:
    args.model = "meta-llama/Meta-Llama-3-8B-Instruct"

print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

while True: 
    # get the next prompt from the list, or from the user interactivey
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
            cprint(f'>> PROMPT: {user_prompt}', 'blue')
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    print('')
    
    # special commands:  load prompts from file
    # 'reset' or 'clear' resets the chat history
    if user_prompt.lower().endswith(('.txt', '.json')):
        user_prompt = ' '.join(load_prompts(user_prompt))
    elif user_prompt.lower() == 'reset' or user_prompt.lower() == 'clear':
        logging.info("resetting chat history")
        chat_history.reset()
        continue

    # add the latest user prompt to the chat history
    entry = chat_history.append(role='user', msg=user_prompt)

    # get the latest embeddings (or tokens) from the chat
    embedding, position = chat_history.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        return_tokens=not model.has_embed,
    )

    # generate bot reply
    reply = model.generate(
        embedding, 
        plugins=[TIME] + BotFunctions() if not args.disable_plugins else None, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    # stream the output
    for token in reply:
        print(token, end='', flush=True)
        #cprint(f"NEW TOKEN '{token}'", "yellow")
        
    print('\n')
    cprint(reply.text, 'green')
    
    print_table(model.stats)
    print('')
    
    chat_history.append(role='bot', text=reply.text) # save the output
    chat_history.kv_cache = reply.kv_cache           # save the KV cache 