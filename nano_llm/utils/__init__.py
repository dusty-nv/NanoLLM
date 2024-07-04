#!/usr/bin/env python3
from nanodb.utils import *

from .args import *
from .audio import *
from .inspection import *
from .keyboard import *
from .model import *
from .prompts import *
from .text import *
from .request import WebRequest


def ends_with_token(input, tokens, tokenizer=None):
    """
    Check to see if the list of input tokens ends with any of the list of stop tokens.
    This is typically used to check if the model produces a stop token like </s> or <eos>
    """
    if not isinstance(input, list):
        input = [input]
        
    if not isinstance(tokens, list):
        tokens = [tokens]
     
    if len(input) == 0 or len(tokens) == 0:
        return False
        
    for stop_token in tokens:
        if isinstance(stop_token, list):
            if len(stop_token) == 1:
                if input[-1] == stop_token[0]:
                    return True
            elif len(input) >= len(stop_token):
                if tokenizer:
                    input_text = tokenizer.decode(input, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    stop_text = tokenizer.decode(stop_token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    #print('input_text', input_text, 'stop_text', f"'{stop_text}'")
                    if input_text.endswith(stop_text):
                        #print('STOPPING TEXT')
                        return True
                else:
                    if input[-len(stop_token):] == stop_token:
                        return True
        elif input[-1] == stop_token:
            return True
            
    return False
    
   
def filter_keys(dictionary, keep=None, remove=None):
    """
    Remove keys from a dict by either a list of keys to keep or remove.
    """
    if isinstance(dictionary, list):
        for x in dictionary:
            filter_keys(x, keep=keep, remove=remove)
        return dictionary
        
    for key in list(dictionary.keys()):
        if (keep and key not in keep) or (remove and key in remove):
            del dictionary[key]
 
    return dictionary
 
def update_default(value, default=None, cast=None):
    """
    If the value is None, return the default instead.
    """
    if isinstance(value, str):
        value = value if value.strip() else default
    else:
        value = default if value is None else value
        
    if cast is not None:
        value = cast(value)
        
    return value
    
     
