#!/usr/bin/env python3
from .args import *
from .audio import *
from .image import *
from .keyboard import *
from .log import *
from .model import *
from .prompts import *
from .request import WebRequest
from .table import *
from .tensor import *
from .inspection import *


def replace_text(text, dict):
    """
    Replace instances of each of the keys in dict in the text string with the values in dict
    """
    for key, value in dict.items():
        text = text.replace(key, value)
    return text    


class AttributeDict(dict):
    """
    A dict where keys are available as attributes:
    
      https://stackoverflow.com/a/14620633
      
    So you can do things like:
    
      x = AttributeDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'
      
    This is using the __getattr__ / __setattr__ implementation
    (as opposed to the more concise original commented out below)
    because of memory leaks encountered without it:
    
      https://bugs.python.org/issue1469629
      
    TODO - rename this to ConfigDict or NamedDict?
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value

'''    
class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
'''

    
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
    
    
def wrap_text(font, image, text='', x=5, y=5, **kwargs):
    """"
    Utility for cudaFont that draws text on a image with word wrapping.
    Returns the new y-coordinate after the text wrapping was applied.
    """
    text_color=kwargs.get("color", font.White) 
    background_color=kwargs.get("background", font.Gray40)
    line_spacing = kwargs.get("line_spacing", 38)
    line_length = kwargs.get("line_length", image.width // 16)

    text = text.split()
    current_line = ""

    for n, word in enumerate(text):
        if len(current_line) + len(word) <= line_length:
            current_line = current_line + word + " "
            
            if n == len(text) - 1:
                font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
                return y + line_spacing
        else:
            current_line = current_line.strip()
            font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
            current_line = word + " "
            y=y+line_spacing
    return y


def extract_code(text):
    """
    Extract code blocks delimited by braces (square and curly, like JSON).
    Returns a list of (begin, end) tuples with the indices of the blocks.
    """
    open_delim=0
    start=-1
    blocks=[]
    
    for i, c in enumerate(text):
        if c == '[' or c == '{':
            if open_delim == 0:
                start = i
            open_delim += 1
        elif c == ']' or c == '}':
            open_delim -= 1
            if open_delim == 0 and start >= 0:
                blocks.append((start,i+1))
                start = -1
                
    return blocks


def code_tags(text, blocks=None, open_tag='<code>', close_tag='</code>'):
    """
    Add code tags to surround blocks of code (i.e. for HTML presentation)
    Returns the text, but with the desired tags added around the code blocks.
    This works for JSON-like code with nested curly and square brackets.
    """
    if blocks is None:
        blocks = extract_code(text)
    
    if not blocks:
        return text

    offset = 0
    
    for start, end in blocks:
        text = text[:start+offset] + open_tag + text[start+offset:end+offset] + close_tag + text[end+offset:]
        offset += len(open_tag) + len(close_tag)
        
    return text 
   
   
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
 
 
def update_default(value, default=None):
    """
    If the value is None, return the default instead.
    """
    if isinstance(value, str):
        return value if value.strip() else default
    else:
        return default if value is None else value
     
