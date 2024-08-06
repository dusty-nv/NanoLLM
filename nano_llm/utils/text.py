#!/usr/bin/env python3
import sys
import time
import signal
import logging

from .keys import validate_key


def replace_text(text, dict):
    """
    Replace instances of the keys in dict found the text string with the values in dict.  For example::
    
        replace_text('The time is: ${TIME}', {'${TIME}': '12:30pm'})`` 
    
    This would return the string ``'The time is: 12:30pm'``
    """
    for key, value in dict.items():
        text = text.replace(key, value)
    return text    


def wrap_text(font, image, text='', x=5, y=5, stream=0, **kwargs):
    """"
    Utility for cudaFont that draws text on a image with word wrapping.
    Returns the new y-coordinate after the text wrapping was applied.
    """
    font_size = font.GetSize()
    text_color = validate_key(kwargs, 'color', font.White) 
    background_color = validate_key(kwargs, 'background', font.Gray40)
    line_spacing = validate_key(kwargs, 'line_spacing', font_size + 4)
    line_length = validate_key(kwargs, 'line_length', image.width // (font_size/2))

    if line_length < 0:
        font.OverlayText(image, text=text, x=x, y=y, color=text_color, background=background_color, stream=stream)
        return y + line_spacing
        
    text = text.split()
    current_line = ""

    for n, word in enumerate(text):
        if len(current_line) + len(word) <= line_length:
            current_line = current_line + word + " "
            
            if n == len(text) - 1:
                font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color, stream=stream)
                return y + line_spacing
        else:
            current_line = current_line.strip()
            font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color, stream=stream)
            current_line = word + " "
            y=y+line_spacing
    return y


def escape_html(text, code=False):
    """
    Apply escape sequences for HTML and other replacements (like '\n' -> '<br/>')
    If ``code=True``, then blocks of code will be surrounded by <code> tags.
    """
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#039;')
    text = text.replace('\\n', '\n')
    text = text.replace('\n', '<br/>')
    text = text.replace('\\"', '\"')
    text = text.replace("\\'", "\'")
    
    if code:
        text = code_tags(text)
        
    return text
    
    
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
 
