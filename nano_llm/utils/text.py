#!/usr/bin/env python3
import sys
import time
import signal
import logging


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
   
   
