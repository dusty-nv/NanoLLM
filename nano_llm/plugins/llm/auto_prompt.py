#!/usr/bin/env python3
import re
import logging
import threading

from termcolor import cprint
from pprint import pformat

from nano_llm import Plugin
from nano_llm.web import WebServer
from nano_llm.utils import AttributeDict as AttrDict, load_prompts, is_image


class AutoPrompt(Plugin):
    """
    Apply prompting templates to incoming streams that form a kind of script.
    For example, "<img> Describe the image" will insert the most recent image.
    """
    def __init__(self, template : str = '<reset><image> Describe the image concisely.', **kwargs):
        """
        Apply prompting templates to incoming streams that form a kind of script.
        For example, "<img> Describe the image" will insert the most recent image.
    
        Args:
          template (str): The prompt template to follow, composed of <text> or <image> tags
                          (which will insert the most recent text or image message recieved)
                          along with interspersed text or chat commands like /reset and /pop
        """
        super().__init__(outputs='list', **kwargs)
        
        self.tag_format = "<{}>"
        
        self.tags = AttrDict(
            text = AttrDict(alias=['txt', 'msg']),
            image = AttrDict(alias=['img']),
            reset = AttrDict(alias=['clear'], command='/reset'),
        )
        
        self.delimiters = {}
        
        for key, tag in self.tags.items():
            tag.name = key
            tag.alias = [key] + tag.get('alias', [])
            for alias in tag.alias:
                self.delimiters[self.tag_format.format(alias)] = tag

        regex_pattern = '({})'.format('|'.join(map(re.escape, self.delimiters)))

        self.tag_regex = re.compile(regex_pattern)
        self.add_parameter('template', default=template)

    @property
    def template(self):
        return self._template
        
    @template.setter
    def template(self, template):
        '''
        template = template.replace('/reset', '<reset>')
        template = template.replace('/pop', '<pop>')
        
        for tag, aliases in self.tags.items():
            for alias in aliases:
                template = template.replace(f"<{alias}>", f"<{tag}>")
            
        for tag, var in self.vars.items():
            var.depth = template.count(f"<{tag}>")
        '''
        for tag in self.tags.values():
            tag.depth = 0
            tag.queue = []

        for delimiter, tag in self.delimiters.items():
            tag.depth += template.count(delimiter)
 
        counts = {}
        splits = self.tag_regex.split(template)
        
        self.splits = []
        
        for split in splits:
            if not split:
                continue
                
            tag = self.delimiters.get(split)
            
            if tag:
                self.splits.append(AttrDict(tag=tag, index=counts.setdefault(tag.name, 0)))
                counts[tag.name] += 1
            else:
                self.splits.append(AttrDict(text=split))
         
        self._template = template
        logging.debug(f"{self.name} set template to:  `{template}`\n{pformat(self.splits, indent=2)}")
                   
    @classmethod
    def type_hints(cls):
        return {'template': {'multiline': 4}}
          
    def process(self, input, **kwargs):
        if isinstance(input, str):
            tag = 'text'
        elif is_image(input):
            tag = 'image'
        else:
            raise TypeError(f"{self.name} expects to recieve str or image (was {type(input)})")
        
        tag = self.tags[tag]
            
        if tag.depth > 0:
            tag.queue.append(input)
            if len(tag.queue) > tag.depth:
                tag.queue = tag.queue[len(tag.queue)-tag.depth : ]

        for tag in self.tags.values():
            if len(tag.queue) < tag.depth and not tag.get('command'):
                logging.warning(f"{self.name} is waiting for {tag.depth - len(tag.queue)} more {tag.name} inputs")
                return

        msg = [None] * len(self.splits)
        is_text = [False] * len(self.splits)
        
        for index, split in enumerate(self.splits):
            if 'text' in split:
                msg[index] = split.text
                is_text[index] = True
            elif 'tag' in split:
                command = split.tag.get('command')
                if command:
                    msg[index] = command
                else:
                    msg[index] = split.tag.queue[split.index]
                    is_text[index] = isinstance(msg[index], str)

        i=1
        
        while i < len(msg):
            if is_text[i-1] and is_text[i]:
                msg[i-1] = msg[i-1] + msg[i]
                del msg[i]
            else:
                i += 1
                
        #print('AUTOPROMPT', pformat(msg, indent=2))
        return msg
        
        
if __name__ == "__main__":
    from nano_llm.utils import LogFormatter
    from jetson_utils import cudaImage
    
    LogFormatter.config(level='debug')
    
    auto_prompt = AutoPrompt(threaded=False)
    test_image = cudaImage(width=320, height=240, format='rgb8')
    
    print('TAGS', pformat(auto_prompt.tags, indent=2))
    print('DELIMITERS', list(auto_prompt.delimiters.keys()))

    def process(input):
        output = auto_prompt(input)
        print('\nINPUT', input, 'OUTPUT', pformat(output, indent=2), '\n')
    
    def test(template):
        auto_prompt.template = template
        process("This is a test")
        process(test_image)
        process("This is another test")
        process(test_image)
        process("This is a third test")
        process("This is a fourth test")
        process("This is a fifth test")
        
    test("<image> Describe the image concisely.<reset>")       
    test("<reset><image> Describe the image concisely.")
    test("<reset> <image> Describe the image concisely.")
    test("<reset><image>Describe the image concisely.")
    test("<reset><image><text>")
    test("<reset><image><text><text>")
    test("""<reset>These are the last few descriptions of the scene, from oldest to most recent.  Summarize any noteworthy changes, and if necessary, alert the user.

- <text>
- <text>
- <text>""")

