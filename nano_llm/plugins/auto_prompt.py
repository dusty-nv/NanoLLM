#!/usr/bin/env python3
import sys
import threading

from termcolor import cprint

from nano_llm import Plugin
from nano_llm.web import WebServer
from nano_llm.utils import AttributeDict, load_prompts, is_image


class AutoPrompt(Plugin):
    """
    Apply prompting templates to incoming stream of data that form a kind of script.
    For example, "<img> Describe the image" will insert the most recent image.
    """
    def __init__(self, template : str = '<text>', **kwargs):
        """
        Parameters:
        
          template (str): The prompt template to follow, composed of <text> or <image> tags
                          (which will insert the most recent text or image message recieved)
                          along with interspersed text or chat commands like /reset and /pop
        """
        super().__init__(outputs='list', **kwargs)
        
        self.tags = {'text': ['txt', 'msg'], 'image': ['img']}
        self.vars = {tag: AttributeDict(queue=[], depth=0) for tag in self.tags}
        
        self.add_parameter('template', default=template)

    @property
    def template(self):
        return self._template
        
    @template.setter
    def template(self, template):
        for tag, aliases in self.tags.items():
            for alias in aliases:
                template = template.replace(f"<{alias}>", f"<{tag}>")
            
        for tag, var in self.vars.items():
            var.depth = template.count(f"<{tag}>")
        
        self._template = template
        
    @classmethod
    def type_hints(cls):
        return {'template': {'multiline': 4}}
          
    def process(self, input, **kwargs):
        if isinstance(input, str):
            tag = 'text'
        elif isimage(input):
            tag = 'image'
        else:
            raise TypeError(f"{self.name} expects to recieve str or image (was {type(input)})")
            
        var = self.vars[tag]
        
        if var.depth > 0:
            var.queue.append(input)
            if len(var.queue) > var.depth:
                var.queue = var.queue[len(var.queue)-var.depth : ]
        
        template = self.template
        
        for i in range(self.vars['text'].depth):
            template = template.replace('<text>', self.vars['text'].queue[i])
        
        for i, text in enumerate(template.split('<image>')):
            if text:
                msg.append(text)
            msg.append(self.vars['image'].queue[i])

        self.output(msg)
    
