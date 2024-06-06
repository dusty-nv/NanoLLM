#!/usr/bin/env python3
import sys
import logging
import threading

from termcolor import cprint

from nano_llm import Plugin
from nano_llm.web import WebServer
from nano_llm.utils import AttributeDict, load_prompts, is_image


class AutoPrompt(Plugin):
    """
    Apply prompting templates to incoming streams that form a kind of script.
    For example, "<img> Describe the image" will insert the most recent image.
    """
    def __init__(self, template : str = '<image> Describe the image concisely.', **kwargs):
        """
        Apply prompting templates to incoming streams that form a kind of script.
        For example, "<img> Describe the image" will insert the most recent image.
    
        Args:
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
        print('AUTOPROMPT TEMPLATE', self._template)
        
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
            
        var = self.vars[tag]
        
        if var.depth > 0:
            var.queue.append(input)
            if len(var.queue) > var.depth:
                var.queue = var.queue[len(var.queue)-var.depth : ]
            
        def check_depth(tag):
            if len(self.vars[tag].queue) < self.vars[tag].depth:
                logging.warning(f"{self.name} is waiting for {self.vars[tag].depth - len(self.vars[tag].queue)} more <{tag}> inputs")
                return False
            return True
         
        for tag in self.tags:
            if not check_depth(tag):
                return
                
        msg = []
        template = self.template
           
        for i in range(self.vars['text'].depth):
            template = template.replace('<text>', self.vars['text'].queue[i], 1)
        
        for i, text in enumerate(template.split('<image>')):
            if text:
                msg.append(text)
            if i < len(self.vars['image'].queue):
                msg.append(self.vars['image'].queue[i])

        from pprint import pprint
        print('AUTOPROMPT')
        pprint(msg, indent=2)
        
        self.output(msg)
    
