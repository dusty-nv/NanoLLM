#!/usr/bin/env python3
import os
import time
import logging

from .nano_llm import NanoLLM
from nano_llm.utils import ImageExtensions, is_image, load_image


class NanoVLA(NanoLLM):
    """
    Plugin for Vision/Language Action models that takes image/text inputs and outputs actions.
    Currently supports openvla-7b - action space (x, y, z, roll, pitch, yaw, gripper)
    These are deltas, except the gripper is absolute (0=open, 1=closed)
    """
    def __init__(self, model: str="openvla/openvla-7b", 
                 api: str="mlc", quantization: str="q4f16_ft", 
                 max_context_len: int=384, drop_inputs: bool=True,
                 chat_template: str=None, system_prompt: str=None, **kwargs):
        """
        Load a Vision/Language Action model.
        
        Args:
          model (str): Either the path to the model, or HuggingFace model repo/name.
          api (str): The model backend to use (MLC - fastest, AWQ - accurate quantization, HF - compatability)
          quantization (str): For MLC: recommend q4f16_ft or 8f16_ft. For AWQ: the path to the quantized weights.
          max_context_len (str): The maximum chat length in tokens (by default, inherited from the model)  
          drop_inputs (bool): If true, only the latest frame will be processed (older frames dropped)
          chat_template (str|dict): The chat template (by default, will attempt to determine from model type)
          system_prompt (str):  Set the system prompt (OpenVLA does not use this)          
        """
        super().__init__(model=model, api=api, quantization=quantization, 
                         max_context_len=max_context_len, drop_inputs=drop_inputs,
                         outputs=['actions'], chat_template=chat_template, 
                         system_prompt=system_prompt, **kwargs)

        if not self.model.vla:
            raise RuntimeError(f"{self.model.config.name} is not a supported VLA model")
 
        self.vla = self.model.vla
        
        self.add_parameter('action_space', type=str, default='normalized', help="Degrees of freedom (xyz, roll/pitch/yaw, gripper) and normalization coefficients", suggestions=list(self.vla.action_spaces.keys()))

    @property
    def action_space(self):
        return self.vla.action_space.name
        
    @action_space.setter
    def action_space(self, key):
        self.vla.action_space = key

    @classmethod
    def type_hints(cls):
        return {
            **NanoLLM.type_hints(), 
            'model': {'suggestions': ["openvla/openvla-7b"]}
        }    
               
    def process(self, input, **kwargs):
        """
        Predict actions from images and instruction prompts.
        If the input is text, it will be saved as the prompt for the robot to follow.
        
        If the input is image, it will be fed through the VLA with the latest prompt,
        and will output the predicted actions (float32 np.ndarray) which can be
        denormalized by setting ``NanoVLA.action_config``
        """ 
        print('input', type(input))
        
        if isinstance(input, list):
            for i in input:
                self.process(i, **kwargs)
            return

        if isinstance(input, str):
            if input.endswith(ImageExtensions):
                input = load_image(input)
            else:
                self.vla.instruction = input
                logging.warning(f"{self.name} using prompt instruction `{input}`")
                return
        
        if not is_image(input):
            raise TypeError(f"{self.name} recieved {type(input)} but expected str or image {ImageTypes}")

        if not self.vla.instruction:
            logging.warning(f"{self.name} recieved image, but waiting for instruction prompt")
            return

        if self.interrupted:
            return

        print('VLA action space', self.vla.action_space.name)
        stream = self.vla.predict_action(input, streaming=True)
        
        for action in stream:
            self.output(stream.actions, partial=True)
        
        self.output(stream.actions)
               
            
