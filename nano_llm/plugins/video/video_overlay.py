#!/usr/bin/env python3
import PIL
import time
import logging
import traceback

from nano_llm import Plugin, StopTokens
from nano_llm.utils import is_image, cuda_image, wrap_text

from jetson_utils import cudaFont, cudaMemcpy, cudaEventRecord


class VideoOverlay(Plugin):
    """
    Overlay static or dynamic text on top of a video stream.
    """
    def __init__(self, font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
                 font_size: float = 32.0, prefix_text: str = None,    
                 color: str = "#78d715", background: str = "#000000", opacity: float = 0.4, 
                 x: int = 5, y: int = 5, line_spacing: int = 38, line_length: int = None, 
                 return_copy: bool = True, **kwargs):
        """
        Overlay text on top of a video stream.
        
        Args:
          font (str):  Name of the TTF font to use.
          font_size (float):  Height of the font (in pixels)
          prefix_text (str):  Static text that is drawn before any text the plugin recieves at runtime.
          color (str):  Foreground color of the text (in hex)
          background (str):  Background color behind the text (for readability)
          opacity (float):  Alpha value of the background between 0 and 1 (0=disabled)
          x (int):  x-coordinate offset in the image to render the text.
          y (int):  y-coordinate offset in the image to render the text.
          line_spacing (int):  The spacing between lines (in pixels) for when the text is wrapped onto multiple lines.
          line_length (int):  The maximum number of characters per line before wrapping (by default, will be calculated from the image dimensions)
          return_copy (bool): Copy incoming frames to prevent other possible consumers from losing the original.
        """
        super().__init__(outputs='image', **kwargs)
        
        self.text = None
        
        self._font_name = None
        self._font_size = font_size

        self.add_parameters(font=font, font_size=font_size, prefix_text=prefix_text,
                            color=color, background=background, opacity=opacity, x=x, y=y, 
                            line_spacing=line_spacing, line_length=line_length,
                            return_copy=return_copy)

    @property
    def font(self):
        return self._font_name
        
    @font.setter
    def font(self, name):
        if self._font_name != name:
            self._font = cudaFont(name, self._font_size)
            self._font_name = name
            
    @property
    def font_size(self):
        return self._font_size
        
    @font_size.setter
    def font_size(self, size):
        if self._font_size != size:
            self._font = cudaFont(self._font_name, self._font_size)
            self._font_size = size

    @classmethod
    def type_hints(cls):
        return {
            'color': {'color': True},
            'background': {'color': True},
        }
        
    def process(self, input, return_copy=None, **kwargs):
        """
        Input should be a jetson_utils.cudaImage, np.ndarray, torch.Tensor, or have __cuda_array_interface__
        """
        if isinstance(input, str):
            self.text = input
            return
            
        if not is_image(input):
            raise TypeError(f"{self.name} expected to recieve str or image (PIL.Image, np.ndarray, torch.Tensor, cudaImage)  (was {type(input)})")
            
        text = ''
        
        if self.prefix_text:
            text = self.prefix_text
            
        if self.text:
            text = text + (' ' if text else '') + self.text
            
        if not text:
            self.output(input)
        
        for stop_token in StopTokens + ['###']:
            text = text.replace(stop_token, '')
        
        input = cuda_image(input)
        stream = input.stream
        
        if return_copy is None:
            return_copy = self.return_copy
            
        if return_copy:
            input = cudaMemcpy(input, stream=stream)
            
        alpha = int(self.opacity * 255)
        color = PIL.ImageColor.getcolor(self.color, 'RGB')
        bg = PIL.ImageColor.getcolor(self.background, 'RGB') 
        
        if len(bg) == 3:
            bg = (bg[0], bg[1], bg[2], alpha)
           
        font_kwargs = {
            'text': text,
            'x': self.x,
            'y': self.y,
            'color': color,
            'background': bg,
            'line_spacing': self.line_spacing
        }
        
        if self.line_length:
            font_kwargs['line_length'] = self.line_length
            
        font_kwargs.update(kwargs)
        wrap_text(self._font, input, stream=stream, **font_kwargs)
        
        if stream:
            input.event = cudaEventRecord(stream=stream)
            input.stream = stream
            
        self.output(input)

            
