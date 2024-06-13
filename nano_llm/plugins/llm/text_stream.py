#!/usr/bin/env python3
from nano_llm import Plugin


class TextStream(Plugin):
    """
    Sends text over websocket to clients.
    """
    def __init__(self, **kwargs):
        super().__init__(inputs='text', outputs=0, **kwargs)
        
        self.add_parameter('text_color', type=str, default='white', help="Color to show normal text in.")
        self.add_parameter('partial_color', type=str, default='dodgerblue', help="Color for partial text that is not yet finalized.")
        self.add_parameter('delta_color', type=str, default='limegreen', help="Color to show the latest chunks of text from the stream.")
        
    def process(self, text, partial=False, delta=False, **kwargs): 
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('\n', '<br/>')
        
        msg = {'text': text}
        
        if delta:
            msg['color'] = self.delta_color
            msg['delta'] = True
        elif partial:
            msg['color'] = self.partial_color
            msg['partial'] = True
        else:
            msg['color'] = self.text_color

        self.send_state(msg)
