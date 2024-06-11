#!/usr/bin/env python3
from nano_llm import Plugin
from nano_llm.web import WebServer


class WebClient(Plugin):
    """
    Forwards output from a plugin to clients over websocket.
    """
    def __init__(self, **kwargs):
        super().__init__(outputs=0, **kwargs)

    def process(self, data, sender=None, channel=0, **kwargs): 
        if not WebServer.Instance or not WebServer.Instance.connected:
            return

        WebServer.Instance.send_message({
            'send_output': {
                'name': sender.name,
                'channel': channel,
                'data': data,
            }
        })

