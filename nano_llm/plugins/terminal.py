#!/usr/bin/env python3
import logging

from nano_llm import Plugin


'''
    DefaultColors = {
        logging.DEBUG: ('light_grey', 'dark'),
        logging.INFO: None,
        logging.WARNING: 'yellow',
        logging.SUCCESS: 'green',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red'
    }
'''
    
class TerminalPlugin(Plugin, logging.Handler):
    """
    View the console log remotely over websocket.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.send_client_output(0)
        self.handler = LoggingHandler(self)
        logging.getLogger().addHandler(self.handler)
        
         
class LoggingHandler(logging.Handler):
    def __init__(self, plugin):
        super().__init__()
        
        self.plugin = plugin
        self.level = logging.DEBUG
        
        self.level_to_str = {
            logging.DEBUG: 'debug',
            logging.INFO: 'info',
            logging.WARNING: 'warning',
            logging.SUCCESS: 'success',
            logging.ERROR: 'error',
            logging.CRITICAL: 'critical'
        }
        
    def emit(self, record):
        self.plugin.output({
            'level': self.level_to_str[record.levelno],
            'message': record.getMessage()
        })   
        
