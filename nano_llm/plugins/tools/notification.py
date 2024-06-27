#!/usr/bin/env python3
import logging

from nano_llm import Plugin
from nano_llm.plugins import Deduplicate


class Notification(Deduplicate):
    """
    Enable the bot to send alerts & notifications.
    """
    Levels = ['critical', 'warning', 'info']
    
    def __init__(self, filter_level: str='warning', similarity_threshold: float=0.25, timeout: float=10.0, **kwargs):
        """
        Enable the bot to send alerts & notifications.
        
        Args:
          filter_level (str): Suppress any notifications below this alert level.
          similarity_threshold (float):  For deduplication, how similar the text should be (between 0 and 1)
          timeout (float):  For deduplication, the time in seconds after which the previous text is forgotten.
        """
        super().__init__(outputs=Notification.Levels, similarity_threshold=similarity_threshold, timeout=timeout, threaded=False, **kwargs)

        self.add_parameter('filter_level', default=filter_level)
        
        self.levels = dict(
            critical = dict(output=0, log=logging.ERROR),
            warning = dict(output=1, log=logging.WARNING),
            info = dict(output=2, log=logging.INFO),
        )
        
        self.add_tool(self.send_alert)
        
    def send_alert(self, message: str, level: str="info"):
        """
        Sends an alert notification to the user.  Only call this function if absolutely necessary, because it will interrupt the user in whatever they are doing.
        
        Args:
          message (str): The text to send in the alert.
          level (str): The importance of the alert - "critical" for emergencies, "warning" for important notifications, and "info" for general information.
        """
        if level not in self.levels:
            logging.warning(f"Notification.send_alert() - unknown logging level:  '{level}'")
            level = "info"
            
        logging.log(self.levels[level]['log'], f"Notification.send_alert() - {message}")
        
        filter_level = Notification.Levels.index(self.filter_level)
        alert_level = Notification.Levels.index(level)
        
        if alert_level > filter_level:
            return
            
        message = super().process(message)
        
        if not message:
            return
            
        self.output(message, channel=self.levels[level]['output'])
        super().send_alert(message, level=level, category='bot')
        
    @classmethod
    def type_hints(cls):
        return dict(filter_level = dict(options = Notification.Levels))

