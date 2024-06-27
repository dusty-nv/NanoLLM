#!/usr/bin/env python3
import logging

from nano_llm import bot_function
from nano_llm.web import WebServer


@bot_function
def send_alert(message: str, level: str="info"):
    """
    Sends an alert notification to the user.  Only call this function if absolutely necessary, because it will interrupt the user in whatever they are doing.
    
    Args:
      message (str): The text to send in the alert.
      level (str): The importance of the alert - "critical" for emergencies, "warning" for important notifications, and "info" for general information.
    """
    logging.warning(f"alert:  {message}")
    if WebServer.Instance:
        WebServer.Instance.send_alert(message, level=level, category='bot')

