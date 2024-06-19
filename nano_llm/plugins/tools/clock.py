#!/usr/bin/env python3
from datetime import datetime
from nano_llm import Plugin


class Clock(Plugin):
    """
    Time and date functions for the bot.
    """
    def __init__(self, **kwargs):
        super().__init__(outputs=None, threaded=False, **kwargs)
        
        self.add_tool(self.time)
        self.add_tool(self.date)
    
        #self.add_parameter('get_time', name='TIME', type=str, read_only=True)
        #self.add_parameter('get_date', name='DATE', type=str, read_only=True)

    def time(self) -> str:
        """
        Returns the current time.
        """
        return datetime.now().strftime("%-I:%M %p")

    def date(self) -> str:
        """
        Returns the current date.
        """
        return datetime.now().strftime("%A, %B %-m %Y")
