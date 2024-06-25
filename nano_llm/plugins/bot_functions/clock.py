#!/usr/bin/env python3
from datetime import datetime
from nano_llm import bot_function


@bot_function
def time():
    """
    Returns the current time.
    """
    return datetime.now().strftime("%-I:%M %p")


@bot_function
def date():
    """
    Returns the current date.
    """
    return datetime.now().strftime("%A, %B %-m %Y")
