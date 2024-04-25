#!/usr/bin/env python3
from nano_llm import bot_function
from nano_llm.utils import WebRequest


@bot_function
def LOCATION(return_type='str'):
    """ 
    Returns the current location, like the name of the city.
    """
    response = WebRequest.get("http://ip-api.com/json", ttl=600)  # zip, lat/lon, timezone, query (IP)
    text = f"{response['city']}, {response['regionName']}"
    return response.to(return_type, text)

