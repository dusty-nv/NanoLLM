#!/usr/bin/env python3
from nano_llm import Plugin
from nano_llm.utils import WebRequest


class Location(Plugin):
    """
    Geolocation functions for the bot.
    """
    def __init__(self, **kwargs):
        super().__init__(outputs=None, threaded=False, **kwargs)
        self.add_tool(self.geolocation)
        
    def geolocation() -> str:
        """
        Returns the current location, like the name of the city.
        This function takes no arguments.
        """
        response = WebRequest.get("http://ip-api.com/json", ttl=600)  # zip, lat/lon, timezone, query (IP)
        return f"{response['city']}, {response['regionName']}"

