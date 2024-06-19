#!/usr/bin/env python3
import requests
import logging

from nano_llm import Plugin


class HomeAssistant(Plugin):
    """
    HomeAssistant tools for the bot.
    """
    def __init__(self, url: str = "http://localhost:8123", api_key: str = None, **kwargs):
        """
        Args:
          url (str):  The URL and port of the HomeAssistant server.
          api_key (str):  Your HomeAssistant API key.
        """
        super().__init__(outputs=None, threaded=False, **kwargs)

        if not api_key:
            raise ValueError(f"The HomeAssistant API key needs set")
            
        self.add_parameters(url=url, api_key=api_key)

        self.states = {
            'switch': ['turn_on', 'turn_off', 'toggle']
        }
        
        self.devices = ['switch.zooz_plug']
        
        self.add_tool('control_device')
       
    def control_device(name : str, state : str) -> str:
        """
        Controls a smart home device, for example turning a light switch on or off.

        Args:
            name (str): The name of the device. These devices are available: switch.zooz_plug
            state (str): The operation to perform. For switches, this can be 'turn_on', 'turn_off', or 'toggle'.
            
        Returns:
            str: A string describing the success or failure of the operation.
        """
        name = name.lower()
        
        if name not in self.devices:
            return f"A device by that name could not be found. These devices are available: {','.join(self.devices)}"
            
        state = state.lower()
        domain = name.split('.')[0]
        
        if state not in self.states[domain]:
            return "Invalid state requested - valid states for {domain} are: {','.join(self.states[domain])}"

        url = f"{self.url}/api/services/{domain}/{state}"
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        data = {"entity_id": name}

        response = requests.post(url, headers=headers, json=data)
        
        if response.text == "[]":
            return f"The device {name} was already in {state}"
            
        return response.text #.replace('\\"', '\"').replace("\\'", "\'")

