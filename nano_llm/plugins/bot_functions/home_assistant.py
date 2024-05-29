#!/usr/bin/env python3
import os
import requests

from nano_llm import bot_function
from nano_llm.utils import filter_keys

HOMEASSISTANT_URL=os.environ.get('HOMEASSISTANT_URL', 'http://localhost:8123')
HOMEASSISTANT_KEY=os.environ.get('HOMEASSISTANT_KEY', None)
USE_HOMEASSISTANT=True if HOMEASSISTANT_KEY else False

HOMEASSISTANT_DEVICES=['switch.zooz_plug']
HOMEASSISTANT_STATES={
    'switch': ['turn_on', 'turn_off', 'toggle']
}

@bot_function(enabled=USE_HOMEASSISTANT, docs='openai')
def control_home_device(name : str, state : str) -> str:
    """
    Controls a smart home device, for example turning a light switch on or off.

    Args:
        name (str): The name of the device. These devices are available: switch.zooz_plug
        state (str): The operation to perform. For switches, this can be 'turn_on', 'turn_off', or 'toggle'.
        
    Returns:
        str: A string describing the success or failure of the operation.
    """
    if not HOMEASSISTANT_KEY:
        raise ValueError(f"$HOMEASSISTANT_KEY should be set to your respective API key")

    name = name.lower()
    
    if name not in HOMEASSISTANT_DEVICES:
        return f"A device by that name could not be found. These devices are available: {','.join(HOMEASSISTANT_DEVICES)}"
        
    state = state.lower()
    domain = name.split('.')[0]
    
    if state not in HOMEASSISTANT_STATES[domain]:
        return "Invalid state requested - valid states for {domain} are: {','.join(HOMEASSISTANT_STATES[domain])}"

    url = f"{HOMEASSISTANT_URL}/api/services/{domain}/{state}"
    headers = {"Authorization": f"Bearer {HOMEASSISTANT_KEY}", "content-type": "application/json"}
    data = {"entity_id": name}

    response = requests.post(url, headers=headers, json=data)
    
    if response.text == "[]":
        return f"The device {name} was already in {state}"
        
    return response.text #.replace('\\"', '\"').replace("\\'", "\'")

    
if __name__ == "__main__":
    import argparse
    import pprint
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=HOMEASSISTANT_DEVICES[0])
    parser.add_argument("--state", type=str, default="toggle")
    args = parser.parse_args()
    
    print(args)
    result = control_home_device(args.name, args.state)
    print(type(result))
    print(result)

    
