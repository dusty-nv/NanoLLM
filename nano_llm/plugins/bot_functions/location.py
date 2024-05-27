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


@bot_function(docs='openai')
def current_location():
    """
    Returns the current location, like the name of the city.
    This function takes no arguments.
    """
    r = WebRequest.get("http://ip-api.com/json", ttl=600).to_dict()
    
    return {
        'city': r['city'],
        'state': r['regionName'],
        'stateCode': r['region'],
        'country': r['country'],
        'countryCode': r['countryCode'],
        'zip': r['zip'],
        'lat': r['lat'],
        'lon': r['lon'],
     }
    
    
"""
Returns the current location, like the name of the city.

Returns:
    dict: A dictionary with geographic info about the current location.
        Keys:
            - 'city': The name of the closest city or town.
            - 'state': The name of the state.
            - 'stateCode': The two-letter abbreviation of the state.
            - 'country': The name of the country.
            - 'countryCode': The two-letter abbreviation of the country.
            - 'zip': The mailing ZIP code of the location.
            - 'lat': The latitude coordinates of the current location.
            - 'lon': The longitude coordinates of the current location.
"""
if __name__ == "__main__":
    import pprint

    print(f"Getting the current location")
    pprint.pprint(get_location(), indent=2)
