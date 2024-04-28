#!/usr/bin/env python3
import os

from nano_llm import bot_function
from nano_llm.utils import WebRequest

from .location import LOCATION


ACCUWEATHER_KEY=os.environ.get('ACCUWEATHER_KEY', None)
OPENWEATHER_KEY=os.environ.get('OPENWEATHER_KEY', None)

WEATHER_API='accuweather' if ACCUWEATHER_KEY else 'openweather' if OPENWEATHER_KEY else None
WEATHER_KEY=ACCUWEATHER_KEY or OPENWEATHER_KEY
WEATHER_TTL=1800 # number of seconds that cached requests expire after


@bot_function(enabled=True if WEATHER_API else False)
def WEATHER(location=None, api=WEATHER_API, api_key=WEATHER_KEY, return_type='str'):
    """
    Returns the current weather and temperature.
    """
    if not api_key:
        raise ValueError(f"$ACCUWEATHER_KEY or $OPENWEATHER_KEY should be set to your respective API key to use weather data")
        
    if not location:
        location = LOCATION(return_type='dict')

    if api == 'openweather':
        response = WebRequest.get(f"https://api.openweathermap.org/data/2.5/weather?lat={location['lat']}&lon={location['lon']}&units=imperial&appid={api_key}", ttl=WEATHER_TTL)
        
        if response['cod'] != 200:
            raise RuntimeError(f"failed to get weather data from OpenWeather ({response['message']})")

        return request.to(return_type, f"{int(response['main']['temp'])} degrees and {response['weather']['description']}")
    elif api == 'accuweather':
        geoposition = _accuweather_geoposition(location, api_key=WEATHER_KEY)
        response = WebRequest.get(f"http://dataservice.accuweather.com/currentconditions/v1/{geoposition['Key']}?apikey={api_key}", ttl=WEATHER_TTL)
        return response.to(return_type, f"{int(response[0]['Temperature']['Imperial']['Value'])} degrees and {response[0]['WeatherText']}")
    else:
        raise ValueError(f"api should be 'openweather' or 'accuweather'  (was '{api}')")
        
    return response
 
#@bot_function(enabled=True if WEATHER_API else False, docs='pydoc_nosig')
@bot_function(enabled=False, docs='pydoc_nosig')
def WEATHER_FORECAST(location=None, day=1, api=WEATHER_API, api_key=WEATHER_KEY, return_type='str'):
    """
    `WEATHER_FORECAST(day=1)` - Returns the weather forecast.  Has an optional keyword argument that specifies the number of days ahead for the forecast (by default, tomorrow).
    """
    if not api_key:
        raise ValueError(f"$ACCUWEATHER_KEY or $OPENWEATHER_KEY should be set to your respective API key to use weather data")

    if api == 'accuweather':
        geoposition = _accuweather_geoposition(location, api_key=WEATHER_KEY)
        response = WebRequest.get(f"http://dataservice.accuweather.com/forecasts/v1/daily/1day/{geoposition['Key']}?apikey={api_key}", ttl=WEATHER_TTL)
        return response.to(return_type, f"{response['Headline']['Text']}, with a high of {int(response['DailyForecasts'][0]['Temperature']['Maximum']['Value'])} degrees.")
    else:
        raise ValueError(f"api should be 'accuweather'  (was '{api}')")
        
    return response
    
    
def _accuweather_geoposition(location=None, api_key=WEATHER_KEY):
    if not location:
        location = LOCATION(return_type='dict')
        
    response = WebRequest.get(f"http://dataservice.accuweather.com/locations/v1/cities/geoposition/search?q={location['lat']},{location['lon']}&apikey={api_key}", ttl=WEATHER_TTL)
        
    if response is None:
        raise RuntimeError(f"failed to get location data from Accuweather")
            
    return response
    
