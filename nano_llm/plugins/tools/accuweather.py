#!/usr/bin/env python3
from nano_llm import Plugin
from nano_llm.utils import WebRequest, filter_keys
from nano_llm.plugins.bot_functions.location import get_current_location


class AccuWeather(Plugin):
    """
    Weather plugin for retrieving current conditions and forecasts using AccuWeather.
    """
    def __init__(self, api_key: str=None, cache_ttl: int=1800, **kwargs):
        """
        Args:
          api_key (str): Your AccuWeather account's access token.
          cache_ttl (str): The number of seconds after cached requests expire.
        """
        super().__init__(outputs=None, threaded=False, **kwargs)
        
        if not api_key:
            raise ValueError(f"api_key should be set to your access token for the weather API")

        self.add_parameter('api_key', default=api_key)
        self.add_parameter('cache_ttl', default=cache_ttl)
        
        self.add_tool(self.get_weather_conditions)
        self.add_tool(self.get_weather_forecast)

    def get_weather_conditions(self, location : str) -> str:
        """
        Returns the current weather conditions and temperature.

        Args:
            location (str): The name of the city or place to get the current weather for.
            
        Returns:
            dict: A dictionary containing current weather information for the given location.
                Keys:
                    - 'temperature': The current temperature, in degrees Fahrenheit.
                    - 'description': A string describing current weather conditions.
                    - 'precipitation': If there is active precipitation, a string such as 'Rain', 'Snow', or 'Ice' (otherwise None).
        """
        geoposition = self.geoposition(location)
        weather = WebRequest.get(f"http://dataservice.accuweather.com/currentconditions/v1/{geoposition['Key']}?apikey={self.api_key}", ttl=self.cache_ttl)[0]
        
        weather = {
            'temperature' : weather['Temperature']['Imperial']['Value'], # { 'Imperial': {'Unit': 'F', 'UnitType': 18, 'Value': 75.0}, 'Metric': {'Unit': 'C', 'UnitType': 17, 'Value': 24.1}},
            'description' : weather['WeatherText'],
            'precipitation': weather['PrecipitationType'],
        }

        return weather
     
    def get_weather_forecast(self, location : str):
        """
        Returns the 5-day weather forecast.
        
        Args:
            location (str): The name of the city or place to get the weather forecast for.
            
        Returns:
            list[dict]: A list of dictionaries containing the weather forecast for each day.
        """
        geoposition = self.geoposition(location)

        forecasts = WebRequest.get(f"http://dataservice.accuweather.com/forecasts/v1/daily/5day/{geoposition['Key']}?apikey={self.api_key}", ttl=self.cache_ttl)["DailyForecasts"]

        for day in forecasts:
            filter_keys(day, keep=['Date', 'Temperature', 'Day', 'Night'])
            filter_keys(day['Day'], remove=['Icon', 'IconPhrase'])
            filter_keys(day['Night'], remove=['Icon', 'IconPhrase'])
            day['Date'] = day['Date'][:10]

        return forecasts
     
    def geoposition(self, location=None):
        """
        Looks up the AccuWeather location ID from a place or name.
        """
        if location is None:
            location = get_current_location()
        elif isinstance(location, str):
            location = WebRequest.get(f"http://dataservice.accuweather.com/locations/v1/cities/search?q={location}&apikey={self.api_key}", ttl=self.cache_ttl)
            location = {'lat': location[0]['GeoPosition']['Latitude'], 'lon': location[0]['GeoPosition']['Longitude']}
        elif not (isinstance(location, dict) and 'lat' in location and 'lon' in location):
            raise ValueError(f"location should either be a string or dict with keys for lat/long (was type '{type(location)}'")
            
        response = WebRequest.get(f"http://dataservice.accuweather.com/locations/v1/cities/geoposition/search?q={location['lat']},{location['lon']}&apikey={self.api_key}", ttl=self.cache_ttl)
            
        if response is None:
            raise RuntimeError(f"failed to get location data from Accuweather")
                
        return response
          
    @classmethod
    def type_hints(cls):
        return {
            'api_key': {'display_name': 'API Key', 'password': True},
            'cache_ttl': {'display_name': 'Cache TTL'},
        }
            
   
