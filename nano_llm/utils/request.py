#!/usr/bin/env python3
import requests
import logging
import traceback
import cachetools.func


class WebRequest:
    """
    Requests wrapper for with rate-limiting, connection retries, and data conversion.
    """
    @staticmethod
    def get(url, retry=None, ttl=None, check_status=True):
        """
        Get the specified URL, and optionally retry if it failed.
        """
        def _get(url, retry):
            if retry is None:
                retry = 0
            elif retry == True:
                retry = 1
                
            for r in range(retry+1):
                try:
                    response = requests.get(url)
                    if not check_status or response.status_code == 200:
                        return WebRequest(url, response)
                    else:
                        logging.error(f"Web request {url} failed  (code {response.status_code})")
                except Exception as error:
                    logging.error(f"Exception occurred during web request {url}\n\n{''.join(traceback.format_exception(error))}")
                    
            return None
     
        if ttl is not None and ttl > 0:
            _get = cachetools.func.ttl_cache(ttl=ttl)(_get)
            
        return _get(url, retry)
        
    def __getitem__(self, key):
        """
        Lookup entries in the JSON response payload (if there was one).
        This returns None if the key is missing instead of throwing an exception.
        """
        obj = self.json if self.json else self.response
        
        if isinstance(obj, dict):
            return obj.get(key, None)
        else:
            return obj[key]

    def to(self, return_type, text=None):
        """
        Convert the response to either a `str`, `dict`, or `json`.
        """
        if return_type == 'str':
            return text if text \
                else json.dumps(self.json if self.json else self.response, indent=2)
        elif return_type == 'dict':
            return self.json if self.json else self.response
        elif return_type == 'json':
            return json.dumps(self.json if self.json else self.response, indent=2)
        else:
            raise ValueError(f"return_type should be 'str', 'dict', or 'json'  (was '{return_type}')")     

    @property
    def status_code(self):
        """
        Returns the response status code (200=ok)
        """
        return self.response.status_code
        
    def __init__(self, url, response):
        """
        Private initializer returned after the request was made.
        """
        self.url = url
        self.response = response
        
        try:
            self.json = self.response.json()
        except Exception:
            self.json = None

