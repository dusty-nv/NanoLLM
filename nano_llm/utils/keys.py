#!/usr/bin/env python3

def KeyMap(keys, to='dict'):
    """
    Parse mappings of the form "x:y a:b", returning a dictionary from LHS->RHS.
    """
    if keys is None:
        return {}
    elif isinstance(keys, dict):
        return keys
    elif isinstance(keys, str):
        keys = [keys.split(' ')]
    elif not isinstance(keys, (list, tuple)):
        raise TypeError("KeyMap expected dict, list[str], or str (was {type(keys)})")
        
    key_map = {}
    
    for key_list in keys:
        if isinstance(key_list, str):
            key_list = [key_list]
            
        for key in key_list:
            key = key.split(':')
            
            if key[-1].lower() == 'none':
                key[-1] = None
                
            key_map[key[0]] = key[-1]
     
    if to == 'str':
        return ' '.join([f'{k}:{v}' for k,v in key_map.items()])
    else:   
        return key_map
    
      
def filter_keys(dictionary, keep=None, remove=None):
    """
    Remove keys from a dict by either a list of keys to keep or remove.
    """
    if isinstance(dictionary, list):
        for x in dictionary:
            filter_keys(x, keep=keep, remove=remove)
        return dictionary
        
    for key in list(dictionary.keys()):
        if (keep and key not in keep) or (remove and key in remove):
            del dictionary[key]
 
    return dictionary
    
    
def validate(value, default=None, cast=None):
    """
    If the value is None, return the default instead.
    """
    if isinstance(value, str):
        value = value if value.strip() else default
    else:
        value = default if value is None else value
        
    if cast is not None:
        value = cast(value)
        
    return value
        
        
def validate_key(dict, key, default=None, cast=None):
    """
    If the value of the given key in the dictionary is None, return the default instead.
    """
    return validate(dict.get(key), default=default, cast=cast)


def validate_attr(object, attr, default=None, cast=None):
    """
    If the value of the given attribute name in the object is None, return the default instead.
    """
    return validate(getattr(object, attr, None), default=default, cast=cast)
    

