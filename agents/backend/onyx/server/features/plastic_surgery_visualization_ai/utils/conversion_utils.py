"""Conversion utilities."""

from typing import Any, Optional, Union
import json


def to_dict(obj: Any) -> dict:
    """
    Convert object to dictionary.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '_asdict'):  # namedtuple
        return obj._asdict()
    else:
        return {"value": obj}


def to_list(value: Any) -> list:
    """
    Convert value to list.
    
    Args:
        value: Value to convert
        
    Returns:
        List
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set)):
        return list(value)
    elif value is None:
        return []
    else:
        return [value]


def to_set(value: Any) -> set:
    """
    Convert value to set.
    
    Args:
        value: Value to convert
        
    Returns:
        Set
    """
    if isinstance(value, set):
        return value
    elif isinstance(value, (list, tuple)):
        return set(value)
    elif value is None:
        return set()
    else:
        return {value}


def to_tuple(value: Any) -> tuple:
    """
    Convert value to tuple.
    
    Args:
        value: Value to convert
        
    Returns:
        Tuple
    """
    if isinstance(value, tuple):
        return value
    elif isinstance(value, (list, set)):
        return tuple(value)
    elif value is None:
        return tuple()
    else:
        return (value,)


def json_to_dict(json_str: str) -> dict:
    """
    Convert JSON string to dictionary.
    
    Args:
        json_str: JSON string
        
    Returns:
        Dictionary
    """
    return json.loads(json_str)


def dict_to_json(d: dict, indent: Optional[int] = None) -> str:
    """
    Convert dictionary to JSON string.
    
    Args:
        d: Dictionary
        indent: Optional indentation
        
    Returns:
        JSON string
    """
    return json.dumps(d, indent=indent)


def bytes_to_string(data: bytes, encoding: str = 'utf-8') -> str:
    """
    Convert bytes to string.
    
    Args:
        data: Bytes to convert
        encoding: Text encoding
        
    Returns:
        String
    """
    return data.decode(encoding)


def string_to_bytes(text: str, encoding: str = 'utf-8') -> bytes:
    """
    Convert string to bytes.
    
    Args:
        text: String to convert
        encoding: Text encoding
        
    Returns:
        Bytes
    """
    return text.encode(encoding)

