"""
Serialization Utilities
=======================

Utilities for serialization and deserialization.
"""

import json
import pickle
from typing import Any, Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types"""
    
    def default(self, obj: Any) -> Any:
        """Handle special types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """
    Convert object to JSON string.
    
    Args:
        obj: Object to serialize
        indent: Optional indentation
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=JSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Parse JSON string to object.
    
    Args:
        json_str: JSON string
        
    Returns:
        Parsed object
    """
    return json.loads(json_str)


def to_pickle(obj: Any) -> bytes:
    """
    Serialize object to pickle bytes.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Pickle bytes
    """
    return pickle.dumps(obj)


def from_pickle(pickle_bytes: bytes) -> Any:
    """
    Deserialize object from pickle bytes.
    
    Args:
        pickle_bytes: Pickle bytes
        
    Returns:
        Deserialized object
    """
    return pickle.loads(pickle_bytes)


def serialize_dict(data: dict, format: str = "json") -> str:
    """
    Serialize dictionary to string.
    
    Args:
        data: Dictionary to serialize
        format: Format ("json" or "pickle")
        
    Returns:
        Serialized string
    """
    if format == "json":
        return to_json(data)
    elif format == "pickle":
        return to_pickle(data).hex()
    else:
        raise ValueError(f"Unsupported format: {format}")


def deserialize_dict(data: str, format: str = "json") -> dict:
    """
    Deserialize string to dictionary.
    
    Args:
        data: Serialized string
        format: Format ("json" or "pickle")
        
    Returns:
        Deserialized dictionary
    """
    if format == "json":
        return from_json(data)
    elif format == "pickle":
        return from_pickle(bytes.fromhex(data))
    else:
        raise ValueError(f"Unsupported format: {format}")

