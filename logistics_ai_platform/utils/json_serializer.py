"""
JSON serialization utilities

This module provides optimized JSON serialization using orjson,
which is faster than the standard library json module. Includes
support for custom types like datetime and Decimal.
"""

import logging
from typing import Any, Union
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    import json

logger = logging.getLogger(__name__)


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize object to JSON string
    
    Args:
        obj: Object to serialize
        **kwargs: Additional options for orjson
        
    Returns:
        JSON string representation of the object
        
    Raises:
        TypeError: If object cannot be serialized
    """
    if not ORJSON_AVAILABLE:
        # Fallback to standard library
        return json.dumps(obj, default=_default_serializer, **kwargs)
    
    try:
        options = (
            orjson.OPT_SERIALIZE_NUMPY |
            orjson.OPT_SERIALIZE_DATACLASS |
            orjson.OPT_NON_STR_KEYS
        )
        
        # Merge with any additional options from kwargs
        if 'option' in kwargs:
            options |= kwargs.pop('option')
        
        return orjson.dumps(
            obj,
            option=options,
            default=_default_serializer,
            **kwargs
        ).decode('utf-8')
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error: {e}")
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable") from e


def json_loads(s: Union[str, bytes]) -> Any:
    """
    Deserialize JSON string or bytes
    
    Args:
        s: JSON string or bytes to deserialize
        
    Returns:
        Deserialized Python object
        
    Raises:
        ValueError: If JSON is invalid
    """
    if not ORJSON_AVAILABLE:
        # Fallback to standard library
        return json.loads(s)
    
    try:
        return orjson.loads(s)
    except (orjson.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON deserialization error: {e}")
        raise ValueError(f"Invalid JSON: {e}") from e


def _default_serializer(obj: Any) -> Any:
    """
    Default serializer for custom types
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serializable representation of the object
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    if isinstance(obj, date):
        return obj.isoformat()
    
    if isinstance(obj, Decimal):
        return float(obj)
    
    if isinstance(obj, UUID):
        return str(obj)
    
    # Try to use __dict__ for custom objects
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    
    # Try to use __str__ as last resort
    if hasattr(obj, '__str__'):
        return str(obj)
    
    raise TypeError(
        f"Object of type {type(obj)} is not JSON serializable. "
        f"Consider implementing a custom serializer."
    )

