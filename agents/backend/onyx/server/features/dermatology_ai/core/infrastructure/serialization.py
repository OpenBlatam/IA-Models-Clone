"""
Serialization Utilities
Optimized serialization for requests and responses
"""

from typing import Any, Dict, Optional
from datetime import datetime
from decimal import Decimal
import json
import logging

logger = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types"""
    
    def default(self, obj: Any) -> Any:
        """Handle custom types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


class Serializer:
    """Optimized serializer for API responses"""
    
    @staticmethod
    def serialize(obj: Any, exclude_none: bool = False, exclude_empty: bool = False) -> Any:
        """
        Serialize object to JSON-serializable format
        
        Args:
            obj: Object to serialize
            exclude_none: Exclude None values
            exclude_empty: Exclude empty values (empty strings, lists, dicts)
            
        Returns:
            Serialized object
        """
        if obj is None:
            return None
        
        # Handle dict
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                serialized_value = Serializer.serialize(value, exclude_none, exclude_empty)
                
                if exclude_none and serialized_value is None:
                    continue
                
                if exclude_empty:
                    if serialized_value == "" or serialized_value == [] or serialized_value == {}:
                        continue
                
                result[key] = serialized_value
            return result
        
        # Handle list
        if isinstance(obj, (list, tuple)):
            return [Serializer.serialize(item, exclude_none, exclude_empty) for item in obj]
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle objects with to_dict method
        if hasattr(obj, 'to_dict'):
            return Serializer.serialize(obj.to_dict(), exclude_none, exclude_empty)
        
        # Handle dataclasses
        if hasattr(obj, '__dict__'):
            return Serializer.serialize(obj.__dict__, exclude_none, exclude_empty)
        
        # Handle enums
        if hasattr(obj, 'value'):
            return obj.value
        
        return obj
    
    @staticmethod
    def to_json(obj: Any, indent: Optional[int] = None, exclude_none: bool = False) -> str:
        """
        Convert object to JSON string
        
        Args:
            obj: Object to convert
            indent: JSON indentation
            exclude_none: Exclude None values
            
        Returns:
            JSON string
        """
        serialized = Serializer.serialize(obj, exclude_none=exclude_none)
        return json.dumps(serialized, indent=indent, cls=JSONEncoder)
    
    @staticmethod
    def optimize_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize response data by removing unnecessary fields
        
        Args:
            data: Response data
            
        Returns:
            Optimized response
        """
        # Remove None values
        optimized = {k: v for k, v in data.items() if v is not None}
        
        # Remove empty collections
        optimized = {
            k: v for k, v in optimized.items()
            if not (isinstance(v, (list, dict, str)) and len(v) == 0)
        }
        
        return optimized


def serialize_response(func):
    """Decorator to automatically serialize response"""
    import functools
    import asyncio
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        return Serializer.serialize(result, exclude_none=True)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return Serializer.serialize(result, exclude_none=True)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper















