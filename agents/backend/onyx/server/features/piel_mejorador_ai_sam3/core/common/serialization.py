"""
Serialization Utilities for Piel Mejorador AI SAM3
==================================================

Unified serialization and deserialization utilities.
"""

import json
import logging
from typing import Any, Dict, Type, TypeVar, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class Serializer:
    """Unified serializer for dataclasses and objects."""
    
    @staticmethod
    def to_dict(
        obj: Any,
        include_none: bool = False,
        exclude_fields: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert object to dictionary.
        
        Args:
            obj: Object to serialize
            include_none: Whether to include None values
            exclude_fields: Fields to exclude
            
        Returns:
            Dictionary representation
        """
        exclude_fields = exclude_fields or []
        
        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            data = asdict(obj)
        # Handle objects with __dict__
        elif hasattr(obj, '__dict__'):
            data = obj.__dict__.copy()
        # Handle dict
        elif isinstance(obj, dict):
            data = obj.copy()
        else:
            raise ValueError(f"Cannot serialize object of type {type(obj)}")
        
        # Process fields
        result = {}
        for key, value in data.items():
            if key in exclude_fields:
                continue
            
            if value is None and not include_none:
                continue
            
            # Convert datetime to ISO format
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            # Convert Enum to value
            elif isinstance(value, Enum):
                result[key] = value.value
            # Convert Path to string
            elif isinstance(value, Path):
                result[key] = str(value)
            # Recursively serialize nested objects
            elif hasattr(value, '__dataclass_fields__') or (hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict))):
                result[key] = Serializer.to_dict(value, include_none, exclude_fields)
            # Handle lists
            elif isinstance(value, list):
                result[key] = [
                    Serializer.to_dict(item, include_none, exclude_fields)
                    if hasattr(item, '__dataclass_fields__') or (hasattr(item, '__dict__') and not isinstance(item, (str, int, float, bool, dict)))
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        target_class: Type[T],
        datetime_fields: Optional[list[str]] = None,
        enum_fields: Optional[Dict[str, Type[Enum]]] = None
    ) -> T:
        """
        Create object from dictionary.
        
        Args:
            data: Dictionary data
            target_class: Target class to instantiate
            datetime_fields: List of field names that are datetime
            enum_fields: Dict mapping field names to Enum types
            
        Returns:
            Instantiated object
        """
        datetime_fields = datetime_fields or []
        enum_fields = enum_fields or {}
        
        # Prepare kwargs
        kwargs = {}
        
        # Get field information if dataclass
        if hasattr(target_class, '__dataclass_fields__'):
            field_info = {f.name: f for f in fields(target_class)}
        else:
            field_info = {}
        
        for key, value in data.items():
            # Skip if field doesn't exist
            if field_info and key not in field_info:
                continue
            
            # Convert datetime strings
            if key in datetime_fields and isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except ValueError:
                    logger.warning(f"Invalid datetime format for {key}: {value}")
                    value = None
            
            # Convert enum values
            if key in enum_fields and isinstance(value, str):
                try:
                    enum_class = enum_fields[key]
                    value = enum_class(value)
                except ValueError:
                    logger.warning(f"Invalid enum value for {key}: {value}")
            
            kwargs[key] = value
        
        # Instantiate
        if hasattr(target_class, '__dataclass_fields__'):
            return target_class(**kwargs)
        else:
            instance = target_class.__new__(target_class)
            instance.__dict__.update(kwargs)
            return instance
    
    @staticmethod
    def to_json(
        obj: Any,
        indent: int = 2,
        include_none: bool = False
    ) -> str:
        """
        Convert object to JSON string.
        
        Args:
            obj: Object to serialize
            indent: JSON indentation
            include_none: Whether to include None values
            
        Returns:
            JSON string
        """
        data = Serializer.to_dict(obj, include_none=include_none)
        return json.dumps(data, indent=indent, ensure_ascii=False, cls=DateTimeEncoder)
    
    @staticmethod
    def from_json(
        json_str: str,
        target_class: Type[T],
        datetime_fields: Optional[list[str]] = None,
        enum_fields: Optional[Dict[str, Type[Enum]]] = None
    ) -> T:
        """
        Create object from JSON string.
        
        Args:
            json_str: JSON string
            target_class: Target class
            datetime_fields: List of datetime field names
            enum_fields: Dict mapping field names to Enum types
            
        Returns:
            Instantiated object
        """
        data = json.loads(json_str)
        return Serializer.from_dict(data, target_class, datetime_fields, enum_fields)


def serialize(obj: Any, **kwargs) -> Dict[str, Any]:
    """Convenience function for serialization."""
    return Serializer.to_dict(obj, **kwargs)


def deserialize(data: Dict[str, Any], target_class: Type[T], **kwargs) -> T:
    """Convenience function for deserialization."""
    return Serializer.from_dict(data, target_class, **kwargs)




