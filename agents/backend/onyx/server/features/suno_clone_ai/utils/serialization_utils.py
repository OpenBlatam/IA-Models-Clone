"""
Unified serialization utilities for converting objects to dictionaries and JSON.

Consolidates common serialization patterns across the codebase.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SerializationUtils:
    """Unified serialization utilities."""
    
    @staticmethod
    def to_dict(
        obj: Any,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert object to dictionary.
        
        Supports:
        - Pydantic models
        - SQLAlchemy models
        - Plain dictionaries
        - Objects with to_dict() method
        - Objects with __dict__
        - Basic types
        
        Args:
            obj: Object to serialize
            exclude_none: Exclude None values
            exclude_unset: Exclude unset fields (Pydantic)
            exclude_fields: List of field names to exclude
        
        Returns:
            Dictionary representation
        """
        if obj is None:
            return {}
        
        exclude_fields = exclude_fields or []
        
        # Pydantic models
        if isinstance(obj, BaseModel):
            return SerializationUtils._pydantic_to_dict(
                obj, exclude_none, exclude_unset, exclude_fields
            )
        
        # SQLAlchemy models
        if hasattr(obj, '__table__'):
            return SerializationUtils._sqlalchemy_to_dict(obj, exclude_fields)
        
        # Objects with to_dict method
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            result = obj.to_dict()
            return SerializationUtils._filter_dict(result, exclude_fields, exclude_none)
        
        # Plain dictionaries
        if isinstance(obj, dict):
            return SerializationUtils._filter_dict(obj, exclude_fields, exclude_none)
        
        # Objects with __dict__
        if hasattr(obj, '__dict__'):
            return SerializationUtils._object_to_dict(obj, exclude_fields, exclude_none)
        
        # Basic types
        return SerializationUtils._basic_to_dict(obj)
    
    @staticmethod
    def _pydantic_to_dict(
        obj: BaseModel,
        exclude_none: bool,
        exclude_unset: bool,
        exclude_fields: List[str]
    ) -> Dict[str, Any]:
        """Convert Pydantic model to dict."""
        result = obj.dict(
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude=set(exclude_fields) if exclude_fields else None
        )
        return SerializationUtils._convert_values(result)
    
    @staticmethod
    def _sqlalchemy_to_dict(
        obj: Any,
        exclude_fields: List[str]
    ) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dict."""
        result = {}
        for column in obj.__table__.columns:
            if column.name in exclude_fields:
                continue
            value = getattr(obj, column.name)
            result[column.name] = SerializationUtils._convert_value(value)
        return result
    
    @staticmethod
    def _object_to_dict(
        obj: Any,
        exclude_fields: List[str],
        exclude_none: bool
    ) -> Dict[str, Any]:
        """Convert object with __dict__ to dict."""
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_') or key in exclude_fields:
                continue
            if exclude_none and value is None:
                continue
            result[key] = SerializationUtils._convert_value(value)
        return result
    
    @staticmethod
    def _filter_dict(
        data: Dict[str, Any],
        exclude_fields: List[str],
        exclude_none: bool
    ) -> Dict[str, Any]:
        """Filter dictionary by exclude fields and None values."""
        result = {}
        for key, value in data.items():
            if key in exclude_fields:
                continue
            if exclude_none and value is None:
                continue
            result[key] = SerializationUtils._convert_value(value)
        return result
    
    @staticmethod
    def _convert_values(data: Any) -> Any:
        """Recursively convert values in data structure."""
        if isinstance(data, dict):
            return {k: SerializationUtils._convert_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SerializationUtils._convert_values(item) for item in data]
        else:
            return SerializationUtils._convert_value(data)
    
    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert a single value to JSON-serializable format."""
        if value is None:
            return None
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (dict, list)):
            return SerializationUtils._convert_values(value)
        elif hasattr(value, '__dict__'):
            return SerializationUtils._object_to_dict(value, [], False)
        else:
            return value
    
    @staticmethod
    def _basic_to_dict(obj: Any) -> Any:
        """Convert basic type to dict representation."""
        return SerializationUtils._convert_value(obj)
    
    @staticmethod
    def to_dict_list(
        objects: List[Any],
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert list of objects to list of dictionaries.
        
        Args:
            objects: List of objects to serialize
            exclude_none: Exclude None values
            exclude_unset: Exclude unset fields (Pydantic)
            exclude_fields: List of field names to exclude
        
        Returns:
            List of dictionary representations
        """
        return [
            SerializationUtils.to_dict(
                obj,
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_fields=exclude_fields
            )
            for obj in objects
        ]


# Convenience functions
def to_dict(
    obj: Any,
    exclude_none: bool = False,
    exclude_unset: bool = False,
    exclude_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convert object to dictionary."""
    return SerializationUtils.to_dict(obj, exclude_none, exclude_unset, exclude_fields)


def to_dict_list(
    objects: List[Any],
    exclude_none: bool = False,
    exclude_unset: bool = False,
    exclude_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Convert list of objects to list of dictionaries."""
    return SerializationUtils.to_dict_list(objects, exclude_none, exclude_unset, exclude_fields)

