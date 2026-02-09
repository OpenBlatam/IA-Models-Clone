"""
Data Transformation Utilities for Piel Mejorador AI SAM3
========================================================

Unified data transformation and mapping utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class DataTransformationUtils:
    """Unified data transformation utilities."""
    
    @staticmethod
    def map_dict(
        data: Dict[str, Any],
        mapping: Dict[str, Union[str, Callable[[Any], Any]]]
    ) -> Dict[str, Any]:
        """
        Map dictionary keys and values.
        
        Args:
            data: Source dictionary
            mapping: Mapping from source keys to (target_key, transform_func)
            
        Returns:
            Mapped dictionary
        """
        result = {}
        for source_key, target in mapping.items():
            if source_key not in data:
                continue
            
            if isinstance(target, str):
                # Simple key rename
                result[target] = data[source_key]
            elif callable(target):
                # Transform function
                result[source_key] = target(data[source_key])
            else:
                # Tuple of (target_key, transform_func)
                target_key, transform_func = target
                result[target_key] = transform_func(data[source_key])
        
        return result
    
    @staticmethod
    def transform_dict(
        data: Dict[str, Any],
        transformer: Callable[[str, Any], tuple[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transform dictionary using function.
        
        Args:
            data: Source dictionary
            transformer: Function (key, value) -> (new_key, new_value)
            
        Returns:
            Transformed dictionary
        """
        return dict(transformer(k, v) for k, v in data.items())
    
    @staticmethod
    def flatten_dict(
        data: Dict[str, Any],
        separator: str = ".",
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Nested dictionary
            separator: Separator for keys
            prefix: Optional prefix for keys
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                nested = DataTransformationUtils.flatten_dict(value, separator, new_key)
                result.update(nested)
            else:
                result[new_key] = value
        
        return result
    
    @staticmethod
    def unflatten_dict(
        data: Dict[str, Any],
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Unflatten dictionary (opposite of flatten).
        
        Args:
            data: Flattened dictionary
            separator: Separator used in keys
            
        Returns:
            Nested dictionary
        """
        result = {}
        
        for key, value in data.items():
            parts = key.split(separator)
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result
    
    @staticmethod
    def convert_types(
        data: Dict[str, Any],
        type_map: Dict[str, type]
    ) -> Dict[str, Any]:
        """
        Convert dictionary values to specified types.
        
        Args:
            data: Source dictionary
            type_map: Mapping of keys to target types
            
        Returns:
            Dictionary with converted types
        """
        result = data.copy()
        
        for key, target_type in type_map.items():
            if key in result:
                value = result[key]
                try:
                    if target_type == bool:
                        result[key] = str(value).lower() in ('true', '1', 'yes', 'on')
                    elif target_type == int:
                        result[key] = int(float(value))
                    elif target_type == float:
                        result[key] = float(value)
                    elif target_type == str:
                        result[key] = str(value)
                    else:
                        result[key] = target_type(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {key} to {target_type.__name__}: {e}")
        
        return result
    
    @staticmethod
    def map_list(
        items: List[Any],
        mapper: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Map list items.
        
        Args:
            items: List of items
            mapper: Mapping function
            
        Returns:
            Mapped list
        """
        return [mapper(item) for item in items]
    
    @staticmethod
    def map_list_async(
        items: List[Any],
        mapper: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Map list items (supports async mappers).
        
        Args:
            items: List of items
            mapper: Mapping function (can be async)
            
        Returns:
            Mapped list
        """
        import asyncio
        
        async def map_item(item):
            if asyncio.iscoroutinefunction(mapper):
                return await mapper(item)
            return mapper(item)
        
        # This would need to be called with await in async context
        # For now, return sync version
        return [mapper(item) for item in items]
    
    @staticmethod
    def filter_dict(
        data: Dict[str, Any],
        predicate: Callable[[str, Any], bool]
    ) -> Dict[str, Any]:
        """
        Filter dictionary entries.
        
        Args:
            data: Source dictionary
            predicate: Function (key, value) -> bool
            
        Returns:
            Filtered dictionary
        """
        return {k: v for k, v in data.items() if predicate(k, v)}
    
    @staticmethod
    def rename_keys(
        data: Dict[str, Any],
        key_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Rename dictionary keys.
        
        Args:
            data: Source dictionary
            key_map: Mapping from old keys to new keys
            
        Returns:
            Dictionary with renamed keys
        """
        result = {}
        for old_key, value in data.items():
            new_key = key_map.get(old_key, old_key)
            result[new_key] = value
        return result
    
    @staticmethod
    def select_keys(
        data: Dict[str, Any],
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Select specific keys from dictionary.
        
        Args:
            data: Source dictionary
            keys: Keys to select
            
        Returns:
            Dictionary with selected keys only
        """
        return {k: data[k] for k in keys if k in data}
    
    @staticmethod
    def exclude_keys(
        data: Dict[str, Any],
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Exclude specific keys from dictionary.
        
        Args:
            data: Source dictionary
            keys: Keys to exclude
            
        Returns:
            Dictionary without excluded keys
        """
        return {k: v for k, v in data.items() if k not in keys}
    
    @staticmethod
    def deep_merge(
        base: Dict[str, Any],
        update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge in
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DataTransformationUtils.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def to_dict(
        obj: Any,
        include_none: bool = False
    ) -> Dict[str, Any]:
        """
        Convert object to dictionary.
        
        Args:
            obj: Object to convert
            include_none: Whether to include None values
            
        Returns:
            Dictionary representation
        """
        if isinstance(obj, dict):
            if include_none:
                return obj
            return {k: v for k, v in obj.items() if v is not None}
        
        if hasattr(obj, '__dict__'):
            data = obj.__dict__
            if include_none:
                return data
            return {k: v for k, v in data.items() if v is not None}
        
        if hasattr(obj, '_asdict'):  # NamedTuple
            data = obj._asdict()
            if include_none:
                return data
            return {k: v for k, v in data.items() if v is not None}
        
        # Try dataclass
        try:
            data = asdict(obj)
            if include_none:
                return data
            return {k: v for k, v in data.items() if v is not None}
        except (TypeError, ValueError):
            pass
        
        return {}


# Convenience functions
def map_dict(data: Dict[str, Any], mapping: Dict[str, Union[str, Callable[[Any], Any]]]) -> Dict[str, Any]:
    """Map dictionary."""
    return DataTransformationUtils.map_dict(data, mapping)


def flatten_dict(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Flatten dictionary."""
    return DataTransformationUtils.flatten_dict(data, **kwargs)


def unflatten_dict(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Unflatten dictionary."""
    return DataTransformationUtils.unflatten_dict(data, **kwargs)


def convert_types(data: Dict[str, Any], type_map: Dict[str, type]) -> Dict[str, Any]:
    """Convert types."""
    return DataTransformationUtils.convert_types(data, type_map)


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries."""
    return DataTransformationUtils.deep_merge(base, update)




