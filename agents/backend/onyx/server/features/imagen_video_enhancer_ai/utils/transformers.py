"""
Data Transformers
=================

Utilities for transforming data between different formats.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path


class DataTransformer:
    """Base class for data transformers."""
    
    @staticmethod
    def transform(data: Any, **options) -> Any:
        """Transform data."""
        raise NotImplementedError


class DictTransformer(DataTransformer):
    """Transform dictionary data."""
    
    @staticmethod
    def flatten(
        data: Dict[str, Any],
        separator: str = ".",
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Nested dictionary
            separator: Key separator
            prefix: Key prefix
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(DictTransformer.flatten(value, separator, new_key))
            else:
                result[new_key] = value
        
        return result
    
    @staticmethod
    def unflatten(
        data: Dict[str, Any],
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Unflatten dictionary with nested keys.
        
        Args:
            data: Flattened dictionary
            separator: Key separator
            
        Returns:
            Nested dictionary
        """
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result
    
    @staticmethod
    def filter_keys(
        data: Dict[str, Any],
        keys: List[str],
        include: bool = True
    ) -> Dict[str, Any]:
        """
        Filter dictionary keys.
        
        Args:
            data: Dictionary to filter
            keys: Keys to filter
            include: If True, include only these keys; if False, exclude them
            
        Returns:
            Filtered dictionary
        """
        if include:
            return {k: v for k, v in data.items() if k in keys}
        else:
            return {k: v for k, v in data.items() if k not in keys}
    
    @staticmethod
    def map_values(
        data: Dict[str, Any],
        mapper: Callable[[Any], Any]
    ) -> Dict[str, Any]:
        """
        Map dictionary values using a function.
        
        Args:
            data: Dictionary to map
            mapper: Mapping function
            
        Returns:
            Dictionary with mapped values
        """
        return {k: mapper(v) for k, v in data.items()}


class ListTransformer(DataTransformer):
    """Transform list data."""
    
    @staticmethod
    def chunk(
        items: List[Any],
        chunk_size: int
    ) -> List[List[Any]]:
        """
        Split list into chunks.
        
        Args:
            items: List to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of chunks
        """
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def group_by(
        items: List[Dict[str, Any]],
        key: str
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Group list items by key.
        
        Args:
            items: List of dictionaries
            key: Key to group by
            
        Returns:
            Grouped dictionary
        """
        result = {}
        
        for item in items:
            group_key = item.get(key)
            if group_key not in result:
                result[group_key] = []
            result[group_key].append(item)
        
        return result
    
    @staticmethod
    def unique(
        items: List[Any],
        key: Optional[Callable[[Any], Any]] = None
    ) -> List[Any]:
        """
        Get unique items from list.
        
        Args:
            items: List of items
            key: Optional key function for uniqueness
            
        Returns:
            List of unique items
        """
        if key is None:
            return list(dict.fromkeys(items))
        
        seen = set()
        result = []
        
        for item in items:
            item_key = key(item)
            if item_key not in seen:
                seen.add(item_key)
                result.append(item)
        
        return result


class DateTimeTransformer(DataTransformer):
    """Transform datetime data."""
    
    @staticmethod
    def to_iso_string(dt: datetime) -> str:
        """Convert datetime to ISO string."""
        return dt.isoformat()
    
    @staticmethod
    def from_iso_string(iso_string: str) -> datetime:
        """Parse ISO string to datetime."""
        return datetime.fromisoformat(iso_string)
    
    @staticmethod
    def to_timestamp(dt: datetime) -> float:
        """Convert datetime to timestamp."""
        return dt.timestamp()
    
    @staticmethod
    def from_timestamp(timestamp: float) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(timestamp)




