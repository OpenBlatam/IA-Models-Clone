"""
Data Transformer
================

Advanced data transformation utilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class DataTransformer:
    """Advanced data transformer."""
    
    @staticmethod
    def flatten_dict(data: Dict[str, Any], separator: str = ".", prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Dictionary to flatten
            separator: Key separator
            prefix: Key prefix
            
        Returns:
            Flattened dictionary
        """
        items = []
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                items.extend(DataTransformer.flatten_dict(value, separator, new_key).items())
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    @staticmethod
    def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """
        Unflatten dictionary.
        
        Args:
            data: Flattened dictionary
            separator: Key separator
            
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
    def transform_keys(data: Dict[str, Any], transformer: Callable[[str], str]) -> Dict[str, Any]:
        """
        Transform dictionary keys.
        
        Args:
            data: Dictionary to transform
            transformer: Key transformation function
            
        Returns:
            Transformed dictionary
        """
        if isinstance(data, dict):
            return {
                transformer(key): DataTransformer.transform_keys(value, transformer)
                if isinstance(value, dict) else value
                for key, value in data.items()
            }
        return data
    
    @staticmethod
    def camel_to_snake(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert camelCase keys to snake_case."""
        def transformer(key: str) -> str:
            result = []
            for i, char in enumerate(key):
                if char.isupper() and i > 0:
                    result.append("_")
                result.append(char.lower())
            return "".join(result)
        
        return DataTransformer.transform_keys(data, transformer)
    
    @staticmethod
    def snake_to_camel(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert snake_case keys to camelCase."""
        def transformer(key: str) -> str:
            parts = key.split("_")
            return parts[0] + "".join(word.capitalize() for word in parts[1:])
        
        return DataTransformer.transform_keys(data, transformer)
    
    @staticmethod
    def filter_dict(data: Dict[str, Any], keys: List[str], include: bool = True) -> Dict[str, Any]:
        """
        Filter dictionary by keys.
        
        Args:
            data: Dictionary to filter
            keys: List of keys
            include: If True, include only these keys; if False, exclude them
            
        Returns:
            Filtered dictionary
        """
        if include:
            return {key: data[key] for key in keys if key in data}
        else:
            return {key: value for key, value in data.items() if key not in keys}
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """
        Merge multiple dictionaries.
        
        Args:
            *dicts: Dictionaries to merge
            deep: If True, deep merge; if False, shallow merge
            
        Returns:
            Merged dictionary
        """
        result = {}
        for d in dicts:
            if deep:
                result = DataTransformer._deep_merge(result, d)
            else:
                result.update(d)
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DataTransformer._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def normalize_types(data: Any) -> Any:
        """
        Normalize data types (e.g., Decimal to float, datetime to string).
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        if isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {key: DataTransformer.normalize_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [DataTransformer.normalize_types(item) for item in data]
        else:
            return data
    
    @staticmethod
    def group_by(data: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Group list of dictionaries by key.
        
        Args:
            data: List of dictionaries
            key: Key to group by
            
        Returns:
            Grouped dictionary
        """
        result = {}
        for item in data:
            group_key = item.get(key)
            if group_key not in result:
                result[group_key] = []
            result[group_key].append(item)
        return result
    
    @staticmethod
    def sort_by(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
        """
        Sort list of dictionaries by key.
        
        Args:
            data: List of dictionaries
            key: Key to sort by
            reverse: Reverse sort order
            
        Returns:
            Sorted list
        """
        return sorted(data, key=lambda x: x.get(key, ""), reverse=reverse)




