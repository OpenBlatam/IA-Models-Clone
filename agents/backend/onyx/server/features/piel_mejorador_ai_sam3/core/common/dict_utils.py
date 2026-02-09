"""
Dictionary Utilities for Piel Mejorador AI SAM3
==============================================

Unified dictionary operations and utilities.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from copy import deepcopy

logger = logging.getLogger(__name__)


class DictUtils:
    """Unified dictionary utilities."""
    
    @staticmethod
    def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge multiple dictionaries.
        
        Args:
            *dicts: Dictionaries to merge (later ones override earlier ones)
            
        Returns:
            Merged dictionary
        """
        if not dicts:
            return {}
        
        result = {}
        
        for d in dicts:
            if not isinstance(d, dict):
                continue
            
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = DictUtils.deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
        
        return result
    
    @staticmethod
    def merge_with_priority(
        *dicts: Dict[str, Any],
        priorities: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Merge dictionaries with priority (higher priority overrides lower).
        
        Args:
            *dicts: Dictionaries to merge
            priorities: Optional priority list (same length as dicts)
            
        Returns:
            Merged dictionary
        """
        if not dicts:
            return {}
        
        if priorities and len(priorities) == len(dicts):
            # Sort by priority (higher first)
            sorted_pairs = sorted(
                zip(dicts, priorities),
                key=lambda x: x[1],
                reverse=True
            )
            dicts = [d for d, _ in sorted_pairs]
        
        return DictUtils.deep_merge(*dicts)
    
    @staticmethod
    def get_nested(
        data: Dict[str, Any],
        path: str,
        default: Any = None,
        separator: str = "."
    ) -> Any:
        """
        Get nested value from dictionary using dot notation.
        
        Args:
            data: Dictionary
            path: Dot-separated path (e.g., "user.profile.name")
            default: Default value if not found
            separator: Path separator
            
        Returns:
            Value or default
        """
        keys = path.split(separator)
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        
        return current
    
    @staticmethod
    def set_nested(
        data: Dict[str, Any],
        path: str,
        value: Any,
        separator: str = "."
    ) -> None:
        """
        Set nested value in dictionary using dot notation.
        
        Args:
            data: Dictionary to modify
            path: Dot-separated path
            value: Value to set
            separator: Path separator
        """
        keys = path.split(separator)
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @staticmethod
    def flatten(
        data: Dict[str, Any],
        separator: str = ".",
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Dictionary to flatten
            separator: Separator for keys
            prefix: Optional prefix for keys
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(DictUtils.flatten(value, separator, new_key))
            else:
                result[new_key] = value
        
        return result
    
    @staticmethod
    def unflatten(
        data: Dict[str, Any],
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Unflatten dictionary with dot notation keys.
        
        Args:
            data: Flattened dictionary
            separator: Separator in keys
            
        Returns:
            Nested dictionary
        """
        result = {}
        
        for key, value in data.items():
            DictUtils.set_nested(result, key, value, separator)
        
        return result
    
    @staticmethod
    def filter_keys(
        data: Dict[str, Any],
        keys: List[str],
        include: bool = True
    ) -> Dict[str, Any]:
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
            return {k: v for k, v in data.items() if k in keys}
        else:
            return {k: v for k, v in data.items() if k not in keys}
    
    @staticmethod
    def filter_values(
        data: Dict[str, Any],
        predicate: Callable[[Any], bool]
    ) -> Dict[str, Any]:
        """
        Filter dictionary by value predicate.
        
        Args:
            data: Dictionary to filter
            predicate: Function that returns True to keep value
            
        Returns:
            Filtered dictionary
        """
        return {k: v for k, v in data.items() if predicate(v)}
    
    @staticmethod
    def remove_none(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from dictionary.
        
        Args:
            data: Dictionary to clean
            
        Returns:
            Dictionary without None values
        """
        return DictUtils.filter_values(data, lambda v: v is not None)
    
    @staticmethod
    def remove_empty(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove empty values from dictionary.
        
        Args:
            data: Dictionary to clean
            
        Returns:
            Dictionary without empty values
        """
        return DictUtils.filter_values(data, lambda v: v not in (None, "", [], {}))
    
    @staticmethod
    def update_nested(
        base: Dict[str, Any],
        update: Dict[str, Any],
        deep: bool = True
    ) -> Dict[str, Any]:
        """
        Update base dictionary with update dictionary.
        
        Args:
            base: Base dictionary
            update: Update dictionary
            deep: Whether to deep merge
            
        Returns:
            Updated dictionary
        """
        if deep:
            return DictUtils.deep_merge(base, update)
        else:
            result = base.copy()
            result.update(update)
            return result


# Convenience functions
def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries."""
    return DictUtils.deep_merge(*dicts)


def get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested value."""
    return DictUtils.get_nested(data, path, default)


def set_nested(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested value."""
    DictUtils.set_nested(data, path, value)


def remove_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values."""
    return DictUtils.remove_none(data)




