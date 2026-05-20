"""
Collection Utilities
====================

Utilities for working with collections (lists, dicts, etc.).
"""

from typing import List, Dict, Any, Callable, Optional, TypeVar

T = TypeVar('T')


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten nested list.
    
    Args:
        nested_list: Nested list
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def group_by(items: List[T], key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
    """
    Group items by key function.
    
    Args:
        items: List of items
        key_func: Function to extract key
        
    Returns:
        Dictionary grouped by key
    """
    grouped = {}
    for item in items:
        key = key_func(item)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    return grouped


def unique(items: List[T]) -> List[T]:
    """
    Get unique items from list (preserves order).
    
    Args:
        items: List of items
        
    Returns:
        List of unique items
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Deep merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to only include specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def exclude_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Exclude specified keys from dictionary.
    
    Args:
        d: Dictionary to filter
        keys: Keys to exclude
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k not in keys}

