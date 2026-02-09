"""
Data Utilities

Utility functions for data manipulation and transformation.
"""

from typing import List, Dict, Any, Optional, Callable
import json


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Unflatten dictionary with dot notation.
    
    Args:
        d: Flattened dictionary
        sep: Separator for keys
    
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def deep_merge(
    base: Dict[str, Any],
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        updates: Updates to merge
    
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def filter_dict(
    d: Dict[str, Any],
    keys: List[str]
) -> Dict[str, Any]:
    """
    Filter dictionary to include only specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
    
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def exclude_dict(
    d: Dict[str, Any],
    keys: List[str]
) -> Dict[str, Any]:
    """
    Exclude specified keys from dictionary.
    
    Args:
        d: Dictionary to filter
        keys: Keys to exclude
    
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k not in keys}


def group_by(
    items: List[Any],
    key_func: Callable[[Any], Any]
) -> Dict[Any, List[Any]]:
    """
    Group items by key function.
    
    Args:
        items: List of items
        key_func: Function to extract key
    
    Returns:
        Dictionary grouped by key
    """
    result = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def chunk_list(
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


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        text: JSON string
        default: Default value if parsing fails
    
    Returns:
        Parsed object or default
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON.
    
    Args:
        obj: Object to serialize
        default: Default string if serialization fails
    
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return default



