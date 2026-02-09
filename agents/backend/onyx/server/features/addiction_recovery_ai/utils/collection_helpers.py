"""
Collection manipulation utilities
Helper functions for working with lists, dicts, etc.
"""

from typing import List, Dict, Any, Callable, Optional, TypeVar

T = TypeVar('T')


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def unique_list(items: List[T], key_func: Optional[Callable[[T], Any]] = None) -> List[T]:
    """
    Get unique items from list
    
    Args:
        items: List of items
        key_func: Optional function to extract key for uniqueness
    
    Returns:
        List of unique items
    """
    if not key_func:
        return list(dict.fromkeys(items))
    
    seen = set()
    result = []
    
    for item in items:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
    
    Returns:
        Merged dictionary (later dicts override earlier ones)
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
    
    Returns:
        Deeply merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_value(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Get nested value from dictionary using dot notation
    
    Args:
        data: Dictionary
        path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found
        separator: Path separator
    
    Returns:
        Value at path or default
    """
    keys = path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(
    data: Dict[str, Any],
    path: str,
    value: Any,
    separator: str = "."
) -> Dict[str, Any]:
    """
    Set nested value in dictionary using dot notation
    
    Args:
        data: Dictionary
        path: Dot-separated path
        value: Value to set
        separator: Path separator
    
    Returns:
        Dictionary with value set
    """
    keys = path.split(separator)
    result = data.copy()
    current = result
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return result

