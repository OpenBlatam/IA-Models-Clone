"""Dictionary utilities."""

from typing import Any, Dict, List, Optional, Union
from functools import reduce


def deep_merge(*dicts: Dict) -> Dict:
    """
    Deep merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge (later ones override earlier)
        
    Returns:
        Merged dictionary
    """
    def merge_two(a: Dict, b: Dict) -> Dict:
        result = a.copy()
        for key, value in b.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_two(result[key], value)
            else:
                result[key] = value
        return result
    
    return reduce(merge_two, dicts, {})


def get_nested_value(d: Dict, path: str, default: Any = None) -> Any:
    """
    Get nested value from dictionary using dot notation.
    
    Args:
        d: Dictionary
        path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path doesn't exist
        
    Returns:
        Value or default
    """
    keys = path.split('.')
    current = d
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(d: Dict, path: str, value: Any) -> None:
    """
    Set nested value in dictionary using dot notation.
    
    Args:
        d: Dictionary
        path: Dot-separated path
        value: Value to set
    """
    keys = path.split('.')
    current = d
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def flatten_dict(d: Dict, separator: str = '.', prefix: str = '') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        separator: Separator for keys
        prefix: Optional prefix for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def filter_dict(d: Dict, keys: List[str]) -> Dict:
    """
    Filter dictionary to only include specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def exclude_dict(d: Dict, keys: List[str]) -> Dict:
    """
    Exclude specified keys from dictionary.
    
    Args:
        d: Dictionary to filter
        keys: Keys to exclude
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k not in keys}


def dict_to_list(d: Dict, key_name: str = 'key', value_name: str = 'value') -> List[Dict]:
    """
    Convert dictionary to list of key-value pairs.
    
    Args:
        d: Dictionary
        key_name: Name for key field
        value_name: Name for value field
        
    Returns:
        List of dictionaries
    """
    return [{key_name: k, value_name: v} for k, v in d.items()]


def list_to_dict(lst: List[Dict], key_field: str, value_field: Optional[str] = None) -> Dict:
    """
    Convert list of dictionaries to dictionary.
    
    Args:
        lst: List of dictionaries
        key_field: Field to use as key
        value_field: Field to use as value (None uses entire dict)
        
    Returns:
        Dictionary
    """
    if value_field:
        return {item[key_field]: item[value_field] for item in lst}
    else:
        return {item[key_field]: item for item in lst}

