"""
Data transformation utilities
Pure functions for transforming data structures
"""

from typing import Dict, Any, List, Callable, Optional


def transform_dict(
    data: Dict[str, Any],
    field_mapping: Dict[str, str],
    default_value: Any = None
) -> Dict[str, Any]:
    """
    Transform dictionary keys using field mapping
    
    Args:
        data: Source dictionary
        field_mapping: Mapping of old_key -> new_key
        default_value: Default value for missing keys
    
    Returns:
        Transformed dictionary
    """
    result = {}
    for old_key, new_key in field_mapping.items():
        result[new_key] = data.get(old_key, default_value)
    
    return result


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        data: Nested dictionary
        separator: Separator for keys
        prefix: Prefix for keys
    
    Returns:
        Flattened dictionary
    """
    result = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        else:
            result[new_key] = value
    
    return result


def nest_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """
    Convert flat dictionary to nested dictionary
    
    Args:
        data: Flat dictionary with keys like "user.name"
        separator: Separator used in keys
    
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


def map_list(
    items: List[Any],
    mapper: Callable[[Any], Any]
) -> List[Any]:
    """
    Map function over list
    
    Args:
        items: List of items
        mapper: Function to apply to each item
    
    Returns:
        Mapped list
    """
    return [mapper(item) for item in items]


def filter_list(
    items: List[Any],
    predicate: Callable[[Any], bool]
) -> List[Any]:
    """
    Filter list using predicate
    
    Args:
        items: List of items
        predicate: Function that returns True for items to keep
    
    Returns:
        Filtered list
    """
    return [item for item in items if predicate(item)]


def group_by(
    items: List[Any],
    key_func: Callable[[Any], str]
) -> Dict[str, List[Any]]:
    """
    Group items by key
    
    Args:
        items: List of items
        key_func: Function to extract key from item
    
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


def pick_fields(
    data: Dict[str, Any],
    fields: List[str]
) -> Dict[str, Any]:
    """
    Pick specific fields from dictionary
    
    Args:
        data: Source dictionary
        fields: List of field names to pick
    
    Returns:
        Dictionary with only picked fields
    """
    return {field: data.get(field) for field in fields if field in data}


def omit_fields(
    data: Dict[str, Any],
    fields: List[str]
) -> Dict[str, Any]:
    """
    Omit specific fields from dictionary
    
    Args:
        data: Source dictionary
        fields: List of field names to omit
    
    Returns:
        Dictionary without omitted fields
    """
    return {key: value for key, value in data.items() if key not in fields}

