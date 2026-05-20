"""
Type checking and conversion helper functions.

This module provides utilities for type checking, type conversion,
and type-safe operations with common patterns.
"""

from typing import Any, List, Dict, Optional, Union, Type, TypeVar
from collections.abc import Collection, Iterable

T = TypeVar('T')


def is_dict(value: Any) -> bool:
    """
    Check if value is a dictionary.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a dict
    
    Example:
        if is_dict(data):
            process_dict(data)
    """
    return isinstance(value, dict)


def is_list(value: Any) -> bool:
    """
    Check if value is a list.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a list
    
    Example:
        if is_list(items):
            process_list(items)
    """
    return isinstance(value, list)


def is_str(value: Any) -> bool:
    """
    Check if value is a string.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a string
    
    Example:
        if is_str(name):
            process_string(name)
    """
    return isinstance(value, str)


def is_int(value: Any) -> bool:
    """
    Check if value is an integer.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is an int
    
    Example:
        if is_int(count):
            process_int(count)
    """
    return isinstance(value, int)


def is_none(value: Any) -> bool:
    """
    Check if value is None.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is None
    
    Example:
        if is_none(data):
            use_default()
    """
    return value is None


def is_not_none(value: Any) -> bool:
    """
    Check if value is not None.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is not None
    
    Example:
        if is_not_none(data):
            process(data)
    """
    return value is not None


def is_empty(value: Any) -> bool:
    """
    Check if value is empty (None, empty string, empty list, empty dict, etc.).
    
    Args:
        value: Value to check
    
    Returns:
        True if value is empty
    
    Example:
        if is_empty(items):
            return []
    """
    if value is None:
        return True
    
    if isinstance(value, str):
        return len(value.strip()) == 0
    
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    
    return False


def is_not_empty(value: Any) -> bool:
    """
    Check if value is not empty.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is not empty
    
    Example:
        if is_not_empty(items):
            process(items)
    """
    return not is_empty(value)


def safe_type_check(value: Any, *types: Type) -> bool:
    """
    Safely check if value is one of the specified types.
    
    Args:
        value: Value to check
        *types: Types to check against
    
    Returns:
        True if value is instance of any of the types
    
    Example:
        if safe_type_check(data, dict, list):
            process_collection(data)
    """
    if not types:
        return False
    
    return isinstance(value, types)


def as_type(value: Any, target_type: Type[T], default: Optional[T] = None) -> Optional[T]:
    """
    Convert value to target type if possible, otherwise return default.
    
    Args:
        value: Value to convert
        target_type: Target type
        default: Default value if conversion fails
    
    Returns:
        Converted value or default
    
    Example:
        count = as_type(data.get("count"), int, default=0)
        name = as_type(data.get("name"), str, default="")
    """
    if value is None:
        return default
    
    if isinstance(value, target_type):
        return value
    
    try:
        if target_type == int:
            return int(value)
        elif target_type == str:
            return str(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif target_type == list:
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]
        elif target_type == dict:
            if isinstance(value, dict):
                return value
            return default
    except (ValueError, TypeError):
        pass
    
    return default


def ensure_type(value: Any, target_type: Type[T], default: T) -> T:
    """
    Ensure value is of target type, using default if not.
    
    Args:
        value: Value to ensure
        target_type: Target type
        default: Default value if value is not of target type
    
    Returns:
        Value if it's the right type, otherwise default
    
    Example:
        limit = ensure_type(request.limit, int, 20)
        name = ensure_type(data.get("name"), str, "Unknown")
    """
    if isinstance(value, target_type):
        return value
    return default


def get_type_name(value: Any) -> str:
    """
    Get the type name of a value as a string.
    
    Args:
        value: Value to get type name for
    
    Returns:
        Type name as string
    
    Example:
        type_name = get_type_name(data)  # "dict", "list", "str", etc.
    """
    if value is None:
        return "None"
    
    type_obj = type(value)
    return type_obj.__name__


def is_collection(value: Any) -> bool:
    """
    Check if value is a collection (list, dict, set, tuple, etc.).
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a collection
    
    Example:
        if is_collection(data):
            iterate_collection(data)
    """
    return isinstance(value, (list, dict, set, tuple, Collection)) and not isinstance(value, str)


def is_iterable(value: Any) -> bool:
    """
    Check if value is iterable (but not a string).
    
    Args:
        value: Value to check
    
    Returns:
        True if value is iterable and not a string
    
    Example:
        if is_iterable(items):
            for item in items:
                process(item)
    """
    if isinstance(value, str):
        return False
    
    return isinstance(value, Iterable)








