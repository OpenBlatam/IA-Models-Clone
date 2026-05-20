"""
Object conversion and transformation helper functions.

This module provides utilities for converting objects to dictionaries,
handling mixed object/dict types, and transforming data structures.
"""

from typing import Any, List, Dict, Optional, Union, Callable


def to_dict(obj: Any, fallback: Optional[Callable] = None) -> Union[Dict[str, Any], Any]:
    """
    Convert an object to dictionary if it has to_dict() method, otherwise return as-is.
    
    Handles multiple object types:
    - Objects with to_dict() method -> calls to_dict()
    - Objects with model_dump() method (Pydantic v2) -> calls model_dump()
    - Objects with dict() method (Pydantic v1) -> calls dict()
    - Already dictionaries -> returns as-is
    - Other types -> returns as-is or uses fallback
    
    Args:
        obj: Object to convert
        fallback: Optional callable to use if no conversion method found
    
    Returns:
        Dictionary if conversion possible, otherwise original object or fallback result
    
    Example:
        result = to_dict(my_object)  # Returns dict or object
        playlist = to_dict(playlist_obj)  # Handles Pydantic models
    """
    # Already a dictionary
    if isinstance(obj, dict):
        return obj
    
    # Try to_dict() method (common pattern)
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    
    # Try model_dump() method (Pydantic v2)
    if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
        return obj.model_dump()
    
    # Try dict() method (Pydantic v1)
    if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
        return obj.dict()
    
    # Try __dict__ attribute
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    
    # Use fallback if provided
    if fallback:
        return fallback(obj)
    
    # Return as-is
    return obj


def to_dict_list(
    items: List[Any],
    fallback: Optional[Callable] = None
) -> List[Union[Dict[str, Any], Any]]:
    """
    Convert a list of objects to a list of dictionaries.
    
    Each item is converted using to_dict(). Items that are already
    dictionaries are left as-is.
    
    Args:
        items: List of objects to convert
        fallback: Optional callable to use for items without conversion method
    
    Returns:
        List of dictionaries (or mixed if conversion not possible)
    
    Example:
        recommendations = [rec1, rec2, rec3]  # Objects
        recommendations_dict = to_dict_list(recommendations)  # List of dicts
    """
    return [to_dict(item, fallback=fallback) for item in items]


def extract_attributes(
    obj: Any,
    attributes: List[str],
    default: Any = None
) -> Dict[str, Any]:
    """
    Extract specific attributes from an object into a dictionary.
    
    Args:
        obj: Object to extract attributes from
        attributes: List of attribute names to extract
        default: Default value for missing attributes
    
    Returns:
        Dictionary with extracted attributes
    
    Example:
        data = extract_attributes(
            result,
            ["track_id", "track_name", "artists"],
            default=None
        )
    """
    result = {}
    for attr in attributes:
        if hasattr(obj, attr):
            result[attr] = getattr(obj, attr, default)
        else:
            result[attr] = default
    return result


def safe_get_attribute(
    obj: Any,
    attribute: str,
    default: Any = None
) -> Any:
    """
    Safely get an attribute from an object.
    
    Works with:
    - Objects with attributes
    - Dictionaries with keys
    - Nested attribute paths (e.g., "user.profile.name")
    
    Args:
        obj: Object or dictionary
        attribute: Attribute name or path (supports dot notation)
        default: Default value if attribute not found
    
    Returns:
        Attribute value or default
    
    Example:
        name = safe_get_attribute(data, "track_basic_info.name")
        artists = safe_get_attribute(result, "artists", [])
    """
    # Handle dot notation for nested attributes
    if '.' in attribute:
        parts = attribute.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
            if current is None:
                return default
        return current
    
    # Single attribute
    if isinstance(obj, dict):
        return obj.get(attribute, default)
    
    if hasattr(obj, attribute):
        return getattr(obj, attribute, default)
    
    return default


def normalize_to_dict(
    data: Any,
    recursive: bool = True
) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Recursively normalize data structure to dictionaries.
    
    Converts objects to dicts, handles lists, and can recursively
    process nested structures.
    
    Args:
        data: Data to normalize (object, dict, list, etc.)
        recursive: Whether to recursively process nested structures
    
    Returns:
        Normalized data structure
    
    Example:
        normalized = normalize_to_dict(complex_object, recursive=True)
    """
    # Already a dictionary
    if isinstance(data, dict):
        if recursive:
            return {k: normalize_to_dict(v, recursive=True) for k, v in data.items()}
        return data
    
    # List - normalize each item
    if isinstance(data, list):
        return [normalize_to_dict(item, recursive=recursive) for item in data]
    
    # Try to convert object to dict
    converted = to_dict(data)
    if isinstance(converted, dict):
        if recursive:
            return {k: normalize_to_dict(v, recursive=True) for k, v in converted.items()}
        return converted
    
    # Return as-is
    return data








