"""
Data transformation and mapping helper functions.

This module provides utilities for transforming lists, dictionaries,
and other data structures with common patterns.
"""

from typing import Any, List, Dict, Optional, Callable, Union
from functools import partial


def map_list(
    items: List[Any],
    transform: Callable[[Any], Any],
    filter_none: bool = False
) -> List[Any]:
    """
    Transform a list of items using a transformation function.
    
    Args:
        items: List of items to transform
        transform: Function to apply to each item
        filter_none: Whether to filter out None results
    
    Returns:
        List of transformed items
    
    Example:
        names = map_list(artists, lambda a: a.get("name") if isinstance(a, dict) else str(a))
        ids = map_list(tracks, lambda t: t["id"], filter_none=True)
    """
    results = [transform(item) for item in items]
    
    if filter_none:
        results = [r for r in results if r is not None]
    
    return results


def map_dict(
    data: Dict[str, Any],
    key_transform: Optional[Callable[[str], str]] = None,
    value_transform: Optional[Callable[[Any], Any]] = None,
    filter_none: bool = False
) -> Dict[str, Any]:
    """
    Transform dictionary keys and/or values.
    
    Args:
        data: Dictionary to transform
        key_transform: Optional function to transform keys
        value_transform: Optional function to transform values
        filter_none: Whether to filter out None values
    
    Returns:
        Transformed dictionary
    
    Example:
        # Transform values only
        normalized = map_dict(data, value_transform=lambda v: v.lower() if isinstance(v, str) else v)
        
        # Transform keys to snake_case
        snake_case = map_dict(data, key_transform=lambda k: k.replace(" ", "_").lower())
    """
    result = {}
    
    for key, value in data.items():
        # Transform key if provided
        new_key = key_transform(key) if key_transform else key
        
        # Transform value if provided
        new_value = value_transform(value) if value_transform else value
        
        # Filter None if requested
        if filter_none and new_value is None:
            continue
        
        result[new_key] = new_value
    
    return result


def filter_dict(
    data: Dict[str, Any],
    predicate: Callable[[str, Any], bool],
    invert: bool = False
) -> Dict[str, Any]:
    """
    Filter dictionary items based on a predicate function.
    
    Args:
        data: Dictionary to filter
        predicate: Function (key, value) -> bool
        invert: If True, keep items where predicate is False
    
    Returns:
        Filtered dictionary
    
    Example:
        # Keep only non-None values
        clean = filter_dict(data, lambda k, v: v is not None)
        
        # Keep only specific keys
        subset = filter_dict(data, lambda k, v: k in ["id", "name", "artists"])
    """
    if invert:
        return {k: v for k, v in data.items() if not predicate(k, v)}
    return {k: v for k, v in data.items() if predicate(k, v)}


def group_by(
    items: List[Any],
    key_func: Callable[[Any], str],
    value_transform: Optional[Callable[[Any], Any]] = None
) -> Dict[str, List[Any]]:
    """
    Group items by a key function.
    
    Args:
        items: List of items to group
        key_func: Function to extract grouping key from item
        value_transform: Optional function to transform values before grouping
    
    Returns:
        Dictionary mapping keys to lists of items
    
    Example:
        # Group tracks by genre
        by_genre = group_by(tracks, lambda t: t.get("genre", "unknown"))
        
        # Group with transformation
        by_artist = group_by(tracks, lambda t: t["artists"][0] if t.get("artists") else "unknown")
    """
    result: Dict[str, List[Any]] = {}
    
    for item in items:
        key = key_func(item)
        value = value_transform(item) if value_transform else item
        
        if key not in result:
            result[key] = []
        result[key].append(value)
    
    return result


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        data: Nested dictionary to flatten
        separator: Separator for nested keys (default: ".")
        prefix: Optional prefix for all keys
    
    Returns:
        Flattened dictionary
    
    Example:
        nested = {"user": {"profile": {"name": "John"}}}
        flat = flatten_dict(nested)  # {"user.profile.name": "John"}
    """
    result = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            nested = flatten_dict(value, separator=separator, prefix=new_key)
            result.update(nested)
        else:
            result[new_key] = value
    
    return result


def unflatten_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot notation keys.
    
    Args:
        data: Flattened dictionary
        separator: Separator used in keys (default: ".")
    
    Returns:
        Nested dictionary
    
    Example:
        flat = {"user.profile.name": "John", "user.profile.age": 30}
        nested = unflatten_dict(flat)  # {"user": {"profile": {"name": "John", "age": 30}}}
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set final value
        current[parts[-1]] = value
    
    return result


def extract_nested_values(
    data: Dict[str, Any],
    keys: List[Union[str, List[str]]],
    default: Any = None
) -> Dict[str, Any]:
    """
    Extract multiple nested values from a dictionary.
    
    Args:
        data: Dictionary to extract from
        keys: List of keys (strings) or key paths (lists)
        default: Default value for missing keys
    
    Returns:
        Dictionary mapping keys to extracted values
    
    Example:
        values = extract_nested_values(
            track_data,
            ["id", "name", ["album", "name"], ["artists", 0, "name"]],
            default=None
        )
    """
    from .object_helpers import safe_get_attribute
    
    result = {}
    
    for key_spec in keys:
        if isinstance(key_spec, str):
            # Simple key
            result[key_spec] = safe_get_attribute(data, key_spec, default=default)
        elif isinstance(key_spec, list):
            # Nested path
            path = ".".join(str(k) for k in key_spec)
            result[path] = safe_get_attribute(data, path, default=default)
    
    return result


def transform_track_list(
    tracks: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    field_transforms: Optional[Dict[str, Callable]] = None
) -> List[Dict[str, Any]]:
    """
    Transform a list of tracks with field filtering and transformation.
    
    Args:
        tracks: List of track dictionaries
        include_fields: Optional list of fields to include (all if None)
        exclude_fields: Optional list of fields to exclude
        field_transforms: Optional dict mapping field names to transform functions
    
    Returns:
        List of transformed track dictionaries
    
    Example:
        simplified = transform_track_list(
            tracks,
            include_fields=["id", "name", "artists"],
            field_transforms={
                "artists": lambda a: [artist.get("name") for artist in a] if isinstance(a, list) else a
            }
        )
    """
    result = []
    
    for track in tracks:
        transformed = {}
        
        # Determine fields to process
        fields = include_fields if include_fields else list(track.keys())
        
        for field in fields:
            # Skip excluded fields
            if exclude_fields and field in exclude_fields:
                continue
            
            # Get value
            value = track.get(field)
            
            # Apply transformation if provided
            if field_transforms and field in field_transforms:
                value = field_transforms[field](value)
            
            transformed[field] = value
        
        result.append(transformed)
    
    return result








