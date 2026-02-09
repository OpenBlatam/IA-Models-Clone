"""
Default value and fallback helper functions.

This module provides utilities for handling default values,
fallbacks, and optional value extraction with consistent patterns.
"""

from typing import Any, Optional, Callable, List, Dict, Union


def get_or_default(
    value: Optional[Any],
    default: Any,
    transform: Optional[Callable[[Any], Any]] = None
) -> Any:
    """
    Get value or return default, with optional transformation.
    
    Args:
        value: Value to check (may be None)
        default: Default value if value is None
        transform: Optional function to transform value before returning
    
    Returns:
        Transformed value or default
    
    Example:
        name = get_or_default(track.get("name"), "Unknown")
        duration = get_or_default(track.get("duration_ms"), 0, transform=lambda x: x / 1000)
    """
    if value is None:
        return default
    
    if transform:
        return transform(value)
    
    return value


def get_first_not_none(*values: Any, default: Any = None) -> Any:
    """
    Get the first non-None value from a list of values.
    
    Args:
        *values: Variable number of values to check
        default: Default value if all are None
    
    Returns:
        First non-None value or default
    
    Example:
        track_id = get_first_not_none(
            request.track_id,
            request.track_name,
            default=None
        )
        
        limit = get_first_not_none(
            request.limit,
            query_params.get("limit"),
            20  # default
        )
    """
    for value in values:
        if value is not None:
            return value
    return default


def coalesce(*values: Any) -> Any:
    """
    Alias for get_first_not_none - returns first non-None value.
    
    Common pattern in many languages (SQL COALESCE, JavaScript ??, etc.)
    
    Args:
        *values: Variable number of values to check
    
    Returns:
        First non-None value or None if all are None
    
    Example:
        name = coalesce(track.get("name"), track.get("title"), "Unknown")
    """
    return get_first_not_none(*values, default=None)


def get_nested_or_default(
    data: Union[Dict[str, Any], Any],
    path: Union[str, List[str]],
    default: Any = None
) -> Any:
    """
    Get nested value or return default.
    
    Convenience wrapper around safe_get_attribute for better readability.
    
    Args:
        data: Dictionary or object to search
        path: Dot-notation path or list of keys
        default: Default value if not found
    
    Returns:
        Value at path or default
    
    Example:
        name = get_nested_or_default(track, "album.name", default="Unknown")
        artist = get_nested_or_default(track, ["artists", 0, "name"], default="Unknown")
    """
    from .object_helpers import safe_get_attribute
    
    if isinstance(path, list):
        path = ".".join(str(p) for p in path)
    
    return safe_get_attribute(data, path, default=default)


def with_defaults(
    data: Dict[str, Any],
    defaults: Dict[str, Any],
    override: bool = False
) -> Dict[str, Any]:
    """
    Merge defaults into data dictionary.
    
    Args:
        data: Dictionary to add defaults to
        defaults: Dictionary of default values
        override: If True, defaults override existing values
    
    Returns:
        Dictionary with defaults applied
    
    Example:
        config = with_defaults(
            user_config,
            {"limit": 20, "offset": 0, "sort": "name"}
        )
    """
    if override:
        return {**defaults, **data}
    return {**defaults, **data}


def ensure_keys(
    data: Dict[str, Any],
    required_keys: List[str],
    default: Any = None
) -> Dict[str, Any]:
    """
    Ensure dictionary has all required keys with defaults.
    
    Args:
        data: Dictionary to check
        required_keys: List of required keys
        default: Default value for missing keys
    
    Returns:
        Dictionary with all required keys
    
    Example:
        track = ensure_keys(
            track_data,
            ["id", "name", "artists", "duration_ms"],
            default=None
        )
    """
    result = data.copy()
    
    for key in required_keys:
        if key not in result:
            result[key] = default
    
    return result


def extract_with_defaults(
    data: Dict[str, Any],
    field_mapping: Dict[str, Union[str, List[str], tuple]],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract fields from data with optional renaming and defaults.
    
    Args:
        data: Source dictionary
        field_mapping: Dict mapping output keys to input paths
                     - String: direct key
                     - List/tuple: nested path
        defaults: Optional dictionary of default values
    
    Returns:
        Dictionary with extracted and renamed fields
    
    Example:
        result = extract_with_defaults(
            track_data,
            {
                "track_id": "id",
                "track_name": "name",
                "album_name": ["album", "name"],
                "artist_name": ["artists", 0, "name"]
            },
            defaults={"duration": 0}
        )
    """
    from .object_helpers import safe_get_attribute
    
    result = {}
    
    # Apply defaults first
    if defaults:
        result.update(defaults)
    
    # Extract fields
    for output_key, input_path in field_mapping.items():
        if isinstance(input_path, str):
            # Direct key
            result[output_key] = safe_get_attribute(data, input_path, default=None)
        elif isinstance(input_path, (list, tuple)):
            # Nested path
            path = ".".join(str(p) for p in input_path)
            result[output_key] = safe_get_attribute(data, path, default=None)
    
    return result








