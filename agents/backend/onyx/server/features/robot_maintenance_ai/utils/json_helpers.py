"""
JSON helper functions for safe parsing and serialization.
Consolidates repetitive JSON operations across the codebase.
"""

import json
from typing import Any, Dict, List, Optional


def safe_json_loads(json_str: Optional[str], default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback to default value.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails or string is None/empty
    
    Returns:
        Parsed JSON object or default value
    """
    if not json_str:
        return default if default is not None else {}
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        default: Default JSON string if serialization fails
    
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


def safe_json_dumps_or_empty(obj: Any) -> str:
    """
    Safely serialize object to JSON string, defaulting to empty object.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string (empty object if serialization fails)
    """
    return safe_json_dumps(obj, default="{}")


def safe_json_dumps_formatted(
    obj: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = str,
    default_fallback: str = "{}"
) -> str:
    """
    Safely serialize object to JSON string with formatting options.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
        default: Function to handle non-serializable objects (default: str)
        default_fallback: Fallback JSON string if serialization fails
    
    Returns:
        JSON string or fallback
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=default)
    except (TypeError, ValueError):
        return default_fallback



