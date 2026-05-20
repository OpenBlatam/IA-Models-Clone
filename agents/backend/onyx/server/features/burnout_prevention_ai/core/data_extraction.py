"""
Data Extraction Helpers
=======================
Helper functions for safely extracting data from dictionaries with defaults.

These helpers provide type-safe extraction with automatic type conversion
and validation, reducing the risk of runtime errors from malformed API responses.
"""

from typing import Any, Dict, List, Optional, Union
from .types import JSONDict


def safe_get(
    data: Dict[str, Any],
    key: str,
    default: Any = None,
    expected_type: Optional[type] = None
) -> Any:
    """
    Safely get value from dictionary with type checking.
    
    Args:
        data: Dictionary to extract from
        key: Key to look up
        default: Default value if key not found or type mismatch
        expected_type: Optional type to validate against
        
    Returns:
        Value from dictionary or default
    """
    if not isinstance(data, dict):
        return default
    
    value = data.get(key, default)
    
    if expected_type and value is not None:
        if not isinstance(value, expected_type):
            return default
    
    return value


def safe_get_list(
    data: Dict[str, Any],
    key: str,
    default: Optional[List[Any]] = None
) -> List[Any]:
    """
    Safely get list from dictionary.
    
    Args:
        data: Dictionary to extract from
        key: Key to look up
        default: Default list if key not found or not a list
        
    Returns:
        List from dictionary or default (empty list if default is None)
    """
    if not isinstance(data, dict):
        return default if default is not None else []
    
    if default is None:
        default = []
    
    value = data.get(key, default)
    
    if not isinstance(value, list):
        return default
    
    return value


def safe_get_float(
    data: Dict[str, Any],
    key: str,
    default: float = 0.0
) -> float:
    """
    Safely get float from dictionary.
    
    Args:
        data: Dictionary to extract from
        key: Key to look up
        default: Default float value
        
    Returns:
        Float value or default
    """
    if not isinstance(data, dict):
        return default
    
    value = data.get(key, default)
    
    # Fast path: already a float
    if isinstance(value, float):
        return value
    
    # Fast path: already an int (can be converted to float)
    if isinstance(value, int):
        return float(value)
    
    # Try conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_get_int(
    data: Dict[str, Any],
    key: str,
    default: int = 0
) -> int:
    """
    Safely get int from dictionary.
    
    Args:
        data: Dictionary to extract from
        key: Key to look up
        default: Default int value
        
    Returns:
        Int value or default
    """
    if not isinstance(data, dict):
        return default
    
    value = data.get(key, default)
    
    # Fast path: already an int
    if isinstance(value, int):
        return value
    
    # Fast path: float that can be converted to int
    if isinstance(value, float):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return default
    
    # Try conversion
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default


def safe_get_str(
    data: Dict[str, Any],
    key: str,
    default: str = ""
) -> str:
    """
    Safely get string from dictionary (optimized).
    
    Args:
        data: Dictionary to extract from
        key: Key to look up
        default: Default string value
        
    Returns:
        String value or default
    """
    if not isinstance(data, dict):
        return default
    
    value = data.get(key)
    
    # Fast path: None or missing key
    if value is None:
        return default
    
    # Fast path: already a string
    if isinstance(value, str):
        return value
    
    # Convert to string (handles int, float, bool, etc.)
    try:
        return str(value)
    except Exception:
        return default

