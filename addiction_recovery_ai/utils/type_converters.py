"""
Type conversion utilities
Pure functions for converting between types
"""

from typing import Any, List, Optional
from datetime import datetime, date


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Convert value to integer
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer or default
    """
    if value is None:
        return default
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, float):
        return int(value)
    
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    return default


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float or default
    """
    if value is None:
        return default
    
    if isinstance(value, float):
        return value
    
    if isinstance(value, int):
        return float(value)
    
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return default


def to_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    """
    Convert value to boolean
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Boolean or default
    """
    if value is None:
        return default
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "1", "yes", "on"):
            return True
        if lower in ("false", "0", "no", "off"):
            return False
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return default


def to_string(value: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Convert value to string
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        String or default
    """
    if value is None:
        return default
    
    if isinstance(value, str):
        return value
    
    try:
        return str(value)
    except Exception:
        return default


def to_list(value: Any, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
    """
    Convert value to list
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        List or default
    """
    if value is None:
        return default or []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, tuple):
        return list(value)
    
    if isinstance(value, str):
        # Try to parse as comma-separated
        return [item.strip() for item in value.split(",") if item.strip()]
    
    return [value]


def to_datetime(value: Any, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Convert value to datetime
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Datetime or default
    """
    if value is None:
        return default
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    
    if isinstance(value, str):
        try:
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            return datetime.fromisoformat(value)
        except ValueError:
            return default
    
    return default

