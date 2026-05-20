"""
Common helper functions

Utility functions for common operations like ID generation and timestamps.
"""

import uuid
from typing import List, Optional, Any, Callable, Tuple
from datetime import datetime
from .math_helpers import round_to_decimal_places
from .validation_common import is_empty_list, is_empty_string


def generate_id() -> str:
    """
    Generate a new UUID as string.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> datetime:
    """
    Get current UTC timestamp.
    
    Returns:
        Current UTC datetime
    """
    return datetime.utcnow()


def parse_date_range(date_from: Optional[str], date_to: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parses a date range from strings.
    
    Args:
        date_from: Start date (format: YYYY-MM-DD)
        date_to: End date (format: YYYY-MM-DD)
        
    Returns:
        Tuple of (datetime_from, datetime_to) or (None, None)
        
    Raises:
        ValueError: If date format is invalid (only if strict validation is needed)
        
    Note:
        Invalid dates are silently ignored and return None
    """
    dt_from = None
    dt_to = None
    
    if date_from:
        if is_empty_string(date_from):
            return None, None
        
        try:
            dt_from = datetime.strptime(date_from.strip(), "%Y-%m-%d")
        except (ValueError, TypeError):
            # Silently ignore invalid date format
            pass
    
    if date_to:
        if is_empty_string(date_to):
            return dt_from, None
        
        try:
            dt_to = datetime.strptime(date_to.strip(), "%Y-%m-%d")
            dt_to = dt_to.replace(hour=23, minute=59, second=59)
        except (ValueError, TypeError):
            # Silently ignore invalid date format
            pass
    
    # Validate date range if both dates are provided
    if dt_from and dt_to and dt_from > dt_to:
        # Return None for invalid range
        return None, None
    
    return dt_from, dt_to


def remove_duplicates(items: List[Any], key: Optional[Callable[[Any], Any]] = None) -> List[Any]:
    """
    Removes duplicates from a list while maintaining order.
    
    Args:
        items: List with possible duplicates
        key: Function to extract comparison key (optional)
        
    Returns:
        List without duplicates
        
    Raises:
        ValueError: If items is None
        TypeError: If key is provided but not callable
    """
    if items is None:
        raise ValueError("items cannot be None")
    
    if is_empty_list(items):
        return []
    
    if key is not None and not callable(key):
        raise TypeError("key must be callable if provided")
    
    if key:
        seen = set()
        result = []
        for item in items:
            try:
                item_key = key(item)
                # Handle unhashable keys
                if isinstance(item_key, (list, dict)):
                    item_key = str(item_key)
                if item_key not in seen:
                    seen.add(item_key)
                    result.append(item)
            except Exception:
                # If key extraction fails, skip item
                continue
        return result
    else:
        seen = set()
        result = []
        for item in items:
            # Handle unhashable items
            try:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            except TypeError:
                # For unhashable types, use string representation
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    result.append(item)
        return result


def safe_convert(value: Any, converter: callable, default: Any) -> Any:
    """
    Safely convert a value using a converter function.
    
    This helper encapsulates the common pattern of try/except for conversions
    that appears in safe_int and safe_float.
    
    Args:
        value: Value to convert
        converter: Converter function (e.g., int, float, str)
        default: Default value if conversion fails
        
    Returns:
        Converted value or default
        
    Example:
        >>> result = safe_convert(value, int, 0)
        >>> result = safe_convert(value, float, 0.0)
    """
    try:
        return converter(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely converts a value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer or default value
    """
    return safe_convert(value, int, default)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float or default value
    """
    return safe_convert(value, float, default)


def format_bytes(bytes_size: int) -> str:
    """
    Formats bytes size to human-readable format.
    
    Args:
        bytes_size: Size in bytes (must be >= 0)
        
    Returns:
        Formatted string (e.g., "1.5 MB")
        
    Raises:
        ValueError: If bytes_size is negative
    """
    if bytes_size < 0:
        raise ValueError(f"bytes_size cannot be negative, got {bytes_size}")
    
    if bytes_size == 0:
        return "0 B"
    
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_percentage(value: int, total: int, decimals: int = 2) -> float:
    """
    Calculates percentage of a value over total.
    
    Args:
        value: Value
        total: Total
        decimals: Number of decimal places (must be >= 0)
        
    Returns:
        Rounded percentage (0.0 if total is 0)
        
    Raises:
        ValueError: If value or total is negative, or decimals is negative
    """
    if value < 0:
        raise ValueError(f"value cannot be negative, got {value}")
    
    if total < 0:
        raise ValueError(f"total cannot be negative, got {total}")
    
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")
    
    if total == 0:
        return 0.0
    
    percentage = (value / total) * 100
    return round_to_decimal_places(percentage, decimals)






