"""
Validation Utilities

Utility functions for validation across the system.
"""

from typing import Any, Callable, Optional
import re


def validate_email(email: str) -> bool:
    """
    Validate email address.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL.
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid
    """
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
    return bool(re.match(pattern, url))


def validate_positive_number(value: Any, allow_zero: bool = False) -> bool:
    """
    Validate positive number.
    
    Args:
        value: Value to validate
        allow_zero: Whether to allow zero
    
    Returns:
        True if valid
    """
    try:
        num = float(value)
        if allow_zero:
            return num >= 0
        return num > 0
    except (ValueError, TypeError):
        return False


def validate_range(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> bool:
    """
    Validate value is within range.
    
    Args:
        value: Value to validate
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
    
    Returns:
        True if valid
    """
    try:
        num = float(value)
        if min_value is not None and num < min_value:
            return False
        if max_value is not None and num > max_value:
            return False
        return True
    except (ValueError, TypeError):
        return False


def validate_not_empty(value: Any) -> bool:
    """
    Validate value is not empty.
    
    Args:
        value: Value to validate
    
    Returns:
        True if not empty
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def validate_required_fields(
    data: dict,
    required_fields: list[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate required fields are present.
    
    Args:
        data: Data dictionary
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, missing_field)
    """
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        if not validate_not_empty(data[field]):
            return False, f"Required field is empty: {field}"
    return True, None



