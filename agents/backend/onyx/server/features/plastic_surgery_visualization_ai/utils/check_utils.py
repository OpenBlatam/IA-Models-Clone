"""Check and validation utilities."""

from typing import Any, Optional, List


def is_empty(value: Any) -> bool:
    """
    Check if value is empty.
    
    Args:
        value: Value to check
        
    Returns:
        True if empty
    """
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    return False


def is_not_empty(value: Any) -> bool:
    """
    Check if value is not empty.
    
    Args:
        value: Value to check
        
    Returns:
        True if not empty
    """
    return not is_empty(value)


def is_none(value: Any) -> bool:
    """
    Check if value is None.
    
    Args:
        value: Value to check
        
    Returns:
        True if None
    """
    return value is None


def is_not_none(value: Any) -> bool:
    """
    Check if value is not None.
    
    Args:
        value: Value to check
        
    Returns:
        True if not None
    """
    return value is not None


def is_truthy(value: Any) -> bool:
    """
    Check if value is truthy.
    
    Args:
        value: Value to check
        
    Returns:
        True if truthy
    """
    return bool(value)


def is_falsy(value: Any) -> bool:
    """
    Check if value is falsy.
    
    Args:
        value: Value to check
        
    Returns:
        True if falsy
    """
    return not bool(value)


def is_even(value: int) -> bool:
    """
    Check if number is even.
    
    Args:
        value: Number to check
        
    Returns:
        True if even
    """
    return value % 2 == 0


def is_odd(value: int) -> bool:
    """
    Check if number is odd.
    
    Args:
        value: Number to check
        
    Returns:
        True if odd
    """
    return value % 2 != 0


def is_positive(value: float) -> bool:
    """
    Check if number is positive.
    
    Args:
        value: Number to check
        
    Returns:
        True if positive
    """
    return value > 0


def is_negative(value: float) -> bool:
    """
    Check if number is negative.
    
    Args:
        value: Number to check
        
    Returns:
        True if negative
    """
    return value < 0


def is_zero(value: float, tolerance: float = 0.0) -> bool:
    """
    Check if number is zero (with tolerance).
    
    Args:
        value: Number to check
        tolerance: Tolerance for comparison
        
    Returns:
        True if zero (within tolerance)
    """
    return abs(value) <= tolerance


def is_in_range(value: float, min_val: float, max_val: float, inclusive: bool = True) -> bool:
    """
    Check if value is in range.
    
    Args:
        value: Value to check
        min_val: Minimum value
        max_val: Maximum value
        inclusive: Include boundaries
        
    Returns:
        True if in range
    """
    if inclusive:
        return min_val <= value <= max_val
    else:
        return min_val < value < max_val


def is_one_of(value: Any, *options: Any) -> bool:
    """
    Check if value is one of the options.
    
    Args:
        value: Value to check
        *options: Options to check against
        
    Returns:
        True if value is in options
    """
    return value in options


def is_not_one_of(value: Any, *options: Any) -> bool:
    """
    Check if value is not one of the options.
    
    Args:
        value: Value to check
        *options: Options to check against
        
    Returns:
        True if value is not in options
    """
    return value not in options

