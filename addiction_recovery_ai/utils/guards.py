"""
Guard clause utilities
Reusable guard functions for validation
"""

from typing import Any, Optional, List


def guard_not_none(value: Any, name: str) -> None:
    """
    Guard clause: value must not be None
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        ValueError if value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def guard_not_empty(value: Any, name: str) -> None:
    """
    Guard clause: value must not be empty
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        ValueError if value is empty
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")


def guard_in_range(
    value: float,
    min_value: float,
    max_value: float,
    name: str
) -> None:
    """
    Guard clause: value must be in range
    
    Args:
        value: Value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value for error message
    
    Raises:
        ValueError if value is out of range
    """
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name} must be between {min_value} and {max_value}")


def guard_in_list(
    value: Any,
    allowed_values: List[Any],
    name: str
) -> None:
    """
    Guard clause: value must be in allowed list
    
    Args:
        value: Value to check
        allowed_values: List of allowed values
        name: Name of the value for error message
    
    Raises:
        ValueError if value is not in allowed list
    """
    if value not in allowed_values:
        raise ValueError(f"{name} must be one of: {allowed_values}")


def guard_positive(value: float, name: str) -> None:
    """
    Guard clause: value must be positive
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        ValueError if value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def guard_non_negative(value: float, name: str) -> None:
    """
    Guard clause: value must be non-negative
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        ValueError if value is negative
    """
    if value < 0:
        raise ValueError(f"{name} cannot be negative")

