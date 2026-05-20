"""
Validation helper functions for use cases.

Provides reusable validation functions to eliminate repetitive
validation code across use cases.
"""

from typing import Type
from ...exceptions import UseCaseException, RecommendationException


def validate_string_not_empty(
    value: str,
    field_name: str = "Field",
    exception_class: Type[Exception] = UseCaseException
) -> str:
    """
    Validate that a string is not empty or whitespace.
    
    Args:
        value: String to validate
        field_name: Name of field for error message
        exception_class: Exception class to raise
    
    Returns:
        Stripped string value
    
    Raises:
        exception_class: If value is empty or whitespace
    
    Example:
        >>> query = validate_string_not_empty(query, "Search query")
        >>> # Raises UseCaseException if query is empty
    """
    if not value or not value.strip():
        raise exception_class(f"{field_name} cannot be empty")
    return value.strip()


def validate_numeric_range(
    value: int,
    min_val: int,
    max_val: int,
    field_name: str = "Value",
    exception_class: Type[Exception] = UseCaseException
) -> int:
    """
    Validate that a number is within a specified range.
    
    Args:
        value: Number to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error message
        exception_class: Exception class to raise
    
    Returns:
        Validated number
    
    Raises:
        exception_class: If value is out of range
    
    Example:
        >>> limit = validate_numeric_range(limit, 1, 50, "Limit")
        >>> length = validate_numeric_range(
        ...     length, 1, 100, "Playlist length", RecommendationException
        ... )
    """
    if value < min_val or value > max_val:
        raise exception_class(
            f"{field_name} must be between {min_val} and {max_val}"
        )
    return value








