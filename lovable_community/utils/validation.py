"""
Enhanced validation utilities

Advanced validation functions with better error messages and type checking.
"""

import re
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime


def validate_not_none(value: Any, field_name: str = "field") -> Any:
    """
    Validate that value is not None.
    
    Args:
        value: Value to validate
        field_name: Field name for error message
        
    Returns:
        Value if not None
        
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")
    return value


def validate_not_empty(value: str, field_name: str = "field") -> str:
    """
    Validate that string is not empty.
    
    Args:
        value: String to validate
        field_name: Field name for error message
        
    Returns:
        Stripped string if valid
        
    Raises:
        ValueError: If value is empty
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty or whitespace only")
    
    return stripped


def validate_length(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    field_name: str = "field"
) -> str:
    """
    Validate string length.
    
    Args:
        value: String to validate
        min_length: Minimum length
        max_length: Maximum length
        field_name: Field name for error message
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If length constraints not met
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    length = len(value)
    
    if min_length is not None and length < min_length:
        raise ValueError(
            f"{field_name} must be at least {min_length} characters long "
            f"(got {length})"
        )
    
    if max_length is not None and length > max_length:
        raise ValueError(
            f"{field_name} must be at most {max_length} characters long "
            f"(got {length})"
        )
    
    return value


def validate_range(
    value: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    field_name: str = "field"
) -> int:
    """
    Validate numeric range.
    
    Args:
        value: Number to validate
        min_value: Minimum value
        max_value: Maximum value
        field_name: Field name for error message
        
    Returns:
        Validated number
        
    Raises:
        ValueError: If range constraints not met
    """
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    
    if min_value is not None and value < min_value:
        raise ValueError(
            f"{field_name} must be at least {min_value} (got {value})"
        )
    
    if max_value is not None and value > max_value:
        raise ValueError(
            f"{field_name} must be at most {max_value} (got {value})"
        )
    
    return value


def validate_one_of(
    value: Any,
    allowed_values: List[Any],
    field_name: str = "field"
) -> Any:
    """
    Validate that value is one of allowed values.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        field_name: Field name for error message
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value not in allowed values
    """
    if value not in allowed_values:
        raise ValueError(
            f"{field_name} must be one of {allowed_values} (got {value})"
        )
    return value


def validate_pattern(
    value: str,
    pattern: str,
    field_name: str = "field",
    error_message: Optional[str] = None
) -> str:
    """
    Validate string against regex pattern.
    
    Args:
        value: String to validate
        pattern: Regex pattern
        field_name: Field name for error message
        error_message: Custom error message
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If pattern doesn't match
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    if not re.match(pattern, value):
        message = error_message or f"{field_name} does not match required pattern"
        raise ValueError(message)
    
    return value


def validate_datetime_range(
    start: datetime,
    end: datetime,
    field_name: str = "datetime range"
) -> None:
    """
    Validate that start datetime is before end datetime.
    
    Args:
        start: Start datetime
        end: End datetime
        field_name: Field name for error message
        
    Raises:
        ValueError: If start is after end
    """
    if start >= end:
        raise ValueError(
            f"{field_name}: start datetime must be before end datetime"
        )


def validate_custom(
    value: Any,
    validator: Callable[[Any], bool],
    error_message: str,
    field_name: str = "field"
) -> Any:
    """
    Validate using custom validator function.
    
    Args:
        value: Value to validate
        validator: Validator function that returns True if valid
        error_message: Error message if validation fails
        field_name: Field name for error message
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If validation fails
    """
    if not validator(value):
        raise ValueError(f"{field_name}: {error_message}")
    return value













