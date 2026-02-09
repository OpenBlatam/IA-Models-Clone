"""
Validation Helpers

Utilities for validating values.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate value is in range.
    
    Args:
        value: Value to validate
        min_val: Minimum value
        max_val: Maximum value
        name: Value name for error message
        
    Returns:
        (is_valid, error_message)
    """
    if min_val is not None and value < min_val:
        return False, f"{name} ({value}) is below minimum ({min_val})"
    
    if max_val is not None and value > max_val:
        return False, f"{name} ({value}) is above maximum ({max_val})"
    
    return True, None


def validate_type(
    value: Any,
    expected_type: type,
    name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate value type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Value name for error message
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(value, expected_type):
        return False, f"{name} is {type(value).__name__}, expected {expected_type.__name__}"
    
    return True, None


def validate_not_none(
    value: Any,
    name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate value is not None.
    
    Args:
        value: Value to validate
        name: Value name for error message
        
    Returns:
        (is_valid, error_message)
    """
    if value is None:
        return False, f"{name} cannot be None"
    
    return True, None



