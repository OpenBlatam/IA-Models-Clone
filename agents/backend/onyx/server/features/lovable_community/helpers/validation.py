"""
Validation helper functions

Functions for validating data formats.
"""

import re


def validate_uuid_format(uuid_str: str) -> bool:
    """
    Validates if a string has UUID format.
    
    Args:
        uuid_str: String to validate
        
    Returns:
        True if has valid UUID format, False otherwise
        
    Raises:
        TypeError: If uuid_str is not a string
    """
    if uuid_str is None:
        return False
    
    if not isinstance(uuid_str, str):
        raise TypeError(f"uuid_str must be a string, got {type(uuid_str).__name__}")
    
    if not uuid_str.strip():
        return False
    
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_str.strip()))











