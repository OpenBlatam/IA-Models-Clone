"""Type checking and conversion utilities."""

from typing import Any, Optional, Union, List
import numbers


def is_numeric(value: Any) -> bool:
    """
    Check if value is numeric.
    
    Args:
        value: Value to check
        
    Returns:
        True if numeric
    """
    return isinstance(value, numbers.Number)


def is_integer(value: Any) -> bool:
    """
    Check if value is integer.
    
    Args:
        value: Value to check
        
    Returns:
        True if integer
    """
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def is_float(value: Any) -> bool:
    """
    Check if value is float.
    
    Args:
        value: Value to check
        
    Returns:
        True if float
    """
    return isinstance(value, float)


def is_string(value: Any) -> bool:
    """
    Check if value is string.
    
    Args:
        value: Value to check
        
    Returns:
        True if string
    """
    return isinstance(value, str)


def is_list(value: Any) -> bool:
    """
    Check if value is list.
    
    Args:
        value: Value to check
        
    Returns:
        True if list
    """
    return isinstance(value, list)


def is_dict(value: Any) -> bool:
    """
    Check if value is dictionary.
    
    Args:
        value: Value to check
        
    Returns:
        True if dictionary
    """
    return isinstance(value, dict)


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Convert value to integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_string(value: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Convert value to string.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        String or default
    """
    try:
        return str(value)
    except Exception:
        return default


def to_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    """
    Convert value to boolean.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Boolean or default
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def get_type_name(value: Any) -> str:
    """
    Get type name of value.
    
    Args:
        value: Value to check
        
    Returns:
        Type name string
    """
    return type(value).__name__


def is_instance_of(value: Any, *types: type) -> bool:
    """
    Check if value is instance of any of the types.
    
    Args:
        value: Value to check
        *types: Types to check against
        
    Returns:
        True if value is instance of any type
    """
    return isinstance(value, types)

