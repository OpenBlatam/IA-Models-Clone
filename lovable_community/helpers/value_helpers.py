"""
Value Helper Functions

Helper functions for common value access patterns with defaults.
"""

from typing import Optional, Any, TypeVar, Union

T = TypeVar('T')


def get_value_or_default(
    value: Optional[T],
    default: T
) -> T:
    """
    Get value if not None, otherwise return default.
    
    This helper encapsulates the common pattern of `value if value is not None else default`
    that appears repeatedly in optional parameter handling.
    
    Args:
        value: Optional value
        default: Default value to return if value is None
        
    Returns:
        Value if not None, otherwise default
        
    Example:
        >>> count = get_value_or_default(vote_count, 0)
        >>> name = get_value_or_default(optional_name, "Unknown")
    """
    return value if value is not None else default


def get_attr_or_default(
    value: Optional[T],
    attr_getter: callable,
    default: T
) -> T:
    """
    Get value if not None, otherwise get attribute from object or return default.
    
    This helper encapsulates the common pattern of:
    `value if value is not None else (obj.attr or default)`
    that appears repeatedly in optional parameter handling with object attributes.
    
    Args:
        value: Optional value
        attr_getter: Callable that returns the attribute value
        default: Default value to return if both value and attribute are None/empty
        
    Returns:
        Value if not None, otherwise attribute value or default
        
    Example:
        >>> vote_count = get_attr_or_default(
        >>>     vote_count,
        >>>     lambda: chat.vote_count,
        >>>     0
        >>> )
    """
    if value is not None:
        return value
    
    attr_value = attr_getter()
    return attr_value if attr_value is not None else default


def get_value_or_attr(
    value: Optional[T],
    attr_getter: callable
) -> Optional[T]:
    """
    Get value if not None, otherwise get attribute from object.
    
    This helper encapsulates the common pattern of:
    `value if value is not None else obj.attr`
    that appears repeatedly in optional parameter handling.
    
    Args:
        value: Optional value
        attr_getter: Callable that returns the attribute value
        
    Returns:
        Value if not None, otherwise attribute value (may be None)
        
    Example:
        >>> count = get_value_or_attr(vote_count, lambda: chat.vote_count)
    """
    return value if value is not None else attr_getter()


def coalesce(*values: Any) -> Any:
    """
    Return the first non-None value from the arguments.
    
    This helper encapsulates the common pattern of finding the first non-None value
    from multiple options.
    
    Args:
        *values: Variable number of values to check
        
    Returns:
        First non-None value, or None if all are None
        
    Example:
        >>> result = coalesce(optional_value, obj.attr, default_value)
        >>> name = coalesce(user_name, default_name, "Unknown")
    """
    for value in values:
        if value is not None:
            return value
    return None

