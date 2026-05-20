"""
Common Validation Helpers

Helper functions for common validation patterns that appear throughout the codebase.
These functions encapsulate repetitive validation logic to improve maintainability.
"""

from typing import Any, Optional, List


def is_empty_string(value: Any) -> bool:
    """
    Check if a value is None, empty string, or whitespace-only string.
    
    This helper encapsulates the common pattern of checking if a string value
    is empty, which appears 100+ times across the codebase.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is None, empty string, or whitespace-only
        
    Example:
        >>> if is_empty_string(text):
        >>>     return ""
    """
    if value is None:
        return True
    
    if not isinstance(value, str):
        return False
    
    return not value.strip()


def ensure_not_empty_string(
    value: Any,
    param_name: str,
    allow_empty: bool = False
) -> str:
    """
    Ensure a value is a non-empty string, raising ValueError if not.
    
    This helper encapsulates the common pattern of validating and normalizing
    string parameters that appears repeatedly across validators and models.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        allow_empty: Whether to allow empty strings (default: False)
        
    Returns:
        Normalized string (stripped)
        
    Raises:
        ValueError: If value is None, not a string, or empty (if allow_empty=False)
        
    Example:
        >>> chat_id = ensure_not_empty_string(chat_id, "chat_id")
        >>> user_id = ensure_not_empty_string(user_id, "user_id")
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if not isinstance(value, str):
        raise ValueError(
            f"{param_name} must be a string, got {type(value).__name__}"
        )
    
    value = value.strip()
    
    if not value and not allow_empty:
        raise ValueError(f"{param_name} cannot be empty")
    
    return value


def normalize_string_or_none(value: Any) -> Optional[str]:
    """
    Normalize a string value, returning None if empty or None.
    
    This helper encapsulates the common pattern of normalizing optional string
    values that can be None or empty.
    
    Args:
        value: Value to normalize (can be None)
        
    Returns:
        Normalized string (stripped) or None if value is None or empty
        
    Example:
        >>> description = normalize_string_or_none(description)
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        return None
    
    value = value.strip()
    return value if value else None


def filter_none_values(items: List[Any]) -> List[Any]:
    """
    Filter out None values from a list.
    
    This helper encapsulates the common pattern of filtering None values
    that appears in converter functions and response builders.
    
    Args:
        items: List that may contain None values
        
    Returns:
        List without None values
        
    Example:
        >>> valid_items = filter_none_values(items)
    """
    if items is None:
        return []
    
    return [item for item in items if item is not None]


def validate_list_not_none(items: Any, param_name: str) -> List[Any]:
    """
    Validate that a list parameter is not None.
    
    This helper encapsulates the common pattern of validating list parameters
    that appears in converter and response functions.
    
    Args:
        items: Value to validate (should be a list)
        param_name: Name of the parameter (for error messages)
        
    Returns:
        Validated list
        
    Raises:
        ValueError: If items is None
        
    Example:
        >>> chats = validate_list_not_none(chats, "chats")
    """
    if items is None:
        raise ValueError(f"{param_name} cannot be None")
    
    return items if isinstance(items, list) else []


def validate_required_not_none(value: Any, param_name: str) -> Any:
    """
    Validate that a required parameter is not None.
    
    This helper encapsulates the common pattern of validating required parameters
    that appears in converter functions and model validators.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is None
        
    Example:
        >>> chat = validate_required_not_none(chat, "chat")
        >>> vote = validate_required_not_none(vote, "vote")
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    return value


def is_empty_list(value: Any) -> bool:
    """
    Check if a value is None or an empty list.
    
    This helper encapsulates the common pattern of checking if a list is empty,
    which appears 23+ times across the codebase.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is None or an empty list
        
    Example:
        >>> if is_empty_list(items):
        >>>     return []
    """
    if value is None:
        return True
    
    if not isinstance(value, (list, tuple)):
        return False
    
    return len(value) == 0


def is_not_empty_list(value: Any) -> bool:
    """
    Check if a value is a non-empty list.
    
    This helper encapsulates the common pattern of checking if a list has items,
    which appears repeatedly across the codebase.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a list/tuple with at least one item
        
    Example:
        >>> if is_not_empty_list(results):
        >>>     process_results(results)
    """
    if value is None:
        return False
    
    if not isinstance(value, (list, tuple)):
        return False
    
    return len(value) > 0

