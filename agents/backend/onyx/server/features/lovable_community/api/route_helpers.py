"""
Route Helper Functions

Helper functions to encapsulate common patterns in route handlers
to improve code maintainability and reduce duplication.
"""

import logging
from typing import Callable, Any, Optional, TypeVar
from functools import wraps
from fastapi import HTTPException

from ..exceptions import ChatNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_route_errors(
    operation_name: str,
    default_status_code: int = 500
) -> Callable:
    """
    Decorator to handle common route errors with consistent logging and HTTPException.
    
    This helper encapsulates the common pattern of:
    - Try/except blocks
    - HTTPException re-raising
    - Error logging
    - Consistent error responses
    
    Args:
        operation_name: Name of the operation (for logging)
        default_status_code: Default HTTP status code for errors (default: 500)
        
    Returns:
        Decorator function
        
    Example:
        @handle_route_errors("creating embedding")
        async def create_embedding(...):
            # route code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPExceptions (they're already properly formatted)
                raise
            except ChatNotFoundError as e:
                # Convert ChatNotFoundError to HTTPException
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error {operation_name}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=default_status_code,
                    detail=str(e)
                ) from e
        return wrapper
    return decorator


def get_chat_or_raise_404(
    chat_repository,
    chat_id: str,
    error_message: Optional[str] = None
):
    """
    Get chat by ID or raise HTTPException 404 if not found.
    
    This helper encapsulates the common pattern of:
    - Getting chat from repository
    - Checking if chat exists
    - Raising HTTPException 404 if not found
    
    Args:
        chat_repository: Chat repository instance
        chat_id: Chat ID to look up
        error_message: Optional custom error message
        
    Returns:
        Chat object
        
    Raises:
        HTTPException: 404 if chat not found
        
    Example:
        chat = get_chat_or_raise_404(chat_repository, chat_id)
    """
    chat = chat_repository.get_by_id(chat_id)
    if not chat:
        message = error_message or f"Chat {chat_id} not found"
        raise HTTPException(status_code=404, detail=message)
    return chat


def validate_required_string(
    value: Any,
    param_name: str,
    allow_empty: bool = False
) -> str:
    """
    Validate and normalize a required string parameter.
    
    This helper encapsulates the common pattern of validating required string
    parameters that appears in handler methods.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        allow_empty: Whether to allow empty strings (default: False)
        
    Returns:
        Normalized string (stripped)
        
    Raises:
        ValueError: If value is None, not a string, or empty (if allow_empty=False)
        
    Example:
        >>> chat_id = validate_required_string(chat_id, "chat_id")
        >>> user_id = validate_required_string(user_id, "user_id")
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


def validate_required_object(
    value: Any,
    param_name: str,
    object_type: Optional[type] = None
) -> Any:
    """
    Validate that a required object parameter is not None.
    
    This helper encapsulates the common pattern of validating that repository
    or service objects are not None.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        object_type: Optional type to check against
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is None or wrong type
        
    Example:
        >>> validate_required_object(view_repository, "view_repository")
        >>> validate_required_object(remix_repository, "remix_repository", RemixRepository)
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if object_type and not isinstance(value, object_type):
        raise ValueError(
            f"{param_name} must be an instance of {object_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    return value


def normalize_optional_string(value: Any) -> Optional[str]:
    """
    Normalize an optional string value (can be None).
    
    This helper normalizes optional string parameters by stripping whitespace
    and returning None if empty.
    
    Args:
        value: Value to normalize (can be None)
        
    Returns:
        Normalized string (stripped) or None if value is None or empty
        
    Example:
        >>> user_id = normalize_optional_string(user_id)
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        return None
    
    value = value.strip()
    return value if value else None

