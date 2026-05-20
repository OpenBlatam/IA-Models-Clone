"""
Service Error Handling Helpers

Helper functions to encapsulate common error handling patterns in services
to improve code maintainability and reduce duplication.
"""

import logging
from typing import Callable, TypeVar, Tuple, Any
from functools import wraps

from ..exceptions import (
    DatabaseError,
    ChatNotFoundError,
    InvalidChatError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_service_errors(
    operation_name: str,
    allowed_exceptions: Tuple[type, ...] = (ChatNotFoundError, InvalidChatError)
) -> Callable:
    """
    Decorator to handle common service errors with consistent logging and DatabaseError.
    
    This helper encapsulates the common pattern of:
    - Try/except blocks
    - Re-raising allowed exceptions
    - Logging errors
    - Raising DatabaseError for unexpected errors
    
    Args:
        operation_name: Name of the operation (for logging)
        allowed_exceptions: Tuple of exception types to re-raise (default: ChatNotFoundError, InvalidChatError)
        
    Returns:
        Decorator function
        
    Example:
        @handle_service_errors("publishing chat")
        def publish_chat(...):
            # service code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except allowed_exceptions:
                # Re-raise allowed exceptions (they're already properly formatted)
                raise
            except Exception as e:
                logger.error(f"Error {operation_name}: {e}", exc_info=True)
                raise DatabaseError(f"Failed to {operation_name}: {str(e)}") from e
        return wrapper
    return decorator


def convert_validation_error(
    value: Any,
    param_name: str,
    error_message: str
) -> str:
    """
    Validate a value and convert ValueError to InvalidChatError.
    
    This helper encapsulates the common pattern in validators of:
    - Calling ensure_not_empty_string
    - Catching ValueError
    - Converting to InvalidChatError with custom message
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for validation)
        error_message: Custom error message for InvalidChatError
        
    Returns:
        Validated and normalized string
        
    Raises:
        InvalidChatError: If validation fails
        
    Example:
        >>> chat_id = convert_validation_error(chat_id, "chat_id", "Chat ID cannot be empty")
    """
    from ..helpers.validation_common import ensure_not_empty_string
    
    try:
        return ensure_not_empty_string(value, param_name)
    except ValueError as e:
        raise InvalidChatError(error_message) from e


def log_and_raise_database_error(
    operation_name: str,
    error: Exception,
    context: dict = None
) -> None:
    """
    Log an error and raise DatabaseError with consistent formatting.
    
    This helper encapsulates the common pattern of:
    - Logging errors with context
    - Raising DatabaseError with descriptive message
    
    Args:
        operation_name: Name of the operation (for error message)
        error: Exception that occurred
        context: Optional context dictionary for logging
        
    Raises:
        DatabaseError: Always raises DatabaseError
        
    Example:
        >>> try:
        >>>     # operation
        >>> except Exception as e:
        >>>     log_and_raise_database_error("publishing chat", e, {"chat_id": chat_id})
    """
    log_context = context or {}
    logger.error(
        f"Error {operation_name}: {error}",
        exc_info=True,
        **log_context
    )
    raise DatabaseError(f"Failed to {operation_name}: {str(error)}") from error


def safe_execute_with_error_handling(
    operation: Callable[[], T],
    operation_name: str,
    allowed_exceptions: Tuple[type, ...] = (ChatNotFoundError, InvalidChatError),
    context: dict = None
) -> T:
    """
    Execute an operation with consistent error handling.
    
    This helper encapsulates the common pattern of:
    - Try/except blocks
    - Re-raising allowed exceptions
    - Logging and raising DatabaseError for unexpected errors
    
    Args:
        operation: Callable that performs the operation
        operation_name: Name of the operation (for logging)
        allowed_exceptions: Tuple of exception types to re-raise
        context: Optional context dictionary for logging
        
    Returns:
        Result of the operation
        
    Raises:
        DatabaseError: If operation fails with unexpected error
        Allowed exceptions are re-raised as-is
        
    Example:
        >>> result = safe_execute_with_error_handling(
        >>>     lambda: chat_repository.create(**data),
        >>>     "creating chat",
        >>>     context={"user_id": user_id}
        >>> )
    """
    try:
        return operation()
    except allowed_exceptions:
        raise
    except Exception as e:
        log_and_raise_database_error(operation_name, e, context)

