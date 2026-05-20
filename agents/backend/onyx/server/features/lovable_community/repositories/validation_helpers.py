"""
Repository Validation and Error Handling Helpers

Helper functions to encapsulate common validation and error handling patterns
used across repositories to improve code maintainability and reduce duplication.
"""

from typing import Optional, Callable, Any
from sqlalchemy.orm import Session
from functools import wraps

from ..exceptions import DatabaseError
from ..utils.logging_config import StructuredLogger


def validate_string_id(
    value: Any,
    param_name: str,
    allow_empty: bool = False
) -> str:
    """
    Validate and normalize a string ID parameter.
    
    This helper encapsulates the common pattern of validating string IDs
    that appears repeatedly across repository methods.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        allow_empty: Whether to allow empty strings (default: False)
        
    Returns:
        Normalized string (stripped)
        
    Raises:
        ValueError: If value is None, not a string, or empty (if allow_empty=False)
        
    Example:
        >>> chat_id = validate_string_id(chat_id, "chat_id")
        >>> user_id = validate_string_id(user_id, "user_id")
    """
    if value is None:
        raise ValueError(f"{param_name} must be a non-empty string, got None")
    
    if not isinstance(value, str):
        raise ValueError(
            f"{param_name} must be a non-empty string, got {type(value).__name__}"
        )
    
    value = value.strip()
    
    if not value and not allow_empty:
        raise ValueError(f"{param_name} must be a non-empty string, got empty string")
    
    return value


def validate_positive_integer(
    value: Any,
    param_name: str,
    min_value: int = 1
) -> int:
    """
    Validate and normalize a positive integer parameter.
    
    This helper encapsulates the common pattern of validating positive integers
    (like limits, hours, etc.) that appears repeatedly across repository methods.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (default: 1)
        
    Returns:
        Validated integer
        
    Raises:
        ValueError: If value is not an integer or is less than min_value
        
    Example:
        >>> limit = validate_positive_integer(limit, "limit")
        >>> hours = validate_positive_integer(hours, "hours")
    """
    if not isinstance(value, int):
        raise ValueError(
            f"{param_name} must be an integer, got {type(value).__name__}"
        )
    
    if value < min_value:
        raise ValueError(
            f"{param_name} must be >= {min_value}, got {value}"
        )
    
    return value


def handle_database_operation(
    operation_name: str,
    entity_type: str,
    entity_id: Optional[str] = None
):
    """
    Decorator for handling database operations with consistent error handling.
    
    This decorator encapsulates the common pattern of:
    - Try/except blocks
    - Rollback on error
    - Structured logging
    - Raising DatabaseError
    
    Args:
        operation_name: Name of the operation (e.g., "delete", "update")
        entity_type: Type of entity (e.g., "chat", "vote", "remix")
        entity_id: Optional entity ID for logging
        
    Returns:
        Decorator function
        
    Example:
        @handle_database_operation("delete", "vote", chat_id)
        def delete_votes():
            # operation code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(db: Session, *args, **kwargs) -> Any:
            logger = StructuredLogger(__name__)
            try:
                result = func(db, *args, **kwargs)
                db.commit()
                return result
            except Exception as e:
                db.rollback()
                entity_context = f" for {entity_type}"
                if entity_id:
                    entity_context += f" {entity_id}"
                logger.error(
                    f"Error {operation_name}ing {entity_type}{entity_context}: {e}",
                    exc_info=True
                )
                raise DatabaseError(
                    f"Failed to {operation_name} {entity_type}: {str(e)}"
                ) from e
        return wrapper
    return decorator


def execute_with_error_handling(
    db: Session,
    operation: Callable[[], Any],
    operation_name: str,
    entity_type: str,
    entity_id: Optional[str] = None
) -> Any:
    """
    Execute a database operation with consistent error handling.
    
    This helper function encapsulates the common pattern of:
    - Try/except blocks
    - Rollback on error
    - Structured logging
    - Raising DatabaseError
    
    Args:
        db: Database session
        operation: Callable that performs the database operation
        operation_name: Name of the operation (e.g., "delete", "update")
        entity_type: Type of entity (e.g., "chat", "vote", "remix")
        entity_id: Optional entity ID for logging
        
    Returns:
        Result of the operation
        
    Raises:
        DatabaseError: If the operation fails
        
    Example:
        >>> deleted_count = execute_with_error_handling(
        ...     db,
        ...     lambda: db.query(ChatVote).filter(...).delete(),
        ...     "delete",
        ...     "vote",
        ...     chat_id
        ... )
    """
    logger = StructuredLogger(__name__)
    try:
        result = operation()
        db.commit()
        return result
    except Exception as e:
        db.rollback()
        entity_context = f" for {entity_type}"
        if entity_id:
            entity_context += f" {entity_id}"
        logger.error(
            f"Error {operation_name}ing {entity_type}{entity_context}: {e}",
            exc_info=True
        )
        raise DatabaseError(
            f"Failed to {operation_name} {entity_type}: {str(e)}"
        ) from e


def validate_list_of_string_ids(
    values: Any,
    param_name: str
) -> list[str]:
    """
    Validate and normalize a list of string IDs.
    
    This helper validates that a parameter is a list of non-empty strings,
    which is commonly used for batch operations.
    
    Args:
        values: Value to validate (should be a list)
        param_name: Name of the parameter (for error messages)
        
    Returns:
        List of normalized strings (stripped)
        
    Raises:
        ValueError: If values is None, not a list, or contains invalid entries
        
    Example:
        >>> chat_ids = validate_list_of_string_ids(chat_ids, "chat_ids")
    """
    if values is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if not isinstance(values, list):
        raise ValueError(
            f"{param_name} must be a list, got {type(values).__name__}"
        )
    
    if not values:
        return []
    
    validated_values = []
    for i, value in enumerate(values):
        if not value or not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{param_name}[{i}] must be a non-empty string, "
                f"got {type(value).__name__}"
            )
        validated_values.append(value.strip())
    
    return validated_values


def validate_optional_string_id(
    value: Any,
    param_name: str
) -> Optional[str]:
    """
    Validate and normalize an optional string ID parameter.
    
    This helper validates string IDs that are optional (can be None).
    
    Args:
        value: Value to validate (can be None)
        param_name: Name of the parameter (for error messages)
        
    Returns:
        Normalized string (stripped) or None
        
    Raises:
        ValueError: If value is not None and not a valid string
        
    Example:
        >>> user_id = validate_optional_string_id(user_id, "user_id")
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        raise ValueError(
            f"{param_name} must be a string or None, got {type(value).__name__}"
        )
    
    value = value.strip()
    
    if not value:
        return None
    
    return value

