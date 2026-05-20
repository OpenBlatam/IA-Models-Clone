"""
Database helper functions

Utility functions for common database operations and error handling.
"""

import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
from sqlalchemy.orm import Session

from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_db_error(
    operation_name: str,
    reraise_exceptions: tuple = ()
) -> Callable:
    """
    Decorator for handling database errors with rollback.
    
    Args:
        operation_name: Name of the operation for logging
        reraise_exceptions: Tuple of exceptions to re-raise without wrapping
        
    Returns:
        Decorator function
        
    Usage:
        @handle_db_error("create embedding")
        def create_embedding(self, ...):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            db: Optional[Session] = getattr(self, 'db', None)
            
            try:
                return func(self, *args, **kwargs)
            except reraise_exceptions:
                raise
            except Exception as e:
                if db:
                    try:
                        db.rollback()
                    except Exception as rollback_error:
                        logger.error(f"Error during rollback: {rollback_error}", exc_info=True)
                
                logger.error(
                    f"Error {operation_name}: {e}",
                    exc_info=True
                )
                raise DatabaseError(f"Failed to {operation_name}: {str(e)}")
        
        return wrapper
    return decorator


def safe_db_operation(
    db: Session,
    operation: Callable[[], T],
    operation_name: str,
    reraise_exceptions: tuple = ()
) -> T:
    """
    Execute a database operation with automatic rollback on error.
    
    Args:
        db: Database session
        operation: Function to execute
        operation_name: Name of the operation for logging
        reraise_exceptions: Tuple of exceptions to re-raise without wrapping
        
    Returns:
        Result of the operation
        
    Raises:
        DatabaseError: If operation fails
    """
    try:
        return operation()
    except reraise_exceptions:
        raise
    except Exception as e:
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Error during rollback: {rollback_error}", exc_info=True)
        
        logger.error(
            f"Error {operation_name}: {e}",
            exc_info=True
        )
        raise DatabaseError(f"Failed to {operation_name}: {str(e)}")






