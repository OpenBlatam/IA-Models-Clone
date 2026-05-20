"""
Error Handlers for API
======================

Centralized error handling utilities for API endpoints.
"""

import logging
from fastapi import HTTPException
from typing import Callable, Any

logger = logging.getLogger(__name__)


def handle_api_error(
    operation: str,
    error: Exception,
    status_code: int = 500
) -> HTTPException:
    """
    Create standardized HTTP exception for API errors.
    
    Args:
        operation: Description of the operation that failed
        error: Exception that occurred
        status_code: HTTP status code
        
    Returns:
        HTTPException with error details
    """
    logger.error(f"Error in {operation}: {error}", exc_info=True)
    return HTTPException(status_code=status_code, detail=str(error))


def api_error_handler(operation: str):
    """
    Decorator for handling errors in API endpoints.
    
    Args:
        operation: Description of the operation
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                raise handle_api_error(operation, e)
        return wrapper
    return decorator

