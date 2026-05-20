"""
Error Handlers
Utility functions for consistent error handling in routes
"""

import logging
from typing import Callable, Any, Optional
from functools import wraps

from fastapi import HTTPException, status, Request

logger = logging.getLogger(__name__)

def handle_route_errors(func: Callable) -> Callable:
    """Decorator to handle common route errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Optional[Request] = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            for value in kwargs.values():
                if isinstance(value, Request):
                    request = value
                    break
        
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(
                "Invalid request",
                extra={
                    "function": func.__name__,
                    "error": str(e),
                    "path": request.url.path if request else None,
                    "method": request.method if request else None
                }
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request: {str(e)}"
            )
        except AttributeError as e:
            logger.error(
                "Service not properly initialized",
                extra={
                    "function": func.__name__,
                    "error": str(e),
                    "path": request.url.path if request else None
                },
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in route",
                extra={
                    "function": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "path": request.url.path if request else None,
                    "method": request.method if request else None
                },
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )
    return wrapper

