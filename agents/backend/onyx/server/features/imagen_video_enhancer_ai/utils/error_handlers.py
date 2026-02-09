"""
Error handlers for Imagen Video Enhancer AI
===========================================

Centralized error handling utilities.
"""

import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


class EnhancementError(Exception):
    """Base exception for enhancement errors."""
    pass


class FileValidationError(EnhancementError):
    """Error in file validation."""
    pass


class ProcessingError(EnhancementError):
    """Error during processing."""
    pass


class APIError(EnhancementError):
    """Error from API calls."""
    pass


def handle_enhancement_error(func):
    """
    Decorator to handle enhancement errors.
    
    Usage:
        @handle_enhancement_error
        async def my_function():
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except EnhancementError:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise ProcessingError(f"Processing error: {e}") from e
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EnhancementError:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise ProcessingError(f"Processing error: {e}") from e
    
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    return sync_wrapper


def format_error_response(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format error response.
    
    Args:
        error: Exception that occurred
        context: Optional context information
        
    Returns:
        Formatted error dictionary
    """
    response = {
        "error": str(error),
        "error_type": type(error).__name__,
    }
    
    if context:
        response["context"] = context
    
    if isinstance(error, EnhancementError):
        response["enhancement_error"] = True
    
    return response




