"""
Error Handler

Enhanced error handling with context and recovery strategies.
"""

import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps

from ...domain.exceptions import QualityControlException

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Enhanced error handler with context and recovery.
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.error_callbacks = []
    
    def register_callback(self, callback: Callable[[Exception], None]):
        """
        Register error callback.
        
        Args:
            callback: Callback function that receives exception
        """
        self.error_callbacks.append(callback)
    
    def handle(
        self,
        error: Exception,
        context: Optional[dict] = None,
        recover: Optional[Callable] = None
    ) -> Any:
        """
        Handle error with optional recovery.
        
        Args:
            error: Exception to handle
            context: Optional context dictionary
            recover: Optional recovery function
        
        Returns:
            Result of recovery function or None
        """
        # Log error
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        if context:
            error_context.update(context)
        
        logger.error(f"Error handled: {error_context}")
        
        # Call registered callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
        
        # Try recovery if provided
        if recover:
            try:
                return recover(error)
            except Exception as e:
                logger.error(f"Recovery failed: {str(e)}")
        
        # Re-raise if it's a domain exception
        if isinstance(error, QualityControlException):
            raise
        
        return None
    
    def handle_with_default(
        self,
        error: Exception,
        default_value: Any,
        context: Optional[dict] = None
    ) -> Any:
        """
        Handle error and return default value.
        
        Args:
            error: Exception to handle
            default_value: Default value to return
            context: Optional context dictionary
        
        Returns:
            Default value
        """
        self.handle(error, context)
        return default_value


def handle_errors(
    default_value: Any = None,
    context: Optional[dict] = None,
    log_error: bool = True
):
    """
    Decorator for error handling.
    
    Args:
        default_value: Default value to return on error
        context: Optional context dictionary
        log_error: Whether to log errors
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    error_context = context or {}
                    error_context["function"] = func.__name__
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if default_value is not None:
                    return default_value
                raise
        
        return wrapper
    return decorator



