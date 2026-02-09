"""
Error Handler for Color Grading AI
===================================

Unified error handling with context, logging, and recovery.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union
from functools import wraps
from datetime import datetime

from .exceptions import ColorGradingError

logger = logging.getLogger(__name__)


class ErrorContext:
    """Error context information."""
    
    def __init__(
        self,
        operation: str,
        service: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.service = service
        self.user_id = user_id
        self.request_id = request_id
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class ErrorHandler:
    """
    Unified error handler.
    
    Features:
    - Context-aware error handling
    - Automatic logging
    - Error recovery
    - Error aggregation
    - Custom handlers
    """
    
    def __init__(self):
        """Initialize error handler."""
        self._handlers: Dict[Type[Exception], Callable] = {}
        self._error_stats: Dict[str, Dict[str, Any]] = {}
        self._default_handler: Optional[Callable] = None
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable
    ):
        """
        Register custom error handler.
        
        Args:
            exception_type: Exception type to handle
            handler: Handler function
        """
        self._handlers[exception_type] = handler
        logger.info(f"Registered error handler for {exception_type.__name__}")
    
    def set_default_handler(self, handler: Callable):
        """Set default error handler."""
        self._default_handler = handler
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = False
    ) -> Optional[Any]:
        """
        Handle error with context.
        
        Args:
            error: Exception to handle
            context: Error context
            reraise: Whether to reraise after handling
            
        Returns:
            Handler result or None
        """
        error_type = type(error)
        
        # Record error statistics
        self._record_error(error, context)
        
        # Get handler
        handler = self._handlers.get(error_type)
        if not handler:
            handler = self._default_handler
        
        # Log error
        self._log_error(error, context)
        
        # Execute handler
        if handler:
            try:
                result = handler(error, context)
                if reraise:
                    raise
                return result
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        # Default: reraise if requested
        if reraise:
            raise
    
    def _log_error(self, error: Exception, context: Optional[ErrorContext]):
        """Log error with context."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        
        if context:
            error_info.update({
                "operation": context.operation,
                "service": context.service,
                "user_id": context.user_id,
                "request_id": context.request_id,
                "metadata": context.metadata,
            })
        
        logger.error(f"Error occurred: {error_info}")
    
    def _record_error(self, error: Exception, context: Optional[ErrorContext]):
        """Record error statistics."""
        error_type = type(error).__name__
        service = context.service if context else "unknown"
        
        key = f"{service}:{error_type}"
        
        if key not in self._error_stats:
            self._error_stats[key] = {
                "count": 0,
                "first_occurrence": datetime.now().isoformat(),
                "last_occurrence": datetime.now().isoformat(),
            }
        
        stats = self._error_stats[key]
        stats["count"] += 1
        stats["last_occurrence"] = datetime.now().isoformat()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self._error_stats.copy()
    
    def reset_stats(self):
        """Reset error statistics."""
        self._error_stats.clear()


def handle_errors(
    context: Optional[ErrorContext] = None,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
):
    """
    Decorator for unified error handling.
    
    Args:
        context: Error context
        default_return: Default return value on error
        log_error: Whether to log errors
        reraise: Whether to reraise after handling
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    operation=func.__name__,
                    service=getattr(args[0], '__class__', {}).__name__ if args else None
                )
                
                # Get error handler from service if available
                error_handler = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler = args[0].error_handler
                elif hasattr(func, '__self__') and hasattr(func.__self__, 'error_handler'):
                    error_handler = func.__self__.error_handler
                
                if error_handler:
                    return error_handler.handle_error(e, error_context, reraise=reraise)
                
                # Default handling
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    operation=func.__name__,
                    service=getattr(args[0], '__class__', {}).__name__ if args else None
                )
                
                error_handler = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler = args[0].error_handler
                
                if error_handler:
                    return error_handler.handle_error(e, error_context, reraise=reraise)
                
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                
                return default_return
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




