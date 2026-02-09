"""
Contextual Logger for Piel Mejorador AI SAM3
============================================

Enhanced logging with context.
"""

import logging
from typing import Dict, Any, Optional
from contextvars import ContextVar
from functools import wraps

logger = logging.getLogger(__name__)

# Context variables for request context
_request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})
_task_context: ContextVar[Dict[str, Any]] = ContextVar('task_context', default={})


class ContextualLogger:
    """
    Logger with automatic context injection.
    
    Features:
    - Request context
    - Task context
    - Automatic context in logs
    - Context managers
    """
    
    def __init__(self, name: str):
        """
        Initialize contextual logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current context."""
        context = {}
        context.update(_request_context.get({}))
        context.update(_task_context.get({}))
        return context
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        *args,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log with context."""
        context = self._get_context()
        
        if extra is None:
            extra = {}
        
        extra.update(context)
        
        self.logger.log(level, message, *args, extra=extra, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def set_request_context(self, **context):
        """Set request context."""
        _request_context.set(context)
    
    def set_task_context(self, **context):
        """Set task context."""
        _task_context.set(context)
    
    def clear_context(self):
        """Clear all context."""
        _request_context.set({})
        _task_context.set({})


def with_logging_context(**context):
    """
    Decorator to add logging context.
    
    Args:
        **context: Context to add
        
    Example:
        @with_logging_context(task_id="123", user_id="456")
        async def process_task():
            logger.info("Processing")  # Will include task_id and user_id
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            contextual_logger = ContextualLogger(func.__module__)
            contextual_logger.set_task_context(**context)
            try:
                return await func(*args, **kwargs)
            finally:
                contextual_logger.clear_context()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            contextual_logger = ContextualLogger(func.__module__)
            contextual_logger.set_task_context(**context)
            try:
                return func(*args, **kwargs)
            finally:
                contextual_logger.clear_context()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

