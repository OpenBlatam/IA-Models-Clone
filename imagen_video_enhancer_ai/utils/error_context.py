"""
Error Context Utilities
=======================

Utilities for adding context to errors.
"""

import traceback
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ErrorContext:
    """Context manager for error handling with additional context."""
    
    def __init__(self, context: Dict[str, Any]):
        """
        Initialize error context.
        
        Args:
            context: Context dictionary
        """
        self.context = context
        self.start_time = datetime.now()
    
    def add(self, key: str, value: Any):
        """Add context value."""
        self.context[key] = value
    
    def update(self, data: Dict[str, Any]):
        """Update context with dictionary."""
        self.context.update(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            **self.context,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }


@contextmanager
def error_context(**context):
    """
    Context manager for error handling with context.
    
    Usage:
        with error_context(user_id="123", task_id="456") as ctx:
            # code that might raise
            ctx.add("step", "processing")
    """
    ctx = ErrorContext(context)
    try:
        yield ctx
    except Exception as e:
        # Add error context
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "context": ctx.to_dict()
        }
        
        logger.error(f"Error occurred: {error_info}")
        raise


def wrap_with_context(context: Dict[str, Any]):
    """
    Decorator to wrap function with error context.
    
    Usage:
        @wrap_with_context({"service": "enhancement"})
        async def my_function():
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with error_context(**context) as ctx:
                ctx.add("function", func.__name__)
                ctx.add("args_count", len(args))
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with error_context(**context) as ctx:
                ctx.add("function", func.__name__)
                ctx.add("args_count", len(args))
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




