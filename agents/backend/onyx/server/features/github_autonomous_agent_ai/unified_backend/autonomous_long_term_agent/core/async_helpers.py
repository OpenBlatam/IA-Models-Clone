"""
Async Helpers
Utility functions for safe async operations
"""

import logging
from typing import TypeVar, Optional, Callable, Awaitable, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def safe_async_call(
    func: Callable[..., Awaitable[T]],
    *args,
    default: Optional[T] = None,
    error_message: Optional[str] = None,
    **kwargs
) -> Optional[T]:
    """
    Safely execute an async function, returning default on error.
    
    Args:
        func: Async function to call
        *args: Positional arguments for func
        default: Value to return on error (default: None)
        error_message: Custom error message (default: uses function name)
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func or default on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error in {func.__name__}"
        logger.warning(f"{msg}: {e}")
        return default


def safe_async_method(
    default_return: Any = None,
    error_message: Optional[str] = None
):
    """
    Decorator for safe async method execution.
    
    Args:
        default_return: Value to return on error
        error_message: Custom error message
    
    Usage:
        @safe_async_method(default_return={})
        async def get_stats(self):
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Optional[T]:
            return await safe_async_call(
                func,
                *args,
                default=default_return,
                error_message=error_message or f"Error in {func.__name__}",
                **kwargs
            )
        return wrapper
    return decorator

