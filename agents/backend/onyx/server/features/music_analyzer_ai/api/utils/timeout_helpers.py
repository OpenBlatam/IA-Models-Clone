"""
Timeout helper functions for operations with time limits.

This module provides utilities for executing operations with timeouts
and handling timeout scenarios gracefully.
"""

from typing import Any, Callable, Optional, Awaitable
import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)


async def with_timeout(
    operation: Callable[..., Awaitable[Any]],
    timeout: float,
    default: Any = None,
    timeout_message: Optional[str] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute an async operation with timeout.
    
    Args:
        operation: Async function to execute
        timeout: Timeout in seconds
        default: Default value to return on timeout
        timeout_message: Optional message to log on timeout
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation or default if timeout
    
    Example:
        result = await with_timeout(
            slow_operation,
            timeout=5.0,
            default=None,
            timeout_message="Operation timed out"
        )
    """
    try:
        return await asyncio.wait_for(
            operation(*args, **kwargs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        message = timeout_message or f"Operation {operation.__name__} timed out after {timeout}s"
        logger.warning(message)
        return default
    except Exception as e:
        logger.error(f"Operation {operation.__name__} failed: {e}")
        raise


def timeout(
    seconds: float,
    default: Any = None,
    timeout_message: Optional[str] = None
):
    """
    Decorator to add timeout to an async function.
    
    Args:
        seconds: Timeout in seconds
        default: Default value to return on timeout
        timeout_message: Optional message to log on timeout
    
    Returns:
        Decorator function
    
    Example:
        @timeout(seconds=5.0, default=None)
        async def slow_operation():
            await asyncio.sleep(10)
            return "done"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_timeout(
                func,
                seconds,
                default=default,
                timeout_message=timeout_message,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


async def timeout_or_raise(
    operation: Callable[..., Awaitable[Any]],
    timeout: float,
    timeout_error: Exception,
    *args,
    **kwargs
) -> Any:
    """
    Execute an async operation with timeout, raising exception on timeout.
    
    Args:
        operation: Async function to execute
        timeout: Timeout in seconds
        timeout_error: Exception to raise on timeout
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation
    
    Raises:
        timeout_error: If operation times out
    
    Example:
        result = await timeout_or_raise(
            api_call,
            timeout=10.0,
            timeout_error=HTTPException(status_code=504, detail="Request timeout")
        )
    """
    try:
        return await asyncio.wait_for(
            operation(*args, **kwargs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Operation {operation.__name__} timed out after {timeout}s")
        raise timeout_error


async def race_operations(
    operations: list[Callable[..., Awaitable[Any]]],
    timeout: Optional[float] = None,
    return_first: bool = True
) -> Any:
    """
    Race multiple operations and return the first to complete.
    
    Args:
        operations: List of async functions to race
        timeout: Optional timeout for all operations
        return_first: If True, return first result; if False, wait for all
    
    Returns:
        First result if return_first=True, list of results if False
    
    Example:
        result = await race_operations([
            lambda: api_call_1(),
            lambda: api_call_2(),
            lambda: api_call_3()
        ], timeout=10.0)
    """
    if not operations:
        return None if return_first else []
    
    tasks = [op() for op in operations]
    
    if return_first:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        if done:
            return await done.pop()
        return None
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results








