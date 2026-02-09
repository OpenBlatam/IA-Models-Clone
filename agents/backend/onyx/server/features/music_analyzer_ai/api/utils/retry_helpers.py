"""
Retry helper functions for resilient operations.

This module provides utilities for retrying operations with various
strategies (exponential backoff, fixed delay, etc.).
"""

from typing import Any, Callable, Optional, Awaitable, Type, Tuple
import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)


async def retry_async(
    operation: Callable[..., Awaitable[Any]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    *args,
    **kwargs
) -> Any:
    """
    Retry an async operation with exponential backoff.
    
    Args:
        operation: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry (attempt_num, exception)
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation
    
    Raises:
        Last exception if all attempts fail
    
    Example:
        result = await retry_async(
            spotify_service.get_track,
            track_id,
            max_attempts=3,
            delay=1.0,
            backoff=2.0
        )
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for {operation.__name__}: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"All {max_attempts} attempts failed for {operation.__name__}: {e}"
                )
    
    raise last_exception


def retry_sync(
    operation: Callable[..., Any],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    *args,
    **kwargs
) -> Any:
    """
    Retry a sync operation with exponential backoff.
    
    Args:
        operation: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation
    
    Raises:
        Last exception if all attempts fail
    """
    import time
    last_exception = None
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for {operation.__name__}: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"All {max_attempts} attempts failed for {operation.__name__}: {e}"
                )
    
    raise last_exception


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry
    
    Returns:
        Decorator function
    
    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def fetch_data():
            return await api_call()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_async(
                func,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions,
                on_retry=on_retry,
                *args,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return retry_sync(
                func,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions,
                on_retry=on_retry,
                *args,
                **kwargs
            )
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


async def retry_with_timeout(
    operation: Callable[..., Awaitable[Any]],
    timeout: float = 30.0,
    max_attempts: int = 3,
    delay: float = 1.0,
    *args,
    **kwargs
) -> Any:
    """
    Retry an async operation with timeout per attempt.
    
    Args:
        operation: Async function to retry
        timeout: Timeout per attempt (seconds)
        max_attempts: Maximum number of attempts
        delay: Delay between retries (seconds)
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation
    
    Raises:
        asyncio.TimeoutError: If operation times out
        Exception: Last exception if all attempts fail
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} timed out for {operation.__name__}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {max_attempts} attempts timed out for {operation.__name__}"
                )
                raise
        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for {operation.__name__}: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise
    
    return None








