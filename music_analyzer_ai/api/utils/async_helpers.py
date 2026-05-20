"""
Async operation helpers
"""

import asyncio
from typing import List, Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator for async function retry logic
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


async def timeout_after(seconds: float):
    """
    Create a timeout context
    
    Args:
        seconds: Timeout in seconds
    
    Returns:
        Timeout context manager
    """
    return asyncio.wait_for(None, timeout=seconds)


async def run_with_timeout(
    coro: Callable,
    timeout: float,
    *args,
    **kwargs
) -> Any:
    """
    Run coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        *args: Coroutine arguments
        **kwargs: Coroutine keyword arguments
    
    Returns:
        Coroutine result
    
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout}s")
        raise


async def gather_with_errors(
    *coros,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Gather coroutines and handle errors gracefully
    
    Args:
        *coros: Coroutines to gather
        return_exceptions: Whether to return exceptions
    
    Returns:
        List of results
    """
    results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
    return results

