"""
Retry handler with exponential backoff for API calls.
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps
import httpx

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from func
    
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except (httpx.HTTPError, httpx.TimeoutException, httpx.NetworkError) as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * exponential_base, max_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        except Exception as e:
            logger.error(f"Non-retryable error: {e}")
            raise
    
    if last_exception:
        raise last_exception


def retryable(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator for retryable async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                func,
                max_retries=max_retries,
                initial_delay=initial_delay,
                *args,
                **kwargs
            )
        return wrapper
    return decorator






