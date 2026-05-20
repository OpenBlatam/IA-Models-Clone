"""
Async Helpers
=============

Helper utilities for async operations.
"""

import asyncio
import logging
from typing import List, Callable, Any, Awaitable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def gather_with_limit(
    tasks: List[Awaitable[T]],
    limit: int,
    return_exceptions: bool = False
) -> List[T]:
    """
    Gather tasks with concurrency limit.
    
    Args:
        tasks: List of awaitable tasks
        limit: Maximum concurrent tasks
        return_exceptions: Whether to return exceptions as results
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[bounded_task(task) for task in tasks],
        return_exceptions=return_exceptions
    )


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {current_delay:.2f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries} retries failed: {e}")
    
    raise last_exception


async def timeout_async(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Default value to return on timeout
        
    Returns:
        Coroutine result or default value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


async def batch_process_async(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await gather_with_limit(
            [processor(item) for item in batch],
            limit=max_concurrent
        )
        results.extend(batch_results)
    
    return results


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for async function retry.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Usage:
        @async_retry(max_retries=3)
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions
            )
        return wrapper
    return decorator

