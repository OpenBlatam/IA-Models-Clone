"""
Async Utilities for Piel Mejorador AI SAM3
==========================================

Common async patterns and utilities.
"""

import asyncio
import logging
from typing import Callable, Any, List, Optional, TypeVar, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    timeout_message: Optional[str] = None
) -> T:
    """
    Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        timeout_message: Optional timeout message
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        msg = timeout_message or f"Operation timed out after {timeout}s"
        logger.error(msg)
        raise asyncio.TimeoutError(msg)


async def gather_with_errors(
    *coros: Awaitable[Any],
    return_exceptions: bool = True,
    log_errors: bool = True
) -> List[Any]:
    """
    Gather coroutines and handle errors gracefully.
    
    Args:
        *coros: Coroutines to gather
        return_exceptions: Whether to return exceptions
        log_errors: Whether to log errors
        
    Returns:
        List of results (or exceptions if return_exceptions=True)
    """
    results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
    
    if log_errors:
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in coroutine {i}: {result}", exc_info=result)
    
    return results


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple = (Exception,),
    operation_name: str = "operation"
) -> T:
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
        operation_name: Name of operation for logging
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                if exponential_backoff:
                    delay = retry_delay * (2 ** attempt)
                else:
                    delay = retry_delay
                
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts")
    
    raise last_exception


def async_to_sync(func: Callable) -> Callable:
    """
    Convert async function to sync (runs in event loop).
    
    Args:
        func: Async function
        
    Returns:
        Sync wrapper
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper


async def batch_process(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control.
    
    Args:
        items: Items to process
        processor: Async processor function
        batch_size: Size of each batch
        batch_size: Number of concurrent operations
        
    Returns:
        List of results
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await processor(item)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await gather_with_errors(
            *[process_with_semaphore(item) for item in batch]
        )
        results.extend(batch_results)
    
    return results




