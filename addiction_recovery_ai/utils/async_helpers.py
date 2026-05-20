"""
Async helper utilities for I/O operations
"""

from typing import List, TypeVar, Callable, Awaitable, Any
import asyncio
from functools import wraps

T = TypeVar('T')


async def run_in_batches(
    items: List[T],
    batch_size: int,
    func: Callable[[List[T]], Awaitable[Any]]
) -> List[Any]:
    """
    Process items in batches asynchronously
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        func: Async function to process each batch
    
    Returns:
        List of results from all batches
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = await func(batch)
        results.append(batch_result)
    
    return results


async def run_parallel(
    tasks: List[Awaitable[T]],
    max_concurrent: int = 10
) -> List[T]:
    """
    Run tasks in parallel with concurrency limit
    
    Args:
        tasks: List of awaitable tasks
        max_concurrent: Maximum concurrent tasks
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """
    Decorator to retry async functions on failure
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio
            import logging
            
            logger = logging.getLogger(__name__)
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise RuntimeError(f"Function {func.__name__} failed after {max_attempts} attempts")
        
        return wrapper
    
    return decorator


async def timeout_after(
    coro: Awaitable[T],
    timeout: float
) -> T:
    """
    Run coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
    
    Returns:
        Result of coroutine
    
    Raises:
        asyncio.TimeoutError if timeout is exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)

