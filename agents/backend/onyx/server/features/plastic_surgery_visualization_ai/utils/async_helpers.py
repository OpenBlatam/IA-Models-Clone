"""Async helper utilities."""

import asyncio
import time
from typing import List, Callable, Any, TypeVar, Optional
from functools import wraps

T = TypeVar('T')
R = TypeVar('R')


async def gather_with_limit(
    tasks: List[Callable],
    limit: int = 10
) -> List[Any]:
    """
    Gather tasks with concurrency limit.
    
    Args:
        tasks: List of async functions
        limit: Maximum concurrent tasks
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_task(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                return task()
    
    return await asyncio.gather(*[bounded_task(task) for task in tasks])


async def timeout_after(coro, timeout: float):
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def async_to_sync(func: Callable) -> Callable:
    """
    Convert async function to sync (for compatibility).
    
    Args:
        func: Async function
        
    Returns:
        Sync wrapper function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper


async def retry_async(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry async function.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum attempts
        delay: Delay between attempts
        exceptions: Exceptions to catch
        
    Returns:
        Function result
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
            else:
                raise
    
    if last_exception:
        raise last_exception


class AsyncQueue:
    """Simple async queue wrapper."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: Any) -> None:
        """Put item in queue."""
        await self.queue.put(item)
    
    async def get(self) -> Any:
        """Get item from queue."""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()


async def sleep_until(condition: Callable, timeout: float = 60.0, interval: float = 0.1) -> bool:
    """
    Sleep until condition is true.
    
    Args:
        condition: Function that returns bool
        timeout: Maximum time to wait
        interval: Check interval
        
    Returns:
        True if condition became true, False if timeout
    """
    start = time.time()
    
    while time.time() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    
    return False


async def parallel_limit(
    items: List[T],
    fn: Callable[[T], R],
    concurrency: int = 5
) -> List[R]:
    """
    Process items in parallel with concurrency limit.
    
    Args:
        items: List of items to process
        fn: Async function to apply
        concurrency: Maximum concurrent operations
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_item(item: T) -> R:
        async with semaphore:
            return await fn(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def sequential(items: List[T], fn: Callable[[T], R]) -> List[R]:
    """
    Process items sequentially.
    
    Args:
        items: List of items to process
        fn: Async function to apply
        
    Returns:
        List of results
    """
    results = []
    for item in items:
        result = await fn(item)
        results.append(result)
    return results


def timeout_decorator(seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Operation timed out after {seconds}s")
        
        return wrapper
    return decorator


async def retry_async_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Function result
    """
    attempt = 0
    delay = initial_delay
    
    while attempt < max_attempts:
        try:
            return await func()
        except exceptions as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

