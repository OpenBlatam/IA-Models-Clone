"""
Async Helpers - Utilities for async/await operations.

Provides:
- Async retry mechanisms
- Async rate limiting
- Async batching
- Async timeouts
- Async progress tracking
"""

import asyncio
import time
import logging
from typing import Callable, Any, List, Optional, TypeVar, Awaitable, Coroutine
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC RETRY
# ════════════════════════════════════════════════════════════════════════════════

async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs,
) -> T:
    """
    Retry an async function on failure.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from function
    
    Example:
        >>> async def unreliable_operation():
        >>>     # May fail
        >>>     pass
        >>> 
        >>> result = await retry_async(unreliable_operation, max_attempts=3)
    """
    current_delay = delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"Failed after {max_attempts} attempts")
    
    raise last_exception


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC TIMEOUT
# ════════════════════════════════════════════════════════════════════════════════

async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Default value if timeout occurs
    
    Returns:
        Result or default value
    
    Example:
        >>> async def slow_operation():
        >>>     await asyncio.sleep(10)
        >>> 
        >>> result = await with_timeout(slow_operation(), timeout=5.0, default=None)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC RATE LIMITING
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimiter:
    """Rate limiter for async operations."""
    
    max_calls: int
    time_window: float
    _calls: deque = None
    
    def __post_init__(self):
        if self._calls is None:
            self._calls = deque()
    
    async def acquire(self):
        """Acquire permission to make a call."""
        now = time.time()
        
        # Remove old calls outside time window
        while self._calls and self._calls[0] < now - self.time_window:
            self._calls.popleft()
        
        # Check if we can make a call
        if len(self._calls) >= self.max_calls:
            # Wait until oldest call expires
            wait_time = self._calls[0] + self.time_window - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Remove expired calls
                while self._calls and self._calls[0] < time.time() - self.time_window:
                    self._calls.popleft()
        
        # Record this call
        self._calls.append(time.time())


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC BATCHING
# ════════════════════════════════════════════════════════════════════════════════

async def batch_process_async(
    items: List[Any],
    processor: Callable[[Any], Awaitable[T]],
    batch_size: int = 10,
    max_concurrent: Optional[int] = None,
) -> List[T]:
    """
    Process items in batches asynchronously.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent operations (None for batch_size)
    
    Returns:
        List of results
    
    Example:
        >>> async def process_item(item):
        >>>     return item * 2
        >>> 
        >>> results = await batch_process_async(
        >>>     items=[1, 2, 3, 4, 5],
        >>>     processor=process_item,
        >>>     batch_size=2
        >>> )
    """
    if max_concurrent is None:
        max_concurrent = batch_size
    
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await processor(item)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch]
        )
        results.extend(batch_results)
    
    return results


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC PROGRESS TRACKING
# ════════════════════════════════════════════════════════════════════════════════

async def process_with_progress(
    items: List[Any],
    processor: Callable[[Any], Awaitable[T]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_concurrent: int = 10,
) -> List[T]:
    """
    Process items with progress tracking.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        progress_callback: Callback(current, total)
        max_concurrent: Maximum concurrent operations
    
    Returns:
        List of results
    
    Example:
        >>> def on_progress(current, total):
        >>>     print(f"Progress: {current}/{total}")
        >>> 
        >>> results = await process_with_progress(
        >>>     items=[1, 2, 3],
        >>>     processor=process_item,
        >>>     progress_callback=on_progress
        >>> )
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(items)
    
    async def process_with_tracking(item, index):
        nonlocal completed
        async with semaphore:
            result = await processor(item)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result
    
    tasks = [
        process_with_tracking(item, i)
        for i, item in enumerate(items)
    ]
    
    results = await asyncio.gather(*tasks)
    return results


__all__ = [
    "retry_async",
    "with_timeout",
    "RateLimiter",
    "batch_process_async",
    "process_with_progress",
]












