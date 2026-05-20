"""
Async Helpers
=============

Async utilities for concurrent operations.
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional, Dict, TypeVar, Awaitable
from functools import wraps
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncBatchProcessor:
    """
    Process items in batches asynchronously.
    
    Features:
    - Batch processing
    - Concurrency control
    - Error handling
    - Progress tracking
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        retry_count: int = 3
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent: Maximum concurrent batches
            retry_count: Number of retries on failure
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.retry_count = retry_count
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._logger = logger
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[Any]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            on_progress: Optional progress callback (current, total)
        
        Returns:
            List of results
        """
        results = []
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self._process_batch_with_retry(batch, processor)
            results.extend(batch_results)
            
            if on_progress:
                on_progress(min(i + self.batch_size, total), total)
        
        return results
    
    async def _process_batch_with_retry(
        self,
        batch: List[T],
        processor: Callable[[T], Awaitable[Any]]
    ) -> List[Any]:
        """Process batch with retry logic."""
        tasks = []
        
        for item in batch:
            task = self._process_item_with_retry(item, processor)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_item_with_retry(
        self,
        item: T,
        processor: Callable[[T], Awaitable[Any]]
    ) -> Any:
        """Process item with retry logic."""
        async with self.semaphore:
            for attempt in range(self.retry_count):
                try:
                    return await processor(item)
                except Exception as e:
                    if attempt == self.retry_count - 1:
                        self._logger.error(f"Failed to process item after {self.retry_count} attempts: {str(e)}")
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff


class AsyncCache:
    """Async cache with TTL support."""
    
    def __init__(self, default_ttl: float = 3600.0):
        """
        Initialize async cache.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._logger = logger
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.get("expires_at", 0) < time.time():
                del self.cache[key]
                return None
            
            return entry["value"]
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            expires_at = time.time() + (ttl or self.default_ttl)
            self.cache[key] = {
                "value": value,
                "expires_at": expires_at
            }
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self.cache)
            self.cache.clear()
            return count


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Decorator for async function retry.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        return wrapper
    return decorator


async def gather_with_limit(
    tasks: List[Awaitable[T]],
    limit: int = 10
) -> List[T]:
    """
    Gather tasks with concurrency limit.
    
    Args:
        tasks: List of awaitables
        limit: Maximum concurrent tasks
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[limited_task(task) for task in tasks])




