"""
PDF Variantes API - Resilience Patterns
Real-world resilience patterns for production systems
"""

import asyncio
import time
from typing import Any, Callable, Optional, Dict, List, TypeVar, Awaitable
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FallbackStrategy:
    """Fallback strategy for degraded service"""
    
    @staticmethod
    async def return_default(default_value: Any):
        """Return default value"""
        return default_value
    
    @staticmethod
    async def return_cached(cache_func: Callable[[], Awaitable[Any]]):
        """Return cached value"""
        try:
            return await cache_func()
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")
            return None
    
    @staticmethod
    async def return_empty_list():
        """Return empty list as fallback"""
        return []
    
    @staticmethod
    async def return_empty_dict():
        """Return empty dict as fallback"""
        return {}


async def with_fallback(
    primary_func: Callable[..., Awaitable[T]],
    fallback_func: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """
    Execute primary function with fallback
    
    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        *args, **kwargs: Arguments to pass to functions
    """
    try:
        return await primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Primary function failed: {e}. Using fallback...")
        try:
            return await fallback_func(*args, **kwargs)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise


def degrade_gracefully(
    fallback_value: Any = None,
    log_fallback: bool = True
):
    """
    Decorator to gracefully degrade on errors
    
    Args:
        fallback_value: Value to return on error
        log_fallback: Whether to log fallback usage
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"{func.__name__} failed: {e}. Returning fallback value."
                    )
                return fallback_value
        
        return wrapper
    return decorator


class Bulkhead:
    """
    Bulkhead pattern - isolate resources to prevent cascading failures
    Limits concurrent executions per resource pool
    """
    
    def __init__(self, max_concurrent: int = 10, pool_name: str = "default"):
        self.max_concurrent = max_concurrent
        self.pool_name = pool_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
    
    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function within bulkhead limits"""
        async with self.semaphore:
            self.active_count += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_count -= 1
    
    def get_active_count(self) -> int:
        """Get current active execution count"""
        return self.active_count


class RequestQueue:
    """Queue for managing request processing with capacity limits"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.rejected_count = 0
    
    async def enqueue(self, item: Any, timeout: float = 5.0) -> bool:
        """
        Add item to queue
        
        Returns:
            True if enqueued, False if queue is full
        """
        try:
            await asyncio.wait_for(
                self.queue.put(item),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            self.rejected_count += 1
            logger.warning(
                f"Queue full, request rejected. Total rejected: {self.rejected_count}"
            )
            return False
    
    async def dequeue(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue"""
        if timeout:
            return await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout
            )
        return await self.queue.get()
    
    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()


def batched_process(
    items: List[Any],
    batch_size: int = 10,
    delay_between_batches: float = 0.1
):
    """
    Process items in batches with delay
    
    Args:
        items: Items to process
        batch_size: Items per batch
        delay_between_batches: Delay in seconds between batches
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        
        if i + batch_size < len(items) and delay_between_batches > 0:
            time.sleep(delay_between_batches)


async def async_batched_process(
    items: List[Any],
    batch_size: int = 10,
    delay_between_batches: float = 0.1
):
    """Async version of batched processing"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        
        if i + batch_size < len(items) and delay_between_batches > 0:
            await asyncio.sleep(delay_between_batches)






