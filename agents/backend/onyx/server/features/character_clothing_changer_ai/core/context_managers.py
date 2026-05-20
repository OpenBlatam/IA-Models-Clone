"""
Context Managers
================

Common context managers for operations.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OperationContext:
    """Context for operations."""
    operation_name: str
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def elapsed(self) -> float:
        """Get elapsed time."""
        return time.time() - self.start_time


@asynccontextmanager
async def timed_operation(name: str, log: bool = True):
    """
    Context manager for timed operations.
    
    Args:
        name: Operation name
        log: Whether to log timing
        
    Usage:
        async with timed_operation("process_image"):
            # operation code
    """
    start = time.time()
    context = OperationContext(operation_name=name)
    
    try:
        yield context
    finally:
        elapsed = time.time() - start
        if log:
            logger.debug(f"Operation {name} took {elapsed:.2f}s")


@asynccontextmanager
async def retry_operation(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential: bool = True,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None
):
    """
    Context manager for retry operations.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay in seconds
        exponential: Whether to use exponential backoff
        on_retry: Optional callback on retry
        
    Usage:
        async with retry_operation(max_attempts=3):
            # operation code that may fail
    """
    attempt = 0
    last_error = None
    
    while attempt < max_attempts:
        try:
            yield attempt
            return
        except Exception as e:
            attempt += 1
            last_error = e
            
            if attempt >= max_attempts:
                raise
            
            wait_time = delay * (2 ** (attempt - 1)) if exponential else delay
            
            if on_retry:
                await on_retry(attempt, e)
            
            logger.warning(f"Operation failed (attempt {attempt}/{max_attempts}), retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
    
    if last_error:
        raise last_error


@asynccontextmanager
async def rate_limited_operation(
    rate_limiter,
    user_id: Optional[str] = None
):
    """
    Context manager for rate-limited operations.
    
    Args:
        rate_limiter: Rate limiter instance
        user_id: Optional user ID
        
    Usage:
        async with rate_limited_operation(rate_limiter, user_id="user123"):
            # operation code
    """
    await rate_limiter.is_allowed(user_id)
    try:
        yield
    finally:
        pass  # Rate limiter handles release automatically


@asynccontextmanager
async def cached_operation(
    cache_manager,
    key: str,
    ttl: Optional[int] = None
):
    """
    Context manager for cached operations.
    
    Args:
        cache_manager: Cache manager instance
        key: Cache key
        ttl: Optional TTL
        
    Usage:
        async with cached_operation(cache_manager, "key") as cached:
            if cached.exists():
                return cached.get()
            # compute value
            cached.set(value)
    """
    class CacheContext:
        def __init__(self, cache, cache_key, ttl_value):
            self.cache = cache
            self.key = cache_key
            self.ttl = ttl_value
            self._value = None
        
        def exists(self) -> bool:
            """Check if cached value exists."""
            return self.cache.exists(self.key)
        
        def get(self):
            """Get cached value."""
            if self._value is None:
                self._value = self.cache.get(self.key)
            return self._value
        
        def set(self, value: Any):
            """Set cached value."""
            self._value = value
            self.cache.set(self.key, value, ttl=self.ttl)
    
    context = CacheContext(cache_manager, key, ttl)
    try:
        yield context
    finally:
        pass


@asynccontextmanager
async def monitored_operation(
    metrics_collector,
    operation_name: str,
    tags: Optional[Dict[str, str]] = None
):
    """
    Context manager for monitored operations.
    
    Args:
        metrics_collector: Metrics collector instance
        operation_name: Operation name
        tags: Optional tags
        
    Usage:
        async with monitored_operation(metrics, "process_image"):
            # operation code
    """
    start = time.time()
    tags = tags or {}
    
    try:
        yield
        # Record success
        metrics_collector.increment(
            f"{operation_name}.success",
            tags=tags
        )
    except Exception as e:
        # Record failure
        metrics_collector.increment(
            f"{operation_name}.error",
            tags={**tags, "error": str(e)}
        )
        raise
    finally:
        # Record duration
        duration = time.time() - start
        metrics_collector.record(
            f"{operation_name}.duration",
            duration,
            tags=tags
        )

