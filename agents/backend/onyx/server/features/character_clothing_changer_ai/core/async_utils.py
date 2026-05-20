"""
Async Utilities
================

Common async utilities and helpers.
"""

import asyncio
import logging
from typing import Any, Optional, List, Callable, Awaitable, TypeVar, Coroutine
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def gather_with_exceptions(*coros: Coroutine) -> List[Any]:
    """
    Gather coroutines and collect exceptions.
    
    Args:
        *coros: Coroutines to gather
        
    Returns:
        List of results (exceptions included as None)
    """
    results = []
    
    async def safe_execute(coro: Coroutine) -> Any:
        try:
            return await coro
        except Exception as e:
            logger.error(f"Error in gathered coroutine: {e}")
            return None
    
    tasks = [asyncio.create_task(safe_execute(coro)) for coro in coros]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results


async def timeout_after(seconds: float, coro: Coroutine) -> Any:
    """
    Execute coroutine with timeout.
    
    Args:
        seconds: Timeout in seconds
        coro: Coroutine to execute
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    return await asyncio.wait_for(coro, timeout=seconds)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential: bool = True,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None
) -> T:
    """
    Retry async function.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum retry attempts
        delay: Initial delay in seconds
        exponential: Whether to use exponential backoff
        on_retry: Optional callback on retry
        
    Returns:
        Function result
        
    Raises:
        Exception: Last exception if all attempts fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_error = e
            
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt) if exponential else delay
                
                if on_retry:
                    await on_retry(attempt + 1, e)
                
                logger.warning(f"Retry {attempt + 1}/{max_attempts} after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    if last_error:
        raise last_error
    raise Exception("All retry attempts failed")


def async_to_sync(coro: Coroutine) -> Any:
    """
    Convert async function to sync (blocking).
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


def ensure_async(func: Callable) -> Callable[[], Awaitable[Any]]:
    """
    Ensure function is async.
    
    Args:
        func: Function to ensure async
        
    Returns:
        Async function
    """
    if asyncio.iscoroutinefunction(func):
        return func
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return async_wrapper


class AsyncLock:
    """Async lock with timeout."""
    
    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize async lock.
        
        Args:
            timeout: Optional timeout in seconds
        """
        self.lock = asyncio.Lock()
        self.timeout = timeout
    
    async def acquire(self):
        """Acquire lock."""
        if self.timeout:
            await asyncio.wait_for(self.lock.acquire(), timeout=self.timeout)
        else:
            await self.lock.acquire()
    
    def release(self):
        """Release lock."""
        self.lock.release()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()


class AsyncSemaphore:
    """Async semaphore with timeout."""
    
    def __init__(self, value: int, timeout: Optional[float] = None):
        """
        Initialize semaphore.
        
        Args:
            value: Semaphore value
            timeout: Optional timeout in seconds
        """
        self.semaphore = asyncio.Semaphore(value)
        self.timeout = timeout
    
    async def acquire(self):
        """Acquire semaphore."""
        if self.timeout:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=self.timeout)
        else:
            await self.semaphore.acquire()
    
    def release(self):
        """Release semaphore."""
        self.semaphore.release()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()

