"""
Unified async utilities for common async patterns.

Consolidates async operations into reusable utilities.
"""

import asyncio
import logging
from typing import List, Callable, Any, TypeVar, Awaitable, Optional, Dict
from functools import wraps
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class AsyncUtils:
    """Unified async utilities."""
    
    @staticmethod
    async def gather_with_limit(
        tasks: List[Awaitable[T]],
        limit: int = 10
    ) -> List[T]:
        """
        Execute multiple async tasks with concurrency limit.
        
        Args:
            tasks: List of async tasks
            limit: Maximum concurrent tasks
        
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(limit)
        
        async def bounded_task(task: Awaitable[T]) -> T:
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[bounded_task(task) for task in tasks])
    
    @staticmethod
    async def timeout(
        coro: Awaitable[T],
        timeout: float,
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Execute coroutine with timeout.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            default: Default value on timeout
        
        Returns:
            Result or default
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return default
    
    @staticmethod
    @asynccontextmanager
    async def retry_context(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0
    ):
        """
        Context manager for retry logic.
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay
            backoff: Backoff multiplier
        """
        attempt = 0
        current_delay = delay
        
        while attempt < max_attempts:
            try:
                yield attempt
                return
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                await asyncio.sleep(current_delay)
                current_delay *= backoff


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for async retry logic.
    
    Args:
        max_attempts: Maximum attempts
        delay: Initial delay
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


async def gather_with_limit(
    tasks: List[Awaitable[T]],
    limit: int = 10
) -> List[T]:
    """Convenience function for gathering with limit."""
    return await AsyncUtils.gather_with_limit(tasks, limit)


async def timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """Convenience function for timeout."""
    return await AsyncUtils.timeout(coro, timeout, default)

