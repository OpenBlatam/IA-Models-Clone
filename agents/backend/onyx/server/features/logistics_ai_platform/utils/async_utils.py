"""
Async utility functions

This module provides common async utilities to reduce code duplication.
"""

import asyncio
from typing import Callable, Awaitable, Any, List, TypeVar, Optional
from functools import wraps

T = TypeVar('T')


async def run_in_background(coro: Awaitable[Any]) -> None:
    """
    Run coroutine in background if event loop is running, otherwise await it
    
    Args:
        coro: Coroutine to run
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            await coro
    except Exception:
        pass


async def gather_with_errors(
    *coros: Awaitable,
    return_exceptions: bool = True
) -> List[Any]:
    """Gather coroutines and handle errors gracefully"""
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def timeout_after(
    coro: Awaitable,
    timeout: float,
    default: Any = None
) -> Any:
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> T:
    """Retry async function with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (backoff ** attempt))
            else:
                raise last_exception
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed without exception")


def async_to_sync(func: Callable) -> Callable:
    """Convert async function to sync (for compatibility)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


async def chunked_process(
    items: List[T],
    processor: Callable[[T], Awaitable[Any]],
    chunk_size: int = 100
) -> List[Any]:
    """Process items in chunks"""
    results = []
    
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_results = await asyncio.gather(
            *[processor(item) for item in chunk],
            return_exceptions=True
        )
        results.extend(chunk_results)
    
    return results

