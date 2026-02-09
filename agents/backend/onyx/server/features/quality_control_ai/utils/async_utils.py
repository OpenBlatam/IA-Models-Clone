"""
Async Utilities

Utility functions for async operations.
"""

import asyncio
from typing import List, Callable, Any, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


async def run_in_executor(
    func: Callable,
    *args,
    executor_type: str = "thread",
    **kwargs
) -> Any:
    """
    Run function in executor (thread or process).
    
    Args:
        func: Function to run
        *args: Positional arguments
        executor_type: Type of executor ('thread' or 'process')
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    
    if executor_type == "thread":
        executor = ThreadPoolExecutor()
    elif executor_type == "process":
        executor = ProcessPoolExecutor()
    else:
        raise ValueError(f"Invalid executor_type: {executor_type}")
    
    try:
        return await loop.run_in_executor(executor, func, *args, **kwargs)
    finally:
        executor.shutdown(wait=False)


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int = 10
) -> List[Any]:
    """
    Gather coroutines with concurrency limit.
    
    Args:
        coros: List of coroutines
        limit: Maximum concurrent executions
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[run_with_semaphore(coro) for coro in coros])


async def timeout_after(
    coro: Coroutine,
    timeout: float
) -> Any:
    """
    Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
    
    Returns:
        Coroutine result
    
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def async_to_sync(coro: Coroutine) -> Any:
    """
    Run async coroutine synchronously.
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)



