"""
Async function composition utilities
Helpers for composing async functions
"""

from typing import Callable, TypeVar, Any, Awaitable
import asyncio

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


async def compose_async(*functions: Callable) -> Callable:
    """
    Compose multiple async functions into a single function
    
    Args:
        *functions: Async functions to compose (right to left)
    
    Returns:
        Composed async function
    """
    if not functions:
        raise ValueError("At least one function is required")
    
    async def composed(value: Any) -> Any:
        result = value
        for func in reversed(functions):
            if asyncio.iscoroutinefunction(func):
                result = await func(result)
            else:
                result = func(result)
        return result
    
    return composed


async def pipe_async(value: Any, *functions: Callable) -> Any:
    """
    Pipe value through multiple async functions (left to right)
    
    Args:
        value: Initial value
        *functions: Async functions to apply
    
    Returns:
        Final result
    """
    result = value
    for func in functions:
        if asyncio.iscoroutinefunction(func):
            result = await func(result)
        else:
            result = func(result)
    return result


async def parallel_map(
    items: list[T],
    func: Callable[[T], Awaitable[U]],
    max_concurrent: int = 10
) -> list[U]:
    """
    Map async function over items in parallel
    
    Args:
        items: List of items to process
        func: Async function to apply
        max_concurrent: Maximum concurrent operations
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: T) -> U:
        async with semaphore:
            return await func(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> T:
    """
    Retry async function with exponential backoff
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
    
    Returns:
        Result from function
    
    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
    
    raise last_exception

