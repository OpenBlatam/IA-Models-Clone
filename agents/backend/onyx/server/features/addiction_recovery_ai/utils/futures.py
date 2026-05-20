"""
Future utilities
Advanced future patterns
"""

from typing import TypeVar, List, Callable, Any
from asyncio import Future, gather, create_task
import asyncio

T = TypeVar('T')
U = TypeVar('U')


async def all_futures(futures: List[Future[T]]) -> List[T]:
    """
    Wait for all futures to complete
    
    Args:
        futures: List of futures
    
    Returns:
        List of results
    """
    return await gather(*futures)


async def any_future(futures: List[Future[T]]) -> T:
    """
    Wait for any future to complete
    
    Args:
        futures: List of futures
    
    Returns:
        First completed result
    """
    done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
    
    # Cancel pending futures
    for future in pending:
        future.cancel()
    
    # Return first completed result
    return await done.pop()


async def race_futures(futures: List[Future[T]]) -> T:
    """
    Race futures and return first result
    
    Args:
        futures: List of futures
    
    Returns:
        First completed result
    """
    return await any_future(futures)


async def timeout_future(future: Future[T], timeout: float) -> T:
    """
    Add timeout to future
    
    Args:
        future: Future to timeout
        timeout: Timeout in seconds
    
    Returns:
        Future result
    
    Raises:
        TimeoutError if timeout exceeded
    """
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Future timed out after {timeout} seconds")


def create_future_from_value(value: T) -> Future[T]:
    """
    Create completed future from value
    
    Args:
        value: Value to wrap
    
    Returns:
        Completed future
    """
    future: Future[T] = Future()
    future.set_result(value)
    return future


def create_future_from_error(error: Exception) -> Future:
    """
    Create failed future from error
    
    Args:
        error: Error to wrap
    
    Returns:
        Failed future
    """
    future: Future = Future()
    future.set_exception(error)
    return future

