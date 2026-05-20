"""
Semaphore utilities
Advanced concurrency control
"""

from typing import TypeVar, Callable, Optional
from asyncio import Semaphore, BoundedSemaphore
import asyncio

T = TypeVar('T')
U = TypeVar('U')


class RateLimiter:
    """
    Rate limiter using semaphore
    """
    
    def __init__(self, max_concurrent: int):
        self.semaphore = Semaphore(max_concurrent)
    
    async def acquire(self) -> None:
        """Acquire rate limit"""
        await self.semaphore.acquire()
    
    def release(self) -> None:
        """Release rate limit"""
        self.semaphore.release()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


def create_rate_limiter(max_concurrent: int) -> RateLimiter:
    """Create new rate limiter"""
    return RateLimiter(max_concurrent)


async def with_semaphore(
    semaphore: Semaphore,
    func: Callable[[], U]
) -> U:
    """
    Execute function with semaphore
    
    Args:
        semaphore: Semaphore to use
        func: Function to execute
    
    Returns:
        Function result
    """
    async with semaphore:
        if asyncio.iscoroutinefunction(func):
            return await func()
        return func()


async def with_rate_limit(
    max_concurrent: int,
    func: Callable[[], U]
) -> U:
    """
    Execute function with rate limit
    
    Args:
        max_concurrent: Maximum concurrent executions
        func: Function to execute
    
    Returns:
        Function result
    """
    limiter = create_rate_limiter(max_concurrent)
    async with limiter:
        if asyncio.iscoroutinefunction(func):
            return await func()
        return func()

