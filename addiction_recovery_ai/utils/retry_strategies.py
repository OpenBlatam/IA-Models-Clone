"""
Retry strategy utilities
Advanced retry patterns
"""

from typing import Callable, TypeVar, Optional
from enum import Enum
import asyncio
import time

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED = "fixed"  # Fixed delay
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    CUSTOM = "custom"  # Custom function


async def retry_with_strategy(
    func: Callable[[], T],
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    initial_delay: float = 1.0,
    max_delay: Optional[float] = None,
    delay_func: Optional[Callable[[int], float]] = None
) -> T:
    """
    Retry function with strategy
    
    Args:
        func: Function to retry
        max_attempts: Maximum attempts
        strategy: Retry strategy
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        delay_func: Custom delay function
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = _calculate_delay(
                    attempt=attempt,
                    strategy=strategy,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    delay_func=delay_func
                )
                await asyncio.sleep(delay)
    
    raise last_exception


def _calculate_delay(
    attempt: int,
    strategy: RetryStrategy,
    initial_delay: float,
    max_delay: Optional[float],
    delay_func: Optional[Callable[[int], float]]
) -> float:
    """Calculate delay based on strategy"""
    if delay_func:
        delay = delay_func(attempt)
    elif strategy == RetryStrategy.FIXED:
        delay = initial_delay
    elif strategy == RetryStrategy.EXPONENTIAL:
        delay = initial_delay * (2 ** attempt)
    elif strategy == RetryStrategy.LINEAR:
        delay = initial_delay * (attempt + 1)
    else:
        delay = initial_delay
    
    if max_delay:
        delay = min(delay, max_delay)
    
    return delay


async def retry_with_jitter(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_jitter: float = 0.5
) -> T:
    """
    Retry with exponential backoff and jitter
    
    Args:
        func: Function to retry
        max_attempts: Maximum attempts
        base_delay: Base delay in seconds
        max_jitter: Maximum jitter in seconds
    
    Returns:
        Function result
    """
    import random
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, max_jitter)
                await asyncio.sleep(delay)
    
    raise last_exception

