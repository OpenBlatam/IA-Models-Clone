"""
Retry utilities with exponential backoff for resilient service calls
"""

import asyncio
import time
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class RetryConfig:
    """Retry configuration"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on


def retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying function calls with exponential backoff
    
    Args:
        config: Retry configuration
        on_retry: Optional callback called on each retry
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts:
                        delay = _calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise
            
            if last_exception:
                raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts:
                        delay = _calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise
            
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff"""
    delay = config.initial_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        import random
        # Add random jitter (±25%)
        jitter_amount = delay * 0.25
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Retry async function with exponential backoff
    
    Args:
        func: Async function to retry
        *args: Function arguments
        config: Retry configuration
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retry_on as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                delay = _calculate_delay(attempt, config)
                logger.warning(
                    f"Retry {attempt}/{config.max_attempts} after {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed: {e}")
                raise
    
    if last_exception:
        raise last_exception















