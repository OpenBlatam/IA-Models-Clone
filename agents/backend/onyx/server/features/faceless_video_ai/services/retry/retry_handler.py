"""
Retry Handler
Intelligent retry with exponential backoff
"""

from typing import Callable, Optional, Type, Tuple, List
import asyncio
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


class RetryHandler:
    """Intelligent retry handler with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        import random
        
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (±20%)
            jitter_amount = delay * 0.2
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ):
        """
        Retry async function with exponential backoff
        
        Args:
            func: Async function to retry
            *args: Function arguments
            retryable_exceptions: Exceptions that should trigger retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    raise
        
        raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ):
        """
        Retry sync function with exponential backoff
        
        Args:
            func: Sync function to retry
            *args: Function arguments
            retryable_exceptions: Exceptions that should trigger retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    raise
        
        raise last_exception
    
    def decorator(
        self,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Decorator for retry logic
        
        Args:
            retryable_exceptions: Exceptions that should trigger retry
        """
        def decorator_func(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self.retry_async(
                        func,
                        *args,
                        retryable_exceptions=retryable_exceptions,
                        **kwargs
                    )
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self.retry_sync(
                        func,
                        *args,
                        retryable_exceptions=retryable_exceptions,
                        **kwargs
                    )
                return sync_wrapper
        return decorator_func


_retry_handler: Optional[RetryHandler] = None


def get_retry_handler(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0
) -> RetryHandler:
    """Get retry handler instance (singleton)"""
    global _retry_handler
    if _retry_handler is None:
        _retry_handler = RetryHandler(
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay
        )
    return _retry_handler

