"""
Retry Utilities for Piel Mejorador AI SAM3
==========================================

Unified retry mechanisms with exponential backoff and configurable strategies.
"""

import asyncio
import logging
import time
from typing import Callable, Type, Tuple, Any, Optional, TypeVar, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy:
    """Retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    CUSTOM = "custom"


class RetryUtils:
    """Unified retry utilities."""
    
    @staticmethod
    def calculate_delay(
        attempt: int,
        base_delay: float,
        strategy: str = RetryStrategy.EXPONENTIAL,
        max_delay: Optional[float] = None,
        multiplier: float = 2.0,
        custom_func: Optional[Callable[[int, float], float]] = None
    ) -> float:
        """
        Calculate retry delay based on strategy.
        
        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds
            strategy: Retry strategy (exponential, linear, fixed, custom)
            max_delay: Maximum delay in seconds
            multiplier: Multiplier for exponential/linear strategies
            custom_func: Custom delay calculation function
            
        Returns:
            Delay in seconds
        """
        if strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (multiplier ** attempt)
        elif strategy == RetryStrategy.LINEAR:
            delay = base_delay * (1 + attempt * multiplier)
        elif strategy == RetryStrategy.FIXED:
            delay = base_delay
        elif strategy == RetryStrategy.CUSTOM and custom_func:
            delay = custom_func(attempt, base_delay)
        else:
            delay = base_delay
        
        if max_delay:
            delay = min(delay, max_delay)
        
        return delay
    
    @staticmethod
    async def retry_async(
        func: Callable[[], Awaitable[T]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        strategy: str = RetryStrategy.EXPONENTIAL,
        max_delay: Optional[float] = None,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        operation_name: str = "operation",
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ) -> T:
        """
        Retry async function with configurable strategy.
        
        Args:
            func: Async function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            strategy: Retry strategy
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types to retry on
            operation_name: Name of operation for logging
            on_retry: Optional callback on retry
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = RetryUtils.calculate_delay(
                        attempt,
                        base_delay,
                        strategy,
                        max_delay
                    )
                    
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        try:
                            if asyncio.iscoroutinefunction(on_retry):
                                await on_retry(e, attempt)
                            else:
                                on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.warning(f"Error in retry callback: {callback_error}")
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts")
        
        raise last_exception
    
    @staticmethod
    def retry_sync(
        func: Callable[[], T],
        max_retries: int = 3,
        base_delay: float = 1.0,
        strategy: str = RetryStrategy.EXPONENTIAL,
        max_delay: Optional[float] = None,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        operation_name: str = "operation",
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ) -> T:
        """
        Retry sync function with configurable strategy.
        
        Args:
            func: Sync function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            strategy: Retry strategy
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types to retry on
            operation_name: Name of operation for logging
            on_retry: Optional callback on retry
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = RetryUtils.calculate_delay(
                        attempt,
                        base_delay,
                        strategy,
                        max_delay
                    )
                    
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.warning(f"Error in retry callback: {callback_error}")
                    
                    time.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts")
        
        raise last_exception
    
    @staticmethod
    def retry_decorator(
        max_retries: int = 3,
        base_delay: float = 1.0,
        strategy: str = RetryStrategy.EXPONENTIAL,
        max_delay: Optional[float] = None,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        operation_name: Optional[str] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        """
        Create retry decorator.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            strategy: Retry strategy
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types to retry on
            operation_name: Name of operation for logging
            on_retry: Optional callback on retry
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or func.__name__
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async def call_func():
                        return await func(*args, **kwargs)
                    
                    return await RetryUtils.retry_async(
                        call_func,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        strategy=strategy,
                        max_delay=max_delay,
                        retryable_exceptions=retryable_exceptions,
                        operation_name=op_name,
                        on_retry=on_retry
                    )
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    def call_func():
                        return func(*args, **kwargs)
                    
                    return RetryUtils.retry_sync(
                        call_func,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        strategy=strategy,
                        max_delay=max_delay,
                        retryable_exceptions=retryable_exceptions,
                        operation_name=op_name,
                        on_retry=on_retry
                    )
                
                return sync_wrapper
        
        return decorator


# Convenience functions
def retry_async(func: Callable[[], Awaitable[T]], **kwargs) -> T:
    """Retry async function."""
    return RetryUtils.retry_async(func, **kwargs)


def retry_sync(func: Callable[[], T], **kwargs) -> T:
    """Retry sync function."""
    return RetryUtils.retry_sync(func, **kwargs)


def retry(max_retries: int = 3, **kwargs):
    """Create retry decorator."""
    return RetryUtils.retry_decorator(max_retries=max_retries, **kwargs)

