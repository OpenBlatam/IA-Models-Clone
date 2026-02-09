"""
Retry service for Multi-Model API
Handles retry logic with exponential backoff
"""

import asyncio
import logging
from typing import Callable, Awaitable, TypeVar, Optional, List
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            # Add random jitter (±20%)
            jitter_amount = delay * 0.2
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure positive
        
        return delay


class RetryService:
    """Service for handling retries with exponential backoff"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry service
        
        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
    
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        operation_name: str = "operation",
        context: Optional[dict] = None
    ) -> T:
        """
        Execute async function with retry logic
        
        Args:
            func: Async function to execute
            operation_name: Name of operation for logging
            context: Optional context for logging
            
        Returns:
            Result of function execution
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        context = context or {}
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func()
                
                # Log success if retried
                if attempt > 0:
                    logger.info(
                        f"{operation_name} succeeded on attempt {attempt + 1}",
                        extra={**context, "attempt": attempt + 1}
                    )
                
                return result
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    logger.error(
                        f"{operation_name} failed after {self.config.max_attempts} attempts",
                        extra={**context, "attempt": attempt + 1, "error": str(e)},
                        exc_info=True
                    )
                    raise
                
                # Calculate delay
                delay = self.config.calculate_delay(attempt)
                
                logger.warning(
                    f"{operation_name} failed on attempt {attempt + 1}/{self.config.max_attempts}, "
                    f"retrying in {delay:.2f}s",
                    extra={**context, "attempt": attempt + 1, "delay": delay, "error": str(e)}
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise Exception(f"{operation_name} failed after {self.config.max_attempts} attempts")
    
    def create_retry_config(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ) -> RetryConfig:
        """
        Create custom retry configuration
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
            retryable_exceptions: Tuple of exceptions that should trigger retry
            
        Returns:
            RetryConfig instance
        """
        return RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions
        )


def retry_on_failure(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions on failure
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        retryable_exceptions: Tuple of exceptions that should trigger retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[[], Awaitable[T]]) -> Callable[[], Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_service = RetryService(
                RetryConfig(
                    max_attempts=max_attempts,
                    initial_delay=initial_delay,
                    retryable_exceptions=retryable_exceptions
                )
            )
            return await retry_service.execute_with_retry(
                lambda: func(*args, **kwargs),
                operation_name=func.__name__
            )
        return wrapper
    return decorator




