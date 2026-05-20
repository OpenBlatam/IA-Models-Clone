"""
Retry utilities

Utilities for retrying operations with exponential backoff.
"""

import time
import logging
from typing import Callable, Type, Tuple, Optional
from functools import wraps

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry
        
    Usage:
        @retry(max_attempts=3, delay=1.0, exceptions=(DatabaseError,))
        def database_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations.
    
    Usage:
        with RetryableOperation(max_attempts=3) as op:
            result = risky_operation()
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type in self.exceptions and self.attempt < self.max_attempts:
            self.attempt += 1
            current_delay = self.delay * (self.backoff ** (self.attempt - 1))
            logger.warning(
                f"Operation failed (attempt {self.attempt}/{self.max_attempts}). "
                f"Retrying in {current_delay:.2f}s..."
            )
            time.sleep(current_delay)
            return True  # Suppress exception and retry
        return False  # Don't suppress exception













