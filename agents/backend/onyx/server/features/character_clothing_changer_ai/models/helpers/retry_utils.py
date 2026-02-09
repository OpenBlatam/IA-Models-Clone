"""
Retry Utilities
================

Utilities for retrying operations with exponential backoff.
"""

import time
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_failure(max_attempts: int = 3, delay: float = 0.5):
    """
    Decorator to retry operations on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Base delay in seconds (will be multiplied by attempt number)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (attempt + 1)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Failed to execute {func.__name__} after {max_attempts} attempts")
        return wrapper
    return decorator


