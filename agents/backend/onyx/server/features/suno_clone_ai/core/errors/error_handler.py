"""
Error Handler

Utilities for error handling and recovery.
"""

import logging
import time
from typing import Callable, Optional, Any, Type
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Handle errors with retry and recovery."""
    
    @staticmethod
    def retry(
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Callable:
        """
        Retry decorator.
        
        Args:
            func: Function to retry
            max_retries: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise
        
        return wrapper
    
    @staticmethod
    def handle(
        func: Callable,
        default_return: Any = None,
        exceptions: tuple = (Exception,),
        log_error: bool = True
    ) -> Callable:
        """
        Error handling decorator.
        
        Args:
            func: Function to wrap
            default_return: Default return value on error
            exceptions: Exceptions to catch
            log_error: Whether to log errors
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        return wrapper


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Retry decorator.
    
    Args:
        max_retries: Maximum retry attempts
        delay: Initial delay
        exceptions: Exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        return ErrorHandler.retry(func, max_retries, delay, exceptions)
    return decorator


def handle_error(
    default_return: Any = None,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Error handling decorator.
    
    Args:
        default_return: Default return value
        exceptions: Exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        return ErrorHandler.handle(func, default_return, exceptions)
    return decorator



