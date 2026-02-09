"""Retry utilities with exponential backoff."""

import asyncio
import time
import random
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def retry_async(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    Decorator for async retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"Max attempts reached for {func.__name__}: {e}",
                            exc_info=True
                        )
                        raise
                    
                    wait_time = min(
                        initial_wait * (exponential_base ** (attempt - 1)),
                        max_wait
                    )
                    
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"after {wait_time:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(wait_time)
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    Decorator for sync retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"Max attempts reached for {func.__name__}: {e}",
                            exc_info=True
                        )
                        raise
                    
                    wait_time = min(
                        initial_wait * (exponential_base ** (attempt - 1)),
                        max_wait
                    )
                    
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"after {wait_time:.2f}s: {e}"
                    )
                    
                    time.sleep(wait_time)
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator

