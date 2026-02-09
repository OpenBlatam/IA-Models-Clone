"""
Retry Utilities
===============

Utilidades para reintentos con backoff exponencial.
"""

import logging
import asyncio
from typing import Callable, TypeVar, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorador para reintentos con backoff exponencial.
    
    Args:
        max_retries: Número máximo de reintentos
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        exponential_base: Base exponencial
        exceptions: Excepciones que deben causar reintento
    
    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            if last_exception:
                raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            if last_exception:
                raise last_exception
        
        # Detectar si es async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    
    return decorator




