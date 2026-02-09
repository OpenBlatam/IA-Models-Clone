"""
Decorators - Decorators comunes
================================

Decorators reutilizables para servicios y funciones.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps
import asyncio

from ..debug.debug_logger import get_debug_logger
from ..debug.profiler import get_profiler
from ..optimizations.caching import CacheDecorator

logger = logging.getLogger(__name__)


def timed(func: Callable):
    """
    Decorator para medir tiempo de ejecución.
    
    Example:
        @timed
        async def my_function():
            pass
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def logged(func: Callable):
    """
    Decorator para logging automático.
    
    Example:
        @logged
        async def my_function():
            pass
    """
    debug_logger = get_debug_logger()
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        debug_logger.debug(f"Calling {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            debug_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            debug_logger.error(f"{func.__name__} failed: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        debug_logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            debug_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            debug_logger.error(f"{func.__name__} failed: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def cached(ttl: int = 3600, tags: Optional[list] = None):
    """
    Decorator para cachear resultados.
    
    Example:
        @cached(ttl=3600, tags=["projects"])
        async def get_project(project_id: str):
            pass
    """
    return CacheDecorator(ttl=ttl, tags=tags or [])


def profiled(func: Callable):
    """
    Decorator para profiling.
    
    Example:
        @profiled
        async def my_function():
            pass
    """
    profiler = get_profiler()
    return profiler.profile_function(func)


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator para retry en caso de fallo.
    
    Example:
        @retry_on_failure(max_attempts=3, delay=1.0)
        async def my_function():
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                            f"retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                            f"retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator















