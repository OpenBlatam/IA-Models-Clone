"""
Utilidades para optimización de performance
"""

import time
import functools
import logging
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def measure_time(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator para medir tiempo de ejecución"""
    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    return sync_wrapper


def cache_result(ttl: int = 3600):
    """Decorator para cachear resultados de funciones"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if cache_key in cache:
                cache_time = cache_times.get(cache_key, 0)
                if current_time - cache_time < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[cache_key]
            
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            cache_times[cache_key] = current_time
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if cache_key in cache:
                cache_time = cache_times.get(cache_key, 0)
                if current_time - cache_time < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[cache_key]
            
            result = func(*args, **kwargs)
            cache[cache_key] = result
            cache_times[cache_key] = current_time
            return result
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator para reintentar en caso de fallo"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import asyncio
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import time
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    
    return decorator

