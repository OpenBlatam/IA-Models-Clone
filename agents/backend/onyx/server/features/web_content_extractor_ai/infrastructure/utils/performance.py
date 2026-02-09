"""
Utilidades de performance y monitoreo
"""

import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """Decorator para medir tiempo de ejecución"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} ejecutado en {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} falló después de {elapsed:.2f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} ejecutado en {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} falló después de {elapsed:.2f}s: {e}")
            raise
    
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
        return async_wrapper
    return sync_wrapper








