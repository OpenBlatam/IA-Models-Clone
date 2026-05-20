"""
Decorators
==========
Decoradores útiles para la aplicación.
"""

import functools
import time
import asyncio
from typing import Callable, Any
from ...utils.logger import get_logger
from ...utils.metrics import request_duration_seconds

logger = get_logger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorador para reintentar en caso de fallo.
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos en segundos
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, retrying...",
                            error=str(e)
                        )
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}",
                            error=str(e)
                        )
            raise last_exception
        return wrapper
    return decorator


def cache_result(ttl: int = 3600):
    """
    Decorador para cachear resultados.
    
    Args:
        ttl: Time to live en segundos
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar clave de cache basada en args y kwargs
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Intentar obtener de cache
            from ...utils.cache import get_cached_response, set_cached_response
            cached = await get_cached_response(cache_key)
            if cached:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            await set_cached_response(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorador para registrar tiempo de ejecución."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Registrar métrica
            request_duration_seconds.labels(
                endpoint=func.__name__,
                method="async"
            ).observe(duration)
            
            logger.info(
                f"{func.__name__} executed in {duration:.3f}s"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {duration:.3f}s",
                error=str(e)
            )
            raise
    return wrapper

