"""
Decorators for Physical Store Designer AI

This module provides useful decorators for logging, retry logic, validation,
and caching.
"""

import time
import asyncio
import functools
from typing import Callable, Any, TypeVar, ParamSpec, Dict
from ..core.logging_config import get_logger

logger = get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def log_execution_time(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator para loggear tiempo de ejecución con alta precisión"""
    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()  # Mayor precisión que time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000
            logger.debug(
                f"{func.__name__} ejecutado en {duration_ms:.2f}ms",
                extra={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "duration_ms": duration_ms
                }
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000
            logger.error(
                f"{func.__name__} falló después de {duration_ms:.2f}ms: {e}",
                extra={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "duration_ms": duration_ms
                },
                exc_info=True
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()  # Mayor precisión que time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000
            logger.debug(
                f"{func.__name__} ejecutado en {duration_ms:.2f}ms",
                extra={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "duration_ms": duration_ms
                }
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000
            logger.error(
                f"{func.__name__} falló después de {duration_ms:.2f}ms: {e}",
                extra={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "duration_ms": duration_ms
                },
                exc_info=True
            )
            raise
    
    if functools.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator para reintentar en caso de fallo con backoff exponencial
    
    Args:
        max_retries: Número máximo de intentos (default: 3)
        delay: Delay inicial en segundos (default: 1.0)
        
    Returns:
        Decorator function that wraps the function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} falló (intento {attempt + 1}/{max_retries}): {e}",
                            extra={"function": func.__name__, "attempt": attempt + 1}
                        )
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(
                            f"{func.__name__} falló después de {max_retries} intentos",
                            extra={"function": func.__name__},
                            exc_info=True
                        )
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} falló (intento {attempt + 1}/{max_retries}): {e}",
                            extra={"function": func.__name__, "attempt": attempt + 1}
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(
                            f"{func.__name__} falló después de {max_retries} intentos",
                            extra={"function": func.__name__},
                            exc_info=True
                        )
            raise last_exception
        
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_input(*validators: Callable[[Any], bool]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator para validar inputs con soporte async/sync
    
    Args:
        *validators: Funciones validadoras que reciben un argumento y retornan bool
        
    Returns:
        Decorator function that validates inputs before execution
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validar argumentos posicionales
            for i, validator in enumerate(validators):
                if i < len(args):
                    if not validator(args[i]):
                        from .exceptions import ValidationError
                        raise ValidationError(
                            f"Validación falló para argumento {i} en {func.__name__}"
                        )
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validar argumentos posicionales
            for i, validator in enumerate(validators):
                if i < len(args):
                    if not validator(args[i]):
                        from .exceptions import ValidationError
                        raise ValidationError(
                            f"Validación falló para argumento {i} en {func.__name__}"
                        )
            return func(*args, **kwargs)
        
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def cache_result(ttl: float = 3600.0) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator para cachear resultados de funciones (simple in-memory cache)
    
    Args:
        ttl: Time to live en segundos (default: 3600.0 = 1 hora)
        
    Returns:
        Decorator function that caches function results
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: Dict[str, tuple[T, float]] = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Crear clave de caché simple
            cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Verificar caché
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit para {func.__name__}")
                    return result
                else:
                    del cache[cache_key]
            
            # Ejecutar función y cachear
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cache miss para {func.__name__}, resultado cacheado")
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Crear clave de caché simple
            cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Verificar caché
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit para {func.__name__}")
                    return result
                else:
                    del cache[cache_key]
            
            # Ejecutar función y cachear
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cache miss para {func.__name__}, resultado cacheado")
            return result
        
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

