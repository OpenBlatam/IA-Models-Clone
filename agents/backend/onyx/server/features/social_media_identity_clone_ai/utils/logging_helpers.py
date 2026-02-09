"""
Helper functions for standardized logging patterns.
Eliminates repetitive logging code throughout the codebase.
"""

import logging
import time
from typing import Any, Optional, Dict
from functools import wraps
from contextlib import contextmanager


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger con configuración estándar.
    
    Args:
        name: Nombre del logger (típicamente __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


@contextmanager
def log_operation(logger: logging.Logger, operation: str, **context):
    """
    Context manager para logging de operaciones con timing.
    
    Args:
        logger: Logger a usar
        operation: Nombre de la operación
        **context: Contexto adicional para el log
        
    Usage:
        with log_operation(logger, "extract_profile", username="user123"):
            # código de la operación
    """
    start_time = time.time()
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.info(f"Starting {operation}" + (f" | {context_str}" if context_str else ""))
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation} in {duration:.3f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation} after {duration:.3f}s: {e}",
            exc_info=True
        )
        raise


def log_function_call(logger: logging.Logger):
    """
    Decorador para logging automático de llamadas a funciones.
    
    Args:
        logger: Logger a usar
        
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            # código
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}", exc_info=True)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}", exc_info=True)
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_error(logger: logging.Logger, error: Exception, operation: str, **context):
    """
    Helper para logging consistente de errores.
    
    Args:
        logger: Logger a usar
        error: Excepción que ocurrió
        operation: Nombre de la operación
        **context: Contexto adicional
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(
        f"Error in {operation}" + (f" | {context_str}" if context_str else ""),
        exc_info=True
    )


def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics):
    """
    Helper para logging de métricas de rendimiento.
    
    Args:
        logger: Logger a usar
        operation: Nombre de la operación
        duration: Duración en segundos
        **metrics: Métricas adicionales
    """
    metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(
        f"Performance: {operation} took {duration:.3f}s" + 
        (f" | {metrics_str}" if metrics_str else "")
    )


def log_cache_hit(logger: logging.Logger, cache_key: str, source: str = "cache"):
    """
    Helper para logging de cache hits.
    
    Args:
        logger: Logger a usar
        cache_key: Clave de caché
        source: Fuente del caché (default: "cache")
    """
    logger.debug(f"Cache hit: {cache_key} from {source}")


def log_cache_miss(logger: logging.Logger, cache_key: str):
    """
    Helper para logging de cache misses.
    
    Args:
        logger: Logger a usar
        cache_key: Clave de caché
    """
    logger.debug(f"Cache miss: {cache_key}")








