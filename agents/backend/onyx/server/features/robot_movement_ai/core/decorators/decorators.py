"""
Decorators
==========

Decoradores útiles para el sistema de movimiento robótico.
"""

import functools
import time
import logging
from typing import Callable, Any
import traceback

from ..exceptions import RobotMovementError

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """
    Decorador para registrar tiempo de ejecución.
    
    Usage:
        @log_execution_time
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{func.__name__} executed in {execution_time:.4f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.4f}s: {e}"
            )
            raise
    
    return wrapper


def log_execution_time_async(func: Callable) -> Callable:
    """
    Decorador para registrar tiempo de ejecución (async).
    
    Usage:
        @log_execution_time_async
        async def my_function():
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{func.__name__} executed in {execution_time:.4f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.4f}s: {e}"
            )
            raise
    
    return wrapper


def handle_robot_errors(func: Callable) -> Callable:
    """
    Decorador para manejar errores del robot de forma consistente.
    
    Usage:
        @handle_robot_errors
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RobotMovementError:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RobotMovementError(f"Error in {func.__name__}: {e}") from e
    
    return wrapper


def handle_robot_errors_async(func: Callable) -> Callable:
    """
    Decorador para manejar errores del robot de forma consistente (async).
    
    Usage:
        @handle_robot_errors_async
        async def my_function():
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RobotMovementError:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RobotMovementError(f"Error in {func.__name__}: {e}") from e
    
    return wrapper


def validate_inputs(**validators):
    """
    Decorador para validar inputs de función.
    
    Args:
        **validators: Diccionario de {param_name: validator_function}
    
    Usage:
        @validate_inputs(position=validate_position, orientation=validate_orientation)
        def my_function(position, orientation):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener nombres de parámetros
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validar cada parámetro especificado
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    validator(value)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """
    Decorador para reintentar en caso de fallo.
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos (segundos)
    
    Usage:
        @retry_on_failure(max_retries=3, delay=0.1)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
            raise last_exception
        
        return wrapper
    return decorator


def retry_on_failure_async(max_retries: int = 3, delay: float = 0.1):
    """
    Decorador para reintentar en caso de fallo (async).
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos (segundos)
    
    Usage:
        @retry_on_failure_async(max_retries=3, delay=0.1)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
            raise last_exception
        
        return wrapper
    return decorator


def cache_result(cache_size: int = 100):
    """
    Decorador para cachear resultados de función.
    
    Args:
        cache_size: Tamaño máximo del caché
    
    Usage:
        @cache_result(cache_size=100)
        def expensive_function(x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave de caché
            import hashlib
            import json
            key_data = {
                "args": args,
                "kwargs": kwargs
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Verificar caché
            if cache_key in cache:
                logger.debug(f"{func.__name__} cache hit")
                return cache[cache_key]
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            if len(cache) >= cache_size:
                # Eliminar más antiguo
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]
            
            cache[cache_key] = result
            cache_order.append(cache_key)
            
            logger.debug(f"{func.__name__} cache miss, result cached")
            return result
        
        return wrapper
    return decorator






