"""
Decorators Module
=================

Decoradores útiles para logging, timing, caching, etc.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Any, Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


def log_execution(func: Callable) -> Callable:
    """
    Decorador para loguear ejecución de funciones.
    
    Registra entrada, salida y tiempo de ejecución.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Exiting {func_name} (took {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func_name} after {elapsed:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Exiting {func_name} (took {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func_name} after {elapsed:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def time_execution(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución.
    
    Retorna el resultado junto con el tiempo de ejecución.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result, elapsed
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result, elapsed
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def cache_result(ttl: Optional[float] = None, max_size: int = 128):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        ttl: Time to live en segundos (None = sin expiración)
        max_size: Tamaño máximo del cache
    """
    cache: Dict[str, tuple] = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            # Verificar cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            if len(cache) >= max_size:
                # Eliminar entrada más antigua
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            # Verificar cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en cache
            if len(cache) >= max_size:
                # Eliminar entrada más antigua
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorador para limitar tasa de llamadas.
    
    Args:
        calls_per_second: Número máximo de llamadas por segundo
    """
    min_interval = 1.0 / calls_per_second
    last_called: Dict[str, float] = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_id = id(func)
            now = time.time()
            
            if func_id in last_called:
                elapsed = now - last_called[func_id]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    logger.debug(f"Rate limiting {func.__name__}, waiting {wait_time:.3f}s")
                    await asyncio.sleep(wait_time)
            
            last_called[func_id] = time.time()
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_id = id(func)
            now = time.time()
            
            if func_id in last_called:
                elapsed = now - last_called[func_id]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    logger.debug(f"Rate limiting {func.__name__}, waiting {wait_time:.3f}s")
                    time.sleep(wait_time)
            
            last_called[func_id] = time.time()
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_input(**validators: Callable):
    """
    Decorador para validar argumentos de entrada.
    
    Args:
        **validators: Funciones de validación por nombre de argumento
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Obtener nombres de argumentos
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validar
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Obtener nombres de argumentos
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validar
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
