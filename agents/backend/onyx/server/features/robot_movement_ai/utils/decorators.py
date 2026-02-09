"""
Decorators - Utilidades de decoradores
=======================================

Decoradores útiles para funciones y métodos.
"""

import asyncio
import functools
import time
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorador para memoización simple (sin TTL).
    
    Args:
        func: Función a memoizar
    
    Returns:
        Función memoizada
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


def throttle(seconds: float):
    """
    Decorador para throttling (limitar frecuencia de ejecución).
    
    Args:
        seconds: Segundos entre ejecuciones
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < seconds:
                time.sleep(seconds - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def debounce(seconds: float):
    """
    Decorador para debounce (esperar antes de ejecutar).
    
    Args:
        seconds: Segundos a esperar
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        timer = [None]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            last_called[0] = current_time
            
            def call_func():
                if time.time() - last_called[0] >= seconds:
                    func(*args, **kwargs)
            
            if timer[0] is not None:
                timer[0].cancel()
            
            timer[0] = asyncio.get_event_loop().call_later(seconds, call_func)
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Decorador para clase singleton.
    
    Args:
        cls: Clase
    
    Returns:
        Clase singleton
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def deprecated(reason: Optional[str] = None):
    """
    Decorador para marcar funciones como deprecated.
    
    Args:
        reason: Razón del deprecation
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            logger.warning(message)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorador para reintentar en caso de fallo.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay entre intentos
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise
            if last_exception:
                raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise
            if last_exception:
                raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorador para timeout en funciones.
    
    Args:
        seconds: Segundos de timeout
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=seconds
            ))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_calls(log_args: bool = False, log_result: bool = False):
    """
    Decorador para loggear llamadas a funciones.
    
    Args:
        log_args: Si True, loggea argumentos
        log_result: Si True, loggea resultado
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            if log_args:
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")
            
            result = func(*args, **kwargs)
            
            if log_result:
                logger.debug(f"Result: {result}")
            
            return result
        
        return wrapper
    return decorator


def validate_args(validator: Callable[[tuple, dict], bool]):
    """
    Decorador para validar argumentos.
    
    Args:
        validator: Función validadora (args, kwargs) -> bool
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(args, kwargs):
                raise ValueError(f"Invalid arguments for {func.__name__}")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Decorador para cachear resultado con TTL opcional.
    
    Args:
        ttl: Time to live en segundos (None = sin expiración)
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache:
                if ttl is None or (current_time - cache_times[key]) < ttl:
                    return cache[key]
                else:
                    del cache[key]
                    del cache_times[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time
            return result
        
        return wrapper
    return decorator

