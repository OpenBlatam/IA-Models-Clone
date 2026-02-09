"""
Decorator Utils - Utilidades de Decoradores
===========================================

Utilidades y decoradores reutilizables comunes.
"""

import logging
import functools
import asyncio
import time
from typing import Any, Callable, Optional, Dict

logger = logging.getLogger(__name__)


def memoize(maxsize: Optional[int] = None, ttl: Optional[float] = None):
    """
    Decorador para memoización con TTL opcional.
    
    Args:
        maxsize: Tamaño máximo de caché
        ttl: Time to live en segundos
        
    Returns:
        Decorador
    """
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generar key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Verificar TTL
            if ttl and key in cache_times:
                if time.time() - cache_times[key] > ttl:
                    cache.pop(key, None)
                    cache_times.pop(key, None)
            
            # Verificar caché
            if key in cache:
                return cache[key]
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache[key] = result
            if ttl:
                cache_times[key] = time.time()
            
            # Limpiar si excede maxsize
            if maxsize and len(cache) > maxsize:
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                cache.pop(oldest_key, None)
                cache_times.pop(oldest_key, None)
            
            return result
        
        wrapper.cache_clear = lambda: (cache.clear(), cache_times.clear())
        return wrapper
    
    return decorator


def singleton(cls):
    """
    Decorador para patrón Singleton.
    
    Args:
        cls: Clase a convertir en singleton
        
    Returns:
        Clase singleton
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def deprecated(reason: Optional[str] = None):
    """
    Decorador para marcar funciones como deprecadas.
    
    Args:
        reason: Razón de deprecación
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            logger.warning(f"⚠️ {message}")
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def rate_limit(calls: int, period: float):
    """
    Decorador para rate limiting simple.
    
    Args:
        calls: Número de llamadas permitidas
        period: Período en segundos
        
    Returns:
        Decorador
    """
    calls_history = []
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Limpiar llamadas antiguas
            calls_history[:] = [t for t in calls_history if now - t < period]
            
            # Verificar límite
            if len(calls_history) >= calls:
                raise Exception(f"Rate limit exceeded: {calls} calls per {period}s")
            
            # Registrar llamada
            calls_history.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def validate_args(**validators):
    """
    Decorador para validar argumentos.
    
    Args:
        **validators: Validadores por nombre de argumento
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener nombres de argumentos
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validar cada argumento
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(f"Invalid argument {arg_name}: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_calls(log_args: bool = False, log_result: bool = False):
    """
    Decorador para logging de llamadas.
    
    Args:
        log_args: Si loggear argumentos
        log_result: Si loggear resultado
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"📞 Calling {func.__name__}")
            
            if log_args:
                logger.debug(f"  Args: {args}")
                logger.debug(f"  Kwargs: {kwargs}")
            
            result = func(*args, **kwargs)
            
            if log_result:
                logger.debug(f"  Result: {result}")
            
            return result
        
        return wrapper
    
    return decorator


def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorador para reintentos en caso de excepción.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial
        backoff: Factor de backoff
        exceptions: Tipos de excepciones a capturar
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}, "
                            f"retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator


def async_retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorador async para reintentos en caso de excepción.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial
        backoff: Factor de backoff
        exceptions: Tipos de excepciones a capturar
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}, "
                            f"retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorador para timeout (sync).
    
    Args:
        seconds: Segundos de timeout
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    
    return decorator


def async_timeout(seconds: float):
    """
    Decorador async para timeout.
    
    Args:
        seconds: Segundos de timeout
        
    Returns:
        Decorador
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                async with asyncio.timeout(seconds):
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
        
        return wrapper
    
    return decorator




