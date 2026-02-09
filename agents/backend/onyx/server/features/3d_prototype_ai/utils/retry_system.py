"""
Retry System - Sistema de retry con exponential backoff
========================================================
"""

import logging
import asyncio
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import random

logger = logging.getLogger(__name__)


class RetrySystem:
    """Sistema de retry con exponential backoff"""
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Any:
        """
        Reintenta una función async con exponential backoff
        
        Args:
            func: Función async a ejecutar
            max_retries: Número máximo de reintentos
            initial_delay: Delay inicial en segundos
            max_delay: Delay máximo en segundos
            exponential_base: Base para exponential backoff
            jitter: Si agregar jitter aleatorio
            exceptions: Tipos de excepciones que deben causar retry
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calcular delay
                    delay = initial_delay * (exponential_base ** attempt)
                    delay = min(delay, max_delay)
                    
                    # Agregar jitter si está habilitado
                    if jitter:
                        jitter_amount = delay * 0.1 * random.random()
                        delay += jitter_amount
                    
                    logger.warning(
                        f"Intento {attempt + 1}/{max_retries + 1} falló. "
                        f"Reintentando en {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Todos los reintentos fallaron después de {max_retries + 1} intentos")
                    raise last_exception
        
        raise last_exception
    
    @staticmethod
    def retry_sync(
        func: Callable,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Any:
        """Reintenta una función sync con exponential backoff"""
        import time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = initial_delay * (exponential_base ** attempt)
                    delay = min(delay, max_delay)
                    
                    if jitter:
                        jitter_amount = delay * 0.1 * random.random()
                        delay += jitter_amount
                    
                    logger.warning(
                        f"Intento {attempt + 1}/{max_retries + 1} falló. "
                        f"Reintentando en {delay:.2f}s: {e}"
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(f"Todos los reintentos fallaron")
                    raise last_exception
        
        raise last_exception


def retry_decorator(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorador para retry automático"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def _func():
                return await func(*args, **kwargs)
            
            return await RetrySystem.retry_async(
                _func,
                max_retries=max_retries,
                initial_delay=initial_delay
            )
        return wrapper
    return decorator




