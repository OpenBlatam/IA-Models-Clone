"""
Retry Middleware con exponential backoff
Para operaciones resilientes
"""

import logging
import asyncio
from typing import Callable, Any, Optional
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    Decorator para retry con exponential backoff
    
    Args:
        max_attempts: Número máximo de intentos
        initial_wait: Tiempo de espera inicial en segundos
        max_wait: Tiempo máximo de espera en segundos
        exponential_base: Base exponencial para el backoff
        retry_on: Tupla de excepciones para las que se debe hacer retry
        
    Usage:
        @retry_with_backoff(max_attempts=5, initial_wait=2.0)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Determinar si es async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryHandler:
    """
    Handler para retry logic con diferentes estrategias
    """
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 60.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta una función async con retry
        
        Args:
            func: Función async a ejecutar
            max_attempts: Número máximo de intentos
            initial_wait: Tiempo de espera inicial
            max_wait: Tiempo máximo de espera
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
        """
        last_exception = None
        wait_time = initial_wait
        
        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait)
                else:
                    logger.error(f"All {max_attempts} attempts failed")
        
        raise last_exception
    
    @staticmethod
    def retry_sync(
        func: Callable,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 60.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta una función sync con retry
        
        Args:
            func: Función sync a ejecutar
            max_attempts: Número máximo de intentos
            initial_wait: Tiempo de espera inicial
            max_wait: Tiempo máximo de espera
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
        """
        import time
        last_exception = None
        wait_time = initial_wait
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait)
                else:
                    logger.error(f"All {max_attempts} attempts failed")
        
        raise last_exception















