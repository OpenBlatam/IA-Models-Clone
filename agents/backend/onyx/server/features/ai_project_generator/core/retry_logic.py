"""
Retry Logic - Lógica de reintentos con backoff exponencial
==========================================================

Implementa lógica de reintentos con backoff exponencial
para comunicación resiliente entre servicios.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple, List
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class BackoffStrategy(str, Enum):
    """Estrategias de backoff"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class RetryConfig:
    """Configuración de retry"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calcula el delay para un intento"""
        if self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay * attempt
        else:  # FIXED
            delay = self.initial_delay
        
        # Aplicar max delay
        delay = min(delay, self.max_delay)
        
        # Aplicar jitter si está habilitado
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # No permitir delays negativos
        
        return delay


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator para agregar lógica de retry a una función.
    
    Args:
        max_attempts: Máximo número de intentos
        backoff_factor: Factor de backoff exponencial
        initial_delay: Delay inicial en segundos
        max_delay: Delay máximo en segundos
        strategy: Estrategia de backoff
        retryable_exceptions: Tupla de excepciones que deben ser reintentadas
        jitter: Agregar jitter al delay
        on_retry: Callback llamado en cada retry
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        jitter=jitter,
    )
    
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except config.retryable_exceptions as e:
                        last_exception = e
                        
                        if attempt < config.max_attempts:
                            delay = config.calculate_delay(attempt)
                            logger.warning(
                                f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                                f"after {delay:.2f}s. Error: {e}"
                            )
                            
                            if on_retry:
                                try:
                                    if asyncio.iscoroutinefunction(on_retry):
                                        await on_retry(e, attempt)
                                    else:
                                        on_retry(e, attempt)
                                except Exception as callback_error:
                                    logger.error(f"Error in retry callback: {callback_error}")
                            
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                f"All {config.max_attempts} attempts failed for {func.__name__}"
                            )
                            raise
                
                # No debería llegar aquí, pero por seguridad
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry logic failed without exception")
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except config.retryable_exceptions as e:
                        last_exception = e
                        
                        if attempt < config.max_attempts:
                            delay = config.calculate_delay(attempt)
                            logger.warning(
                                f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                                f"after {delay:.2f}s. Error: {e}"
                            )
                            
                            if on_retry:
                                try:
                                    on_retry(e, attempt)
                                except Exception as callback_error:
                                    logger.error(f"Error in retry callback: {callback_error}")
                            
                            time.sleep(delay)
                        else:
                            logger.error(
                                f"All {config.max_attempts} attempts failed for {func.__name__}"
                            )
                            raise
                
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry logic failed without exception")
            
            return sync_wrapper
    
    return decorator


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Ejecuta función async con retry.
    
    Args:
        func: Función async a ejecutar
        *args: Argumentos para la función
        config: Configuración de retry (opcional)
        **kwargs: Keyword arguments para la función
    
    Returns:
        Resultado de la función
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt}/{config.max_attempts} "
                    f"after {delay:.2f}s. Error: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed")
                raise
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without exception")


def retry_sync(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Ejecuta función sync con retry.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos para la función
        config: Configuración de retry (opcional)
        **kwargs: Keyword arguments para la función
    
    Returns:
        Resultado de la función
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt}/{config.max_attempts} "
                    f"after {delay:.2f}s. Error: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed")
                raise
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without exception")















