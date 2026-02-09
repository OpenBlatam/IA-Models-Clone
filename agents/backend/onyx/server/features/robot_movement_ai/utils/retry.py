"""
Retry Utilities - Utilidades de reintento
==========================================

Utilidades para reintentos con diferentes estrategias.
"""

import asyncio
import time
from typing import Callable, TypeVar, Optional, List, Type, Union
from functools import wraps
from enum import Enum
import logging

from ..core.exceptions import BaseRobotException

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Estrategias de reintento."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


class RetryConfig:
    """Configuración de reintento."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None
    ):
        """
        Inicializar configuración.
        
        Args:
            max_attempts: Número máximo de intentos
            initial_delay: Delay inicial en segundos
            max_delay: Delay máximo en segundos
            multiplier: Multiplicador para exponential backoff
            strategy: Estrategia de reintento
            retryable_exceptions: Lista de excepciones que deben reintentarse
            on_retry: Callback llamado en cada reintento
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.on_retry = on_retry
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calcular delay para un intento.
        
        Args:
            attempt: Número de intento (1-indexed)
        
        Returns:
            Delay en segundos
        """
        if self.strategy == RetryStrategy.FIXED:
            return min(self.initial_delay, self.max_delay)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.multiplier ** (attempt - 1))
            return min(delay, self.max_delay)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * attempt
            return min(delay, self.max_delay)
        else:
            return self.initial_delay


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorador para reintentos automáticos.
    
    Args:
        max_attempts: Número máximo de intentos
        initial_delay: Delay inicial en segundos
        max_delay: Delay máximo en segundos
        multiplier: Multiplicador para exponential backoff
        strategy: Estrategia de reintento
        retryable_exceptions: Lista de excepciones que deben reintentarse
        on_retry: Callback llamado en cada reintento
    
    Example:
        @retry(max_attempts=5, initial_delay=2.0)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exception = None
                
                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                            raise
                        
                        if attempt >= config.max_attempts:
                            logger.error(
                                f"{func.__name__} failed after {attempt} attempts",
                                exc_info=e
                            )
                            raise
                        
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        
                        if config.on_retry:
                            try:
                                if asyncio.iscoroutinefunction(config.on_retry):
                                    await config.on_retry(attempt, e)
                                else:
                                    config.on_retry(attempt, e)
                            except Exception as callback_error:
                                logger.error(f"on_retry callback failed: {callback_error}")
                        
                        await asyncio.sleep(delay)
                
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry logic error")
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_exception = None
                
                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                            raise
                        
                        if attempt >= config.max_attempts:
                            logger.error(
                                f"{func.__name__} failed after {attempt} attempts",
                                exc_info=e
                            )
                            raise
                        
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        
                        if config.on_retry:
                            try:
                                config.on_retry(attempt, e)
                            except Exception as callback_error:
                                logger.error(f"on_retry callback failed: {callback_error}")
                        
                        time.sleep(delay)
                
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry logic error")
            
            return sync_wrapper
    
    return decorator


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> T:
    """
    Ejecutar función async con reintentos.
    
    Args:
        func: Función async a ejecutar
        *args: Argumentos posicionales
        config: Configuración de reintento
        **kwargs: Argumentos nombrados
    
    Returns:
        Resultado de la función
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                raise
            
            if attempt >= config.max_attempts:
                raise
            
            delay = config.calculate_delay(attempt)
            await asyncio.sleep(delay)
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")

