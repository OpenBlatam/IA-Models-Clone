"""
MCP Retry - Lógica de reintentos con exponential backoff
========================================================

Implementa estrategias de reintento con exponential backoff y jitter
para operaciones que pueden fallar temporalmente.
"""

import asyncio
import logging
import random
from typing import Callable, Any, Optional, TypeVar, Tuple, Type
from functools import wraps
from datetime import datetime, timedelta

from .exceptions import MCPError

logger = logging.getLogger(__name__)
T = TypeVar('T')


class RetryConfig:
    """
    Configuración de reintentos.
    
    Define los parámetros para la estrategia de reintento:
    - Número máximo de intentos
    - Delays con exponential backoff
    - Jitter para evitar thundering herd
    - Excepciones que se pueden reintentar
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        """
        Inicializar configuración de reintentos.
        
        Args:
            max_attempts: Número máximo de intentos (default: 3).
            initial_delay: Delay inicial en segundos (default: 1.0).
            max_delay: Delay máximo en segundos (default: 60.0).
            exponential_base: Base para exponential backoff (default: 2.0).
            jitter: Agregar jitter aleatorio para evitar sincronización (default: True).
            retryable_exceptions: Tupla de excepciones que se pueden reintentar
                (default: todas las excepciones).
        
        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if initial_delay < 0:
            raise ValueError("initial_delay must be non-negative")
        if max_delay < initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if exponential_base <= 0:
            raise ValueError("exponential_base must be positive")
        
        self.max_attempts: int = max_attempts
        self.initial_delay: float = initial_delay
        self.max_delay: float = max_delay
        self.exponential_base: float = exponential_base
        self.jitter: bool = jitter
        self.retryable_exceptions: Tuple[Type[Exception], ...] = (
            retryable_exceptions or (Exception,)
        )


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any
) -> Any:
    """
    Ejecuta función con reintentos y exponential backoff.
    
    Args:
        func: Función a ejecutar (puede ser sync o async).
        *args: Argumentos posicionales para la función.
        config: Configuración de reintentos. Si es None, usa valores por defecto.
        **kwargs: Argumentos nombrados para la función.
    
    Returns:
        Resultado de la función si tiene éxito.
    
    Raises:
        Exception: Última excepción si todos los intentos fallan.
        ValueError: Si config es inválido.
    """
    if config is None:
        config = RetryConfig()
    
    if not callable(func):
        raise ValueError("func must be callable")
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt >= config.max_attempts:
                logger.warning(
                    f"Max retry attempts ({config.max_attempts}) reached for "
                    f"{getattr(func, '__name__', 'unknown')}: {e}"
                )
                raise
            
            # Calcular delay con exponential backoff
            delay = min(
                config.initial_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay
            )
            
            # Agregar jitter si está habilitado
            if config.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            func_name = getattr(func, '__name__', 'unknown')
            logger.info(
                f"Retry attempt {attempt}/{config.max_attempts} for {func_name} "
                f"after {delay:.2f}s delay (error: {type(e).__name__})"
            )
            
            await asyncio.sleep(delay)
    
    # Si llegamos aquí sin retornar, lanzar última excepción
    if last_exception:
        raise last_exception
    
    # Esto no debería ocurrir, pero por seguridad
    raise RuntimeError("Retry logic completed without result or exception")


def retryable(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorador para hacer funciones retryable.
    
    Args:
        max_attempts: Número máximo de intentos (default: 3).
        initial_delay: Delay inicial en segundos (default: 1.0).
        max_delay: Delay máximo en segundos (default: 60.0).
        exponential_base: Base para exponential backoff (default: 2.0).
        jitter: Agregar jitter aleatorio (default: True).
    
    Returns:
        Decorador que envuelve la función con lógica de reintento.
    
    Example:
        @retryable(max_attempts=5, initial_delay=2.0)
        async def my_function():
            # código que puede fallar
            pass
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
            )
            return await retry_with_backoff(func, *args, config=config, **kwargs)
        return wrapper
    return decorator
