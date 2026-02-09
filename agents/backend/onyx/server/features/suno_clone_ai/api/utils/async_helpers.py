"""
Helpers para operaciones async optimizadas

Incluye utilidades para trabajar con operaciones asíncronas de forma eficiente.
"""

import asyncio
import logging
from typing import List, Callable, Any, TypeVar, Awaitable, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorador para reintentar operaciones async.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff exponencial
        exceptions: Tupla de excepciones a capturar
        
    Example:
        @retry_async(max_attempts=3, delay=1.0)
        async def my_async_function():
            ...
    """
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


async def gather_with_limit(
    tasks: List[Awaitable[T]],
    limit: int = 10
) -> List[T]:
    """
    Ejecuta múltiples tareas async con límite de concurrencia.
    
    Args:
        tasks: Lista de tareas async
        limit: Límite de concurrencia
        
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[bounded_task(task) for task in tasks])


async def timeout_async(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Ejecuta una coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        default: Valor a retornar si hay timeout
        
    Returns:
        Resultado de la coroutine o default si hay timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default

