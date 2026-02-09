"""
Helper functions for async operations.
Eliminates repetitive async patterns and error handling.
"""

from typing import TypeVar, List, Callable, Optional, Any, Iterable
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


async def safe_gather(
    *coros,
    return_exceptions: bool = True,
    default: Optional[Any] = None
) -> List[Any]:
    """
    Ejecuta coroutines en paralelo de forma segura.
    
    Args:
        *coros: Coroutines a ejecutar
        return_exceptions: Si retornar excepciones en lugar de lanzarlas
        default: Valor por defecto para errores (opcional)
        
    Returns:
        Lista de resultados
        
    Usage:
        >>> results = await safe_gather(
        ...     extract_tiktok_profile(username1),
        ...     extract_instagram_profile(username2)
        ... )
    """
    results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
    
    if default is not None:
        return [
            default if isinstance(r, Exception) else r
            for r in results
        ]
    
    return results


async def safe_map_async(
    items: Iterable[T],
    func: Callable[[T], Any],
    max_concurrent: Optional[int] = None,
    skip_errors: bool = True,
    operation: Optional[str] = None
) -> List[Any]:
    """
    Aplica una función async a cada item de forma segura y concurrente.
    
    Args:
        items: Iterable de items
        func: Función async a aplicar
        max_concurrent: Máximo de operaciones concurrentes (opcional)
        skip_errors: Si saltar items con error
        operation: Nombre de la operación para logging
        
    Returns:
        Lista de resultados
        
    Usage:
        >>> profiles = await safe_map_async(
        ...     usernames,
        ...     extract_tiktok_profile,
        ...     max_concurrent=5
        ... )
    """
    if max_concurrent:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_func(item):
            async with semaphore:
                try:
                    return await func(item)
                except Exception as e:
                    op_name = operation or func.__name__
                    if skip_errors:
                        logger.warning(f"Error in {op_name} for item: {e}")
                        return None
                    raise
        
        tasks = [bounded_func(item) for item in items]
    else:
        tasks = [func(item) for item in items]
    
    results = await safe_gather(*tasks, return_exceptions=skip_errors)
    return [r for r in results if r is not None]


async def retry_async(
    func: Callable[..., Any],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    Ejecuta una función async con retry automático.
    
    Args:
        func: Función async a ejecutar
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff exponencial
        *args: Argumentos para la función
        **kwargs: Keyword arguments para la función
        
    Returns:
        Resultado de la función
        
    Raises:
        Exception: Si falla después de todos los intentos
        
    Usage:
        >>> result = await retry_async(
        ...     fetch_data,
        ...     max_attempts=5,
        ...     url="https://api.example.com"
        ... )
    """
    current_delay = delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {current_delay}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"{func.__name__} failed after {max_attempts} attempts")
    
    raise last_exception


async def timeout_async(
    coro: Any,
    timeout: float,
    default: Optional[Any] = None
) -> Any:
    """
    Ejecuta una coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        default: Valor por defecto si hay timeout
        
    Returns:
        Resultado de la coroutine o default
        
    Usage:
        >>> result = await timeout_async(
        ...     slow_operation(),
        ...     timeout=5.0,
        ...     default={}
        ... )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default








