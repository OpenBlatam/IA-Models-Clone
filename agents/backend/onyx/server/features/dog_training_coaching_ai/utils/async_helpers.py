"""
Async Helpers
==============
Utilidades para programación asíncrona.
"""

import asyncio
from typing import List, Callable, Any, Coroutine
from functools import wraps


async def run_in_parallel(
    coroutines: List[Coroutine],
    max_concurrent: int = 10
) -> List[Any]:
    """
    Ejecutar múltiples coroutines en paralelo con límite de concurrencia.
    
    Args:
        coroutines: Lista de coroutines a ejecutar
        max_concurrent: Número máximo de coroutines concurrentes
        
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(coro: Coroutine):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[run_with_semaphore(coro) for coro in coroutines])


async def timeout_after(seconds: float, coro: Coroutine) -> Any:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        seconds: Segundos de timeout
        coro: Coroutine a ejecutar
        
    Returns:
        Resultado de la coroutine
        
    Raises:
        asyncio.TimeoutError: Si se excede el timeout
    """
    return await asyncio.wait_for(coro, timeout=seconds)


def async_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorador para reintentar funciones async.
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos en segundos
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


async def batch_process_async(
    items: List[Any],
    func: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Procesar items en batches de forma asíncrona.
    
    Args:
        items: Lista de items a procesar
        func: Función async a aplicar
        batch_size: Tamaño de cada batch
        max_concurrent: Número máximo de batches concurrentes
        
    Returns:
        Lista de resultados
    """
    results = []
    
    # Dividir en batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Procesar batches con límite de concurrencia
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(batch: List[Any]):
        async with semaphore:
            return await asyncio.gather(*[func(item) for item in batch])
    
    batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Aplanar resultados
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results

