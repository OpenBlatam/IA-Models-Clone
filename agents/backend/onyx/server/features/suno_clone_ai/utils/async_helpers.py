"""
Utilidades async para operaciones comunes
"""

import asyncio
from typing import List, Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')


async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Ejecuta una función síncrona en un executor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def batch_process(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """Procesa items en lotes con límite de concurrencia"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Procesar batch con límite de concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item: T) -> Any:
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(item)
                return await run_in_executor(processor, item)
        
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch]
        )
        results.extend(batch_results)
    
    return results


def to_async(func: Callable) -> Callable:
    """Convierte una función síncrona a async usando executor"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_executor(func, *args, **kwargs)
    return wrapper

