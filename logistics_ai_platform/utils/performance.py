"""Performance optimization utilities"""

from typing import List, TypeVar, Callable, Any, Optional
from functools import partial
import asyncio

T = TypeVar('T')


async def batch_process(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """Process items in batches with concurrency control"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item: T) -> Any:
            async with semaphore:
                return await processor(item)
        
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch],
            return_exceptions=True
        )
        
        # Filter out exceptions
        results.extend([
            r for r in batch_results
            if not isinstance(r, Exception)
        ])
    
    return results


async def parallel_execute(
    *coroutines: Callable,
    max_concurrent: int = 10
) -> List[Any]:
    """Execute multiple coroutines in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(coro: Callable) -> Any:
        async with semaphore:
            return await coro()
    
    return await asyncio.gather(
        *[execute_with_semaphore(coro) for coro in coroutines],
        return_exceptions=True
    )


def lazy_load(
    factory: Callable,
    cache_key: Optional[str] = None
) -> Callable:
    """Create a lazy loading function"""
    value = None
    loaded = False
    
    async def loader() -> Any:
        nonlocal value, loaded
        
        if loaded:
            return value
        
        if cache_key:
            from utils.cache import cache_service
            cached = await cache_service.get(cache_key)
            if cached is not None:
                value = cached
                loaded = True
                return value
        
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        loaded = True
        
        if cache_key:
            from utils.cache import cache_service
            await cache_service.set(cache_key, value)
        
        return value
    
    return loader


def memoize_async(max_size: int = 128):
    """
    Memoize async function results with LRU eviction
    
    Args:
        max_size: Maximum cache size (default: 128)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []
        
        async def memoized(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                cache_order.remove(key)
                cache_order.append(key)
                return cache[key]
            
            result = await func(*args, **kwargs)
            
            if len(cache) >= max_size:
                oldest = cache_order.pop(0)
                del cache[oldest]
            
            cache[key] = result
            cache_order.append(key)
            
            return result
        
        return memoized
    return decorator

