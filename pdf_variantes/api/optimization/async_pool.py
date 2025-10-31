"""
Async Pool Optimization
Connection pooling and async optimizations
"""

import asyncio
from typing import Any, Callable, List, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools


class AsyncPool:
    """Pool for async operations"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
    
    async def execute(self, coro: Coroutine) -> Any:
        """Execute coroutine with semaphore limit"""
        async with self._semaphore:
            return await coro


class ParallelExecutor:
    """Execute multiple operations in parallel"""
    
    @staticmethod
    async def execute_parallel(
        coros: List[Coroutine],
        limit: Optional[int] = None
    ) -> List[Any]:
        """Execute coroutines in parallel with optional limit"""
        if limit:
            semaphore = asyncio.Semaphore(limit)
            
            async def execute_with_limit(coro):
                async with semaphore:
                    return await coro
            
            coros = [execute_with_limit(coro) for coro in coros]
        
        return await asyncio.gather(*coros, return_exceptions=True)
    
    @staticmethod
    async def execute_batch(
        items: List[Any],
        func: Callable,
        batch_size: int = 100,
        parallel: bool = True
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if parallel:
                batch_results = await asyncio.gather(*[
                    func(item) for item in batch
                ])
            else:
                batch_results = []
                for item in batch:
                    result = await func(item)
                    batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


def async_parallel(max_concurrent: int = 10):
    """Decorator to execute function in parallel when called multiple times"""
    pending_calls = {}
    lock = asyncio.Lock()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            async with lock:
                if key in pending_calls:
                    # Return existing future
                    return await pending_calls[key]
                
                # Create new future
                future = asyncio.create_task(func(*args, **kwargs))
                pending_calls[key] = future
                
                try:
                    result = await future
                    return result
                finally:
                    # Clean up
                    if key in pending_calls:
                        del pending_calls[key]
        
        return wrapper
    return decorator

