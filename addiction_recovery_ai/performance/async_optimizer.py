"""
Async Performance Optimizer
Advanced async optimizations for maximum throughput
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, List
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import uvloop

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """
    Async performance optimizer
    
    Features:
    - Batch processing
    - Parallel execution
    - Connection pooling optimization
    - Event loop optimization
    """
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._use_uvloop = False
    
    def enable_uvloop(self) -> None:
        """Enable uvloop for better performance"""
        try:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            self._use_uvloop = True
            logger.info("uvloop enabled")
        except Exception as e:
            logger.warning(f"Failed to enable uvloop: {str(e)}")
    
    async def batch_execute(
        self,
        tasks: List[Coroutine],
        max_concurrent: int = 10
    ) -> List[Any]:
        """
        Execute tasks in batches with concurrency limit
        
        Args:
            tasks: List of coroutines to execute
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task: Coroutine) -> Any:
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[execute_with_semaphore(task) for task in tasks])
    
    async def parallel_map(
        self,
        func: Callable,
        items: List[Any],
        max_workers: int = 10
    ) -> List[Any]:
        """
        Parallel map with async function
        
        Args:
            func: Async function to apply
            items: Items to process
            max_workers: Maximum concurrent workers
            
        Returns:
            List of results
        """
        tasks = [func(item) for item in items]
        return await self.batch_execute(tasks, max_concurrent=max_workers)
    
    def run_in_executor(self, func: Callable, *args) -> Coroutine:
        """Run blocking function in executor"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._executor, func, *args)


def async_cache(ttl: int = 300):
    """Async cache decorator with TTL"""
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import time
            import hashlib
            import json
            
            # Generate cache key
            key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                if time.time() - cache_times[cache_key] < ttl:
                    return cache[cache_key]
                else:
                    # Expired
                    del cache[cache_key]
                    del cache_times[cache_key]
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            cache_times[cache_key] = time.time()
            
            return result
        
        return wrapper
    return decorator


def parallelize(max_workers: int = 10):
    """Decorator to parallelize function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            optimizer = AsyncOptimizer()
            tasks = [func(item, *args, **kwargs) for item in items]
            return await optimizer.batch_execute(tasks, max_concurrent=max_workers)
        return wrapper
    return decorator


# Global optimizer instance
_optimizer: AsyncOptimizer = None


def get_async_optimizer() -> AsyncOptimizer:
    """Get global async optimizer"""
    global _optimizer
    if _optimizer is None:
        _optimizer = AsyncOptimizer()
        _optimizer.enable_uvloop()
    return _optimizer















