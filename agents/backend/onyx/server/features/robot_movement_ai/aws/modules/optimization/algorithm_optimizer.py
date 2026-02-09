"""
Algorithm Optimizer
==================

Optimize algorithms for maximum performance.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


class AlgorithmOptimizer:
    """Algorithm optimizer for performance."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._memoized: Dict[str, Callable] = {}
    
    def memoize(self, func: Callable, maxsize: int = 128):
        """Memoize function for faster execution."""
        cached_func = lru_cache(maxsize=maxsize)(func)
        self._memoized[func.__name__] = cached_func
        logger.debug(f"Memoized function: {func.__name__}")
        return cached_func
    
    def time_function(self, func: Callable, *args, **kwargs):
        """Time function execution."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    
    async def time_async_function(self, func: Callable, *args, **kwargs):
        """Time async function execution."""
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    
    def optimize_sort(self, data: List[Any], key: Optional[Callable] = None) -> List[Any]:
        """Optimize sorting based on data size."""
        # Use Timsort (Python's default) for most cases
        # For very large datasets, consider external sorting
        return sorted(data, key=key)
    
    def optimize_search(self, data: List[Any], value: Any) -> Optional[int]:
        """Optimize search based on data characteristics."""
        # Use binary search for sorted data
        # Use hash lookup for hashable data
        try:
            return data.index(value)
        except ValueError:
            return None
    
    async def batch_process(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 100,
        parallel: bool = True
    ) -> List[Any]:
        """Process items in optimized batches."""
        if not parallel:
            # Sequential processing
            results = []
            for item in items:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(item)
                else:
                    result = processor(item)
                results.append(result)
            return results
        
        # Parallel processing
        import asyncio
        
        async def process_batch(batch: List[Any]):
            tasks = []
            for item in batch:
                if asyncio.iscoroutinefunction(processor):
                    tasks.append(processor(item))
                else:
                    tasks.append(asyncio.to_thread(processor, item))
            return await asyncio.gather(*tasks)
        
        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "memoized_functions": len(self._memoized),
            "cached_items": len(self._cache)
        }

