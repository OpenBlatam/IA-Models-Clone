"""
Performance optimization utilities.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class PerformanceOptimizer:
    """
    Performance optimization utilities.
    """
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.default_ttl = timedelta(minutes=5)
    
    @lru_cache(maxsize=128)
    def cached_computation(self, key: str, computation_func: Callable, *args, **kwargs):
        """Cache computation results."""
        return computation_func(*args, **kwargs)
    
    def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[Any]:
        """
        Process items in batches with concurrency control.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Size of each batch
            max_concurrent: Maximum concurrent operations
        
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
        
        return results
    
    async def async_batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[Any]:
        """Async batch processing with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_func(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def optimize_query(self, query: str) -> str:
        """Optimize query string."""
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Truncate if too long
        if len(query) > 2000:
            query = query[:2000]
            logger.warning("Query truncated to 2000 characters")
        
        return query
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "cache_size": len(self.query_cache),
            "cache_entries": list(self.query_cache.keys())[:10]
        }




