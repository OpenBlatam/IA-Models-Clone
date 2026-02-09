"""
Performance optimization utilities.
"""

import asyncio
import time
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def async_timed(func: Callable) -> Callable:
    """
    Decorator to measure execution time of async functions.
    
    Args:
        func: Async function to measure
    
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
    
    return wrapper


def sync_timed(func: Callable) -> Callable:
    """
    Decorator to measure execution time of sync functions.
    
    Args:
        func: Sync function to measure
    
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
    
    return wrapper


class AsyncBatchProcessor:
    """
    Process items in batches asynchronously for better performance.
    """
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
    
    async def process(
        self,
        items: list,
        processor: Callable[[Any], Any]
    ) -> list:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
        
        Returns:
            List of processed results
        """
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await asyncio.gather(*[processor(item) for item in batch])
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        
        return results






