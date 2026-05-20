"""
Async Operations for Better Performance

Asynchronous operations for non-blocking I/O and parallel processing.
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """
    Async processor for parallel operations.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async processor.
        
        Args:
            max_workers: Maximum number of workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        items: List[Any],
        func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: List of items to process
            func: Function to apply
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches in parallel
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda batch: [func(item) for item in batch],
                batch
            )
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for batch_results in results for item in batch_results]
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


async def async_map(
    func: Callable,
    items: List[Any],
    max_concurrent: int = 10
) -> List[Any]:
    """
    Async map function.
    
    Args:
        func: Function to apply
        items: List of items
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: Any) -> Any:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)


def run_async(func: Callable) -> Callable:
    """
    Decorator to run function asynchronously.
    
    Args:
        func: Function to wrap
        
    Returns:
        Async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper













