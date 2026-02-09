"""Batch processing utilities."""

from typing import List, TypeVar, Callable, Iterator, Optional
import asyncio

T = TypeVar('T')
R = TypeVar('R')


def batch(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """
    Split items into batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


async def batch_process_async(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 10,
    concurrency: int = 5
) -> List[R]:
    """
    Process items in batches asynchronously.
    
    Args:
        items: List of items to process
        processor: Async processing function
        batch_size: Size of each batch
        concurrency: Max concurrent operations
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def process_item(item: T) -> R:
        async with semaphore:
            if asyncio.iscoroutinefunction(processor):
                return await processor(item)
            return processor(item)
    
    batches = list(batch(items, batch_size))
    
    for batch_items in batches:
        tasks = [process_item(item) for item in batch_items]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results


def batch_process(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 10
) -> List[R]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        processor: Processing function
        batch_size: Size of each batch
        
    Returns:
        List of results
    """
    results = []
    for batch_items in batch(items, batch_size):
        batch_results = [processor(item) for item in batch_items]
        results.extend(batch_results)
    return results


class BatchProcessor:
    """Batch processor with progress tracking."""
    
    def __init__(
        self,
        processor: Callable[[T], R],
        batch_size: int = 10,
        concurrency: Optional[int] = None
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.concurrency = concurrency
        self._processed = 0
        self._total = 0
    
    def process(self, items: List[T]) -> List[R]:
        """
        Process items.
        
        Args:
            items: List of items
            
        Returns:
            List of results
        """
        self._total = len(items)
        self._processed = 0
        
        if asyncio.iscoroutinefunction(self.processor):
            return asyncio.run(self._process_async(items))
        return self._process_sync(items)
    
    def _process_sync(self, items: List[T]) -> List[R]:
        """Process synchronously."""
        results = []
        for batch_items in batch(items, self.batch_size):
            batch_results = [self.processor(item) for item in batch_items]
            results.extend(batch_results)
            self._processed += len(batch_items)
        return results
    
    async def _process_async(self, items: List[T]) -> List[R]:
        """Process asynchronously."""
        if self.concurrency:
            return await batch_process_async(
                items,
                self.processor,
                self.batch_size,
                self.concurrency
            )
        
        results = []
        for batch_items in batch(items, self.batch_size):
            tasks = [self.processor(item) for item in batch_items]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            self._processed += len(batch_items)
        return results
    
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self._total == 0:
            return 0.0
        return (self._processed / self._total) * 100



