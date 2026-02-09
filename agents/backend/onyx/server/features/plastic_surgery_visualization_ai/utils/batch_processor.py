"""Batch processing utilities."""

from typing import List, Callable, Any, TypeVar, Optional, Iterator
import asyncio
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchProcessor:
    """Utility for processing items in batches."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        stop_on_error: bool = False
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent operations
            stop_on_error: Stop processing on first error
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error
    
    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_item(item: T, index: int) -> Any:
            """Process a single item."""
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = processor(item)
                    
                    if progress_callback:
                        progress_callback(index + 1, len(items))
                    
                    return {"success": True, "result": result, "index": index}
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    if self.stop_on_error:
                        raise
                    return {"success": False, "error": str(e), "index": index}
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_tasks = [
                process_item(item, i + j)
                for j, item in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i // self.batch_size + 1}")
        
        return results


# Additional batch utilities from batch_utils.py
def batch(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Split items into batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


async def batch_process_async(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 10,
    concurrency: int = 5
) -> List[R]:
    """Process items in batches asynchronously."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def process_item(item: T) -> R:
        async with semaphore:
            if asyncio.iscoroutinefunction(processor):
                return await processor(item)
            return processor(item)
    
    batches = list(batch(items, batch_size))
    for batch_items in batches:
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch_items],
            return_exceptions=True
        )
        results.extend(batch_results)
    
    return results


class BatchProcessor:
    """Utility for processing items in batches (alias for backward compatibility)."""
    pass
    
    async def process_with_retry(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        max_retries: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Process items with retry logic.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            max_retries: Maximum number of retries
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        from utils.retry import exponential_backoff
        
        async def process_with_retry_item(item: T, index: int) -> Any:
            """Process item with retry."""
            try:
                result = await exponential_backoff(
                    lambda: processor(item) if asyncio.iscoroutinefunction(processor) else processor(item),
                    max_attempts=max_retries
                )
                
                if progress_callback:
                    progress_callback(index + 1, len(items))
                
                return {"success": True, "result": result, "index": index}
            except Exception as e:
                logger.error(f"Error processing item {index} after {max_retries} retries: {e}")
                return {"success": False, "error": str(e), "index": index}
        
        tasks = [process_with_retry_item(item, i) for i, item in enumerate(items)]
        return await asyncio.gather(*tasks, return_exceptions=True)

