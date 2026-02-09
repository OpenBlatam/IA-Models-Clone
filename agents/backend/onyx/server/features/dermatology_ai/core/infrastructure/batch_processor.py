"""
Batch Processing Utilities
Efficient processing of multiple items in batches
"""

import asyncio
from typing import List, TypeVar, Callable, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 10
    max_concurrent: int = 5
    timeout: Optional[float] = None
    stop_on_error: bool = False


class BatchProcessor:
    """Process items in batches with concurrency control"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        batch_size: Optional[int] = None
    ) -> List[R]:
        """
        Process items in batches
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            batch_size: Override default batch size
            
        Returns:
            List of results
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        errors = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} items)")
            
            try:
                batch_results = await self._process_batch_concurrent(
                    batch,
                    processor
                )
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}", exc_info=True)
                if self.config.stop_on_error:
                    raise
                errors.append(e)
        
        if errors and self.config.stop_on_error:
            raise Exception(f"Batch processing failed with {len(errors)} errors")
        
        return results
    
    async def _process_batch_concurrent(
        self,
        batch: List[T],
        processor: Callable[[T], Any]
    ) -> List[R]:
        """Process batch items concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(item: T) -> R:
            async with semaphore:
                if self.config.timeout:
                    return await asyncio.wait_for(
                        processor(item),
                        timeout=self.config.timeout
                    )
                return await processor(item)
        
        tasks = [process_with_semaphore(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Item {i} processing failed: {result}")
                if self.config.stop_on_error:
                    raise result
            else:
                processed_results.append(result)
        
        return processed_results


async def process_in_batches(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[R]:
    """
    Convenience function for batch processing
    
    Args:
        items: Items to process
        processor: Processing function
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of results
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )
    processor_instance = BatchProcessor(config)
    return await processor_instance.process_batch(items, processor)















