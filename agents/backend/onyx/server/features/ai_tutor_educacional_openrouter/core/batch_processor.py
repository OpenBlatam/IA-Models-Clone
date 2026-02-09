"""
Batch processing utilities for efficient data handling.
"""

import asyncio
import logging
from typing import List, Any, Callable, Optional, Dict
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Efficient batch processing with concurrency control.
    """
    
    def __init__(self, max_concurrent: int = 10, batch_size: int = 50):
        """
        Initialize batch processor.
        
        Args:
            max_concurrent: Maximum concurrent operations
            batch_size: Size of each batch
        """
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items in batches with concurrency control.
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of processed results
        """
        results = []
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self._process_batch_items(batch, process_func)
            results.extend(batch_results)
            
            if progress_callback:
                progress = min(i + self.batch_size, total)
                await progress_callback(progress, total)
        
        return results
    
    async def _process_batch_items(
        self,
        batch: List[Any],
        process_func: Callable
    ) -> List[Any]:
        """Process a single batch of items."""
        tasks = [
            self._process_with_semaphore(item, process_func)
            for item in batch
        ]
        return await asyncio.gather(*tasks)
    
    async def _process_with_semaphore(
        self,
        item: Any,
        process_func: Callable
    ) -> Any:
        """Process single item with semaphore control."""
        async with self.semaphore:
            return await process_func(item)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
            "available_slots": self.semaphore._value
        }




