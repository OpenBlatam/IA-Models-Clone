"""
Batch Processing Utilities
===========================

Utilities for batch processing of images.
"""

import logging
import asyncio
from typing import List, Callable, Optional, Union, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

logger = logging.getLogger(__name__)


class BatchProcessingUtils:
    """Utilities for batch processing."""
    
    @staticmethod
    def process_batch_sync(
        items: List[Any],
        process_func: Callable,
        batch_size: int = 4,
        max_workers: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process items in batch with parallel processing (synchronous).
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Number of items to process in parallel
            max_workers: Maximum worker threads
            progress_callback: Optional callback(current, total) for progress
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        results = []
        total = len(items)
        
        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = items[batch_start:batch_end]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for item in batch_items:
                    future = executor.submit(process_func, item, **kwargs)
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        current = batch_start + i + 1
                        if progress_callback:
                            progress_callback(current, total)
                        logger.info(f"Processed item {current}/{total}")
                    except Exception as e:
                        logger.error(f"Failed to process item {batch_start + i + 1}: {e}")
                        results.append(None)
        
        return results
    
    @staticmethod
    async def process_batch_async(
        items: List[Any],
        process_func: Callable,
        max_concurrent: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process items in batch with async concurrency control.
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback(current, total) for progress
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(items)
        
        async def process_one(item, index):
            nonlocal completed
            async with semaphore:
                try:
                    result = await process_func(item, **kwargs)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    return (index, result)
                except Exception as e:
                    logger.error(f"Failed to process item {index + 1}: {e}")
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    return (index, None)
        
        # Create tasks
        tasks = [
            process_one(item, i)
            for i, item in enumerate(items)
        ]
        
        # Execute all tasks
        task_results = await asyncio.gather(*tasks)
        
        # Sort by index and extract results
        task_results.sort(key=lambda x: x[0])
        results = [result for _, result in task_results]
        
        return results


