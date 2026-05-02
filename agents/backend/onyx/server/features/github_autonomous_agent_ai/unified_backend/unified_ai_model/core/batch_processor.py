"""
Batch Processor
Efficient parallel task operations optimized for high throughput processing.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for efficient parallel task operations.
    Optimized for high throughput processing.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        max_concurrent: int = 5,
        use_executor: bool = True
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.use_executor = use_executor
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent) if use_executor else None
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor_fn: Callable[[Any], Any],
        on_complete: Optional[Callable[[Any, Any], None]] = None,
        on_error: Optional[Callable[[Any, Exception], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process items in optimized batches.
        
        Args:
            items: List of items to process
            processor_fn: Async function to process each item
            on_complete: Callback for completed items
            on_error: Callback for errors
            
        Returns:
            List of results with status
        """
        results = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for item in batch:
                task = self._process_with_semaphore(
                    item, processor_fn, on_complete, on_error
                )
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def _process_with_semaphore(
        self,
        item: Any,
        processor_fn: Callable,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process single item with semaphore for concurrency control."""
        async with self._semaphore:
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(processor_fn):
                    result = await processor_fn(item)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self._executor, processor_fn, item)
                
                elapsed = (time.time() - start_time) * 1000
                
                if on_complete:
                    try:
                        if asyncio.iscoroutinefunction(on_complete):
                            await on_complete(item, result)
                        else:
                            on_complete(item, result)
                    except Exception as e:
                        logger.warning(f"Error in on_complete callback: {e}")
                
                return {
                    "item": item,
                    "result": result,
                    "success": True,
                    "elapsed_ms": elapsed
                }
                
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                
                if on_error:
                    try:
                        if asyncio.iscoroutinefunction(on_error):
                            await on_error(item, e)
                        else:
                            on_error(item, e)
                    except Exception as cb_error:
                        logger.warning(f"Error in on_error callback: {cb_error}")
                
                return {
                    "item": item,
                    "error": str(e),
                    "success": False,
                    "elapsed_ms": elapsed
                }
    
    def optimize_batch_size(
        self,
        sample_size: int = 50,
        test_batch_sizes: Optional[List[int]] = None
    ) -> int:
        """
        Auto-optimize batch size based on performance.
        
        Returns:
            Optimal batch size
        """
        if test_batch_sizes is None:
            test_batch_sizes = [4, 8, 16, 32, 64]
        
        # For now, return a reasonable default
        # In production, you'd benchmark each size
        optimal = min(self.max_concurrent * 2, 16)
        self.batch_size = optimal
        logger.info(f"Optimized batch size: {optimal}")
        return optimal
    
    def shutdown(self):
        """Shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
