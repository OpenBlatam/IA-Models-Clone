"""
Batch Processor
Efficient batch processing for inference.
"""

import torch
from typing import List, Any, Callable, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Efficient batch processor for inference.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        process_fn: Optional[Callable] = None,
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.process_fn = process_fn
        self.queue: deque = deque()
    
    def add(self, item: Any) -> bool:
        """
        Add item to batch queue.
        
        Returns:
            True if batch is ready to process
        """
        self.queue.append(item)
        return len(self.queue) >= self.batch_size
    
    def process_batch(self, items: List[Any]) -> List[Any]:
        """Process a batch of items."""
        if self.process_fn:
            return self.process_fn(items)
        return items
    
    def flush(self) -> List[Any]:
        """Process remaining items in queue."""
        if not self.queue:
            return []
        
        items = list(self.queue)
        self.queue.clear()
        return self.process_batch(items)


class DynamicBatchProcessor:
    """
    Dynamic batch processor that adapts batch size.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        process_fn: Optional[Callable] = None,
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.process_fn = process_fn
        self.processing_times = []
    
    def adjust_batch_size(self, processing_time: float, target_time: float = 1.0):
        """Adjust batch size based on processing time."""
        self.processing_times.append(processing_time)
        
        if len(self.processing_times) > 10:
            avg_time = sum(self.processing_times[-10:]) / 10
            
            if avg_time < target_time * 0.8:
                # Increase batch size
                self.batch_size = min(
                    self.batch_size * 2,
                    self.max_batch_size
                )
            elif avg_time > target_time * 1.2:
                # Decrease batch size
                self.batch_size = max(
                    self.batch_size // 2,
                    self.min_batch_size
                )
    
    def process(self, items: List[Any]) -> List[Any]:
        """Process items with dynamic batching."""
        import time
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        results = []
        for batch in batches:
            start_time = time.time()
            
            if self.process_fn:
                batch_results = self.process_fn(batch)
            else:
                batch_results = batch
            
            processing_time = time.time() - start_time
            self.adjust_batch_size(processing_time)
            
            results.extend(batch_results)
        
        return results
