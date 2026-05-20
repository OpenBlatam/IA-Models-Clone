"""
Batch Manager
Advanced batch processing with dynamic sizing and prioritization.
"""

import asyncio
from typing import List, Any, Callable, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for batch items."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchItem:
    """Item in a batch."""
    data: Any
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchManager:
    """Manage batching with dynamic sizing and prioritization."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 1.0,
        max_batch_size: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size or batch_size * 2
        self.queue: List[BatchItem] = []
        self.processing = False
    
    def add(self, data: Any, priority: Priority = Priority.NORMAL, **metadata) -> BatchItem:
        """Add item to batch queue."""
        item = BatchItem(data=data, priority=priority, metadata=metadata)
        self.queue.append(item)
        
        # Sort by priority (higher first) and timestamp
        self.queue.sort(
            key=lambda x: (x.priority.value, -x.timestamp),
            reverse=True
        )
        
        return item
    
    def get_batch(self) -> List[BatchItem]:
        """Get next batch to process."""
        if not self.queue:
            return []
        
        # Get items up to batch_size
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        return batch
    
    async def process_batches(
        self,
        processor: Callable[[List[Any]], List[Any]],
        stop_event: Optional[asyncio.Event] = None,
    ):
        """Process batches continuously."""
        self.processing = True
        
        while self.processing:
            if stop_event and stop_event.is_set():
                break
            
            # Wait for items or timeout
            if not self.queue:
                await asyncio.sleep(0.1)
                continue
            
            # Get batch
            batch_items = self.get_batch()
            if not batch_items:
                continue
            
            # Extract data
            batch_data = [item.data for item in batch_items]
            
            try:
                # Process
                results = await processor(batch_data)
                
                # Store results in metadata
                for item, result in zip(batch_items, results):
                    item.metadata["result"] = result
                    item.metadata["processed"] = True
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for item in batch_items:
                    item.metadata["error"] = str(e)
                    item.metadata["processed"] = False
    
    def stop(self):
        """Stop batch processing."""
        self.processing = False


class DynamicBatchProcessor:
    """Dynamic batch processor that adjusts batch size based on performance."""
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_latency: float = 1.0,
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.recent_latencies: List[float] = []
    
    def adjust_batch_size(self, latency: float):
        """Adjust batch size based on latency."""
        self.recent_latencies.append(latency)
        if len(self.recent_latencies) > 10:
            self.recent_latencies.pop(0)
        
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        
        if avg_latency < self.target_latency * 0.8:
            # Too fast, increase batch size
            self.batch_size = min(
                int(self.batch_size * 1.2),
                self.max_batch_size
            )
        elif avg_latency > self.target_latency * 1.2:
            # Too slow, decrease batch size
            self.batch_size = max(
                int(self.batch_size * 0.8),
                self.min_batch_size
            )
    
    async def process(
        self,
        items: List[Any],
        processor: Callable[[List[Any]], List[Any]],
    ) -> List[Any]:
        """Process items in dynamic batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            start_time = time.time()
            batch_results = await processor(batch)
            latency = time.time() - start_time
            
            self.adjust_batch_size(latency)
            results.extend(batch_results)
        
        return results



