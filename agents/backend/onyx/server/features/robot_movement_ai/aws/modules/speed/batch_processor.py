"""
Batch Processor
===============

Process requests in batches for maximum throughput.
"""

import logging
import asyncio
from typing import List, Any, Callable, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_size: int = 100
    max_wait: float = 0.1  # seconds
    max_concurrent: int = 10


class BatchProcessor:
    """Batch processor for high throughput."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._batches: Dict[str, List[Any]] = {}
        self._batch_times: Dict[str, datetime] = {}
        self._processors: Dict[str, Callable] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
    
    def register_processor(self, batch_type: str, processor: Callable):
        """Register batch processor."""
        self._processors[batch_type] = processor
        self._batches[batch_type] = []
        self._batch_times[batch_type] = datetime.now()
        logger.info(f"Registered batch processor: {batch_type}")
    
    async def add_item(self, batch_type: str, item: Any) -> Any:
        """Add item to batch."""
        if batch_type not in self._batches:
            raise ValueError(f"Batch type {batch_type} not registered")
        
        self._batches[batch_type].append(item)
        
        # Check if batch should be processed
        batch = self._batches[batch_type]
        should_process = (
            len(batch) >= self.config.max_size or
            (datetime.now() - self._batch_times[batch_type]).total_seconds() >= self.config.max_wait
        )
        
        if should_process:
            return await self._process_batch(batch_type)
        
        return None
    
    async def _process_batch(self, batch_type: str) -> Any:
        """Process batch."""
        batch = self._batches[batch_type].copy()
        self._batches[batch_type].clear()
        self._batch_times[batch_type] = datetime.now()
        
        if not batch:
            return None
        
        processor = self._processors[batch_type]
        
        async with self._semaphore:
            try:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(batch)
                else:
                    result = processor(batch)
                
                logger.debug(f"Processed batch {batch_type}: {len(batch)} items")
                return result
            
            except Exception as e:
                logger.error(f"Batch processing failed for {batch_type}: {e}")
                raise
    
    async def flush_all(self):
        """Flush all pending batches."""
        results = []
        for batch_type in list(self._batches.keys()):
            if self._batches[batch_type]:
                result = await self._process_batch(batch_type)
                results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            "registered_types": list(self._processors.keys()),
            "pending_batches": {
                batch_type: len(batch)
                for batch_type, batch in self._batches.items()
            }
        }















