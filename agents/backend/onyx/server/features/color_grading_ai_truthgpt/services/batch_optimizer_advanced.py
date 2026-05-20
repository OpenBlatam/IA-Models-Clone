"""
Advanced Batch Optimizer for Color Grading AI
==============================================

Advanced batch optimization with intelligent grouping and parallel processing.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CHUNKED = "chunked"
    PRIORITY = "priority"
    ADAPTIVE = "adaptive"


@dataclass
class BatchItem:
    """Batch item."""
    item_id: str
    data: Any
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchResult:
    """Batch processing result."""
    batch_id: str
    items_processed: int
    items_succeeded: int
    items_failed: int
    total_duration: float
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedBatchOptimizer:
    """
    Advanced batch optimizer.
    
    Features:
    - Multiple batch strategies
    - Intelligent grouping
    - Parallel processing
    - Priority handling
    - Adaptive optimization
    - Progress tracking
    """
    
    def __init__(self, max_parallel: int = 10):
        """
        Initialize advanced batch optimizer.
        
        Args:
            max_parallel: Maximum parallel items
        """
        self.max_parallel = max_parallel
        self._batches: Dict[str, List[BatchItem]] = {}
        self._results: Dict[str, BatchResult] = {}
    
    def create_batch(
        self,
        batch_id: str,
        items: List[Any],
        item_ids: Optional[List[str]] = None,
        priorities: Optional[List[int]] = None
    ) -> str:
        """
        Create batch.
        
        Args:
            batch_id: Batch ID
            items: List of items to process
            item_ids: Optional item IDs
            priorities: Optional priorities
            
        Returns:
            Batch ID
        """
        batch_items = []
        
        for i, item in enumerate(items):
            item_id = item_ids[i] if item_ids and i < len(item_ids) else f"{batch_id}_item_{i}"
            priority = priorities[i] if priorities and i < len(priorities) else 0
            
            batch_items.append(BatchItem(
                item_id=item_id,
                data=item,
                priority=priority
            ))
        
        # Sort by priority
        batch_items.sort(key=lambda x: x.priority, reverse=True)
        
        self._batches[batch_id] = batch_items
        logger.info(f"Created batch {batch_id} with {len(batch_items)} items")
        
        return batch_id
    
    async def process_batch(
        self,
        batch_id: str,
        processor: Callable,
        strategy: BatchStrategy = BatchStrategy.PARALLEL,
        chunk_size: Optional[int] = None
    ) -> BatchResult:
        """
        Process batch.
        
        Args:
            batch_id: Batch ID
            processor: Processor function
            strategy: Processing strategy
            chunk_size: Optional chunk size for chunked strategy
            
        Returns:
            Batch result
        """
        if batch_id not in self._batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch_items = self._batches[batch_id]
        import time
        start_time = time.time()
        
        results = []
        errors = []
        succeeded = 0
        failed = 0
        
        if strategy == BatchStrategy.SEQUENTIAL:
            # Process sequentially
            for item in batch_items:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item.data)
                    else:
                        result = processor(item.data)
                    results.append(result)
                    succeeded += 1
                except Exception as e:
                    errors.append({"item_id": item.item_id, "error": str(e)})
                    failed += 1
        
        elif strategy == BatchStrategy.PARALLEL:
            # Process in parallel
            semaphore = asyncio.Semaphore(self.max_parallel)
            
            async def process_item(item: BatchItem):
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            return await processor(item.data)
                        else:
                            return processor(item.data)
                    except Exception as e:
                        errors.append({"item_id": item.item_id, "error": str(e)})
                        return None
            
            tasks = [process_item(item) for item in batch_items]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter results
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                elif result is not None:
                    filtered_results.append(result)
                    succeeded += 1
                else:
                    failed += 1
            
            results = filtered_results
        
        elif strategy == BatchStrategy.CHUNKED:
            # Process in chunks
            chunk_size = chunk_size or self.max_parallel
            
            for i in range(0, len(batch_items), chunk_size):
                chunk = batch_items[i:i + chunk_size]
                
                async def process_chunk(chunk_items):
                    return await asyncio.gather(*[
                        processor(item.data) if asyncio.iscoroutinefunction(processor)
                        else asyncio.to_thread(processor, item.data)
                        for item in chunk_items
                    ], return_exceptions=True)
                
                chunk_results = await process_chunk(chunk)
                
                for result in chunk_results:
                    if isinstance(result, Exception):
                        failed += 1
                    else:
                        results.append(result)
                        succeeded += 1
        
        elif strategy == BatchStrategy.PRIORITY:
            # Process by priority (already sorted)
            semaphore = asyncio.Semaphore(self.max_parallel)
            
            async def process_item(item: BatchItem):
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            return await processor(item.data)
                        else:
                            return processor(item.data)
                    except Exception as e:
                        errors.append({"item_id": item.item_id, "error": str(e)})
                        return None
            
            tasks = [process_item(item) for item in batch_items]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                elif result is not None:
                    filtered_results.append(result)
                    succeeded += 1
                else:
                    failed += 1
            
            results = filtered_results
        
        elif strategy == BatchStrategy.ADAPTIVE:
            # Adaptive: start with parallel, adjust based on performance
            # Simplified version
            return await self.process_batch(batch_id, processor, BatchStrategy.PARALLEL)
        
        duration = time.time() - start_time
        
        batch_result = BatchResult(
            batch_id=batch_id,
            items_processed=len(batch_items),
            items_succeeded=succeeded,
            items_failed=failed,
            total_duration=duration,
            results=results,
            errors=errors
        )
        
        self._results[batch_id] = batch_result
        logger.info(f"Batch {batch_id} processed: {succeeded} succeeded, {failed} failed in {duration:.2f}s")
        
        return batch_result
    
    def get_batch_result(self, batch_id: str) -> Optional[BatchResult]:
        """Get batch result."""
        return self._results.get(batch_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total_batches = len(self._batches)
        total_results = len(self._results)
        
        if total_results == 0:
            return {
                "batches_count": total_batches,
                "results_count": 0,
                "avg_success_rate": 0.0,
            }
        
        total_items = sum(r.items_processed for r in self._results.values())
        total_succeeded = sum(r.items_succeeded for r in self._results.values())
        
        return {
            "batches_count": total_batches,
            "results_count": total_results,
            "total_items_processed": total_items,
            "total_items_succeeded": total_succeeded,
            "avg_success_rate": total_succeeded / total_items if total_items > 0 else 0.0,
        }




