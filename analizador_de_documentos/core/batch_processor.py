"""
Batch Processor for Document Analyzer
======================================

Advanced batch processing with intelligent batching, progress tracking, and error recovery.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Status of batch processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    total_items: int
    processed: int
    successful: int
    failed: int
    errors: List[str] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed / self.total_items) * 100

class BatchProcessor:
    """Advanced batch processor for document analysis"""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 5,
        retry_on_failure: bool = True,
        max_retries: int = 3,
        progress_callback: Optional[Callable] = None
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self.progress_callback = progress_callback
        self.active_batches: Dict[str, BatchResult] = {}
        logger.info(f"BatchProcessor initialized. Batch size: {batch_size}, Max workers: {max_workers}")
    
    def calculate_optimal_batch_size(self, total_items: int, estimated_time_per_item: float = 1.0) -> int:
        """Calculate optimal batch size based on total items and estimated time"""
        if total_items <= 10:
            return total_items
        
        # Target: complete in reasonable time (e.g., 30 seconds per batch)
        target_batch_time = 30.0
        optimal = int(target_batch_time / estimated_time_per_item)
        
        # Ensure reasonable bounds
        optimal = max(5, min(optimal, 100))
        
        return optimal
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_id: Optional[str] = None,
        **processor_kwargs
    ) -> BatchResult:
        """
        Process a batch of items
        
        Args:
            items: List of items to process
            processor_func: Async function to process each item
            batch_id: Optional batch ID
            **processor_kwargs: Additional kwargs for processor function
        
        Returns:
            BatchResult with processing results
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
        
        result = BatchResult(
            batch_id=batch_id,
            total_items=len(items),
            start_time=datetime.now()
        )
        result.status = BatchStatus.PROCESSING
        self.active_batches[batch_id] = result
        
        logger.info(f"Starting batch {batch_id} with {len(items)} items")
        
        # Use semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_item(item: Any, index: int) -> Tuple[int, Any, Optional[str]]:
            """Process single item with retry logic"""
            async with semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        item_result = await processor_func(item, **processor_kwargs)
                        return (index, item_result, None)
                    except Exception as e:
                        if attempt < self.max_retries and self.retry_on_failure:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Item {index} failed (attempt {attempt + 1}/{self.max_retries}). Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_msg = f"Item {index} failed after {attempt + 1} attempts: {str(e)}"
                            logger.error(error_msg)
                            return (index, None, error_msg)
                return (index, None, "Max retries exceeded")
        
        # Process items
        tasks = [process_item(item, i) for i, item in enumerate(items)]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for res in results_list:
            if isinstance(res, Exception):
                result.failed += 1
                result.errors.append(f"Processing exception: {str(res)}")
            else:
                index, item_result, error = res
                result.processed += 1
                if error:
                    result.failed += 1
                    result.errors.append(error)
                else:
                    result.successful += 1
                    result.results.append(item_result)
        
        # Update status
        if result.failed == 0:
            result.status = BatchStatus.COMPLETED
        elif result.successful == 0:
            result.status = BatchStatus.FAILED
        else:
            result.status = BatchStatus.PARTIAL
        
        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds()
        
        # Call progress callback
        if self.progress_callback:
            try:
                await self.progress_callback(result)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
        
        logger.info(
            f"Batch {batch_id} completed. "
            f"Processed: {result.processed}/{result.total_items}, "
            f"Success: {result.successful}, Failed: {result.failed}, "
            f"Duration: {result.duration:.2f}s"
        )
        
        return result
    
    async def process_in_batches(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_id_prefix: str = "batch",
        **processor_kwargs
    ) -> List[BatchResult]:
        """
        Process items in multiple batches
        
        Args:
            items: List of items to process
            processor_func: Async function to process each item
            batch_id_prefix: Prefix for batch IDs
            **processor_kwargs: Additional kwargs for processor function
        
        Returns:
            List of BatchResult objects
        """
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        all_results = []
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            
            batch_id = f"{batch_id_prefix}_{batch_num + 1}_{total_batches}"
            result = await self.process_batch(
                batch_items,
                processor_func,
                batch_id=batch_id,
                **processor_kwargs
            )
            all_results.append(result)
            
            # Optional delay between batches
            if batch_num < total_batches - 1:
                await asyncio.sleep(0.1)
        
        return all_results
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchResult]:
        """Get status of a batch"""
        return self.active_batches.get(batch_id)
    
    def get_all_batches(self) -> Dict[str, BatchResult]:
        """Get all active batches"""
        return self.active_batches.copy()
    
    def clear_completed_batches(self, max_age_seconds: int = 3600):
        """Clear completed batches older than max_age_seconds"""
        current_time = datetime.now()
        to_remove = []
        
        for batch_id, result in self.active_batches.items():
            if result.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                if result.end_time:
                    age = (current_time - result.end_time).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.active_batches[batch_id]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old batches")
















