"""
Batch Processing Utilities for Piel Mejorador AI SAM3
======================================================

Unified batch processing utilities with progress tracking.
"""

import asyncio
import logging
from typing import List, Callable, Optional, TypeVar, Awaitable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime

from .progress_utils import ProgressTracker, ProgressUtils

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds."""
        if not self.start_time or not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results_count": len(self.results),
            "errors_count": len(self.errors),
        }


class BatchProcessingUtils:
    """Unified batch processing utilities."""
    
    @staticmethod
    async def process_batch(
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        batch_size: int = 10,
        max_concurrent: int = 5,
        progress_callback: Optional[Callable[[ProgressTracker], None]] = None,
        skip_errors: bool = False
    ) -> BatchResult:
        """
        Process items in batches with concurrency control.
        
        Args:
            items: List of items to process
            processor: Async processor function
            batch_size: Size of each batch
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional progress callback
            skip_errors: Whether to skip errors and continue
            
        Returns:
            BatchResult
        """
        result = BatchResult(
            total=len(items),
            start_time=datetime.now()
        )
        
        tracker = ProgressUtils.create_tracker(len(items))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item: T, index: int) -> Optional[R]:
            async with semaphore:
                try:
                    processed = await processor(item)
                    result.successful += 1
                    tracker.increment()
                    return processed
                except Exception as e:
                    result.failed += 1
                    tracker.increment_failed()
                    error_info = {
                        "index": index,
                        "item": str(item),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    result.errors.append(error_info)
                    
                    if not skip_errors:
                        raise
                    
                    logger.warning(f"Error processing item {index}: {e}")
                    return None
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch)))
            
            batch_results = await asyncio.gather(
                *[process_item(item, idx) for item, idx in zip(batch, batch_indices)],
                return_exceptions=True
            )
            
            # Collect results
            for processed in batch_results:
                if processed is not None and not isinstance(processed, Exception):
                    result.results.append(processed)
            
            # Update progress
            if progress_callback:
                progress_callback(tracker)
            else:
                logger.info(ProgressUtils.format_progress(tracker))
        
        result.end_time = datetime.now()
        tracker.complete()
        
        logger.info(
            f"Batch processing completed: {result.successful} succeeded, "
            f"{result.failed} failed in {result.duration_seconds:.2f}s"
        )
        
        return result
    
    @staticmethod
    async def process_chunks(
        items: List[T],
        processor: Callable[[List[T]], Awaitable[List[R]]],
        chunk_size: int = 10,
        progress_callback: Optional[Callable[[ProgressTracker], None]] = None
    ) -> BatchResult:
        """
        Process items in chunks (each chunk processed as a unit).
        
        Args:
            items: List of items to process
            processor: Async processor function that takes a chunk
            chunk_size: Size of each chunk
            progress_callback: Optional progress callback
            
        Returns:
            BatchResult
        """
        result = BatchResult(
            total=len(items),
            start_time=datetime.now()
        )
        
        tracker = ProgressUtils.create_tracker(len(items))
        
        # Split into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                chunk_results = await processor(chunk)
                result.successful += len(chunk_results)
                tracker.increment(len(chunk_results))
                result.results.extend(chunk_results)
            except Exception as e:
                result.failed += len(chunk)
                tracker.increment_failed(len(chunk))
                error_info = {
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                result.errors.append(error_info)
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
            
            # Update progress
            if progress_callback:
                progress_callback(tracker)
            else:
                logger.info(ProgressUtils.format_progress(tracker))
        
        result.end_time = datetime.now()
        tracker.complete()
        
        return result
    
    @staticmethod
    async def process_with_retry(
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
        progress_callback: Optional[Callable[[ProgressTracker], None]] = None
    ) -> BatchResult:
        """
        Process items with retry logic.
        
        Args:
            items: List of items to process
            processor: Async processor function
            max_retries: Maximum retries per item
            batch_size: Size of each batch
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional progress callback
            
        Returns:
            BatchResult
        """
        from .retry_utils import RetryUtils
        
        async def process_with_retry_item(item: T) -> R:
            return await RetryUtils.retry_async(
                lambda: processor(item),
                max_retries=max_retries,
                operation_name="batch_item_processing"
            )
        
        return await BatchProcessingUtils.process_batch(
            items,
            process_with_retry_item,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback,
            skip_errors=True
        )


# Convenience functions
async def process_batch(
    items: List[T],
    processor: Callable[[T], Awaitable[R]],
    **kwargs
) -> BatchResult:
    """Process batch."""
    return await BatchProcessingUtils.process_batch(items, processor, **kwargs)


async def process_chunks(
    items: List[T],
    processor: Callable[[List[T]], Awaitable[List[R]]],
    **kwargs
) -> BatchResult:
    """Process chunks."""
    return await BatchProcessingUtils.process_chunks(items, processor, **kwargs)




