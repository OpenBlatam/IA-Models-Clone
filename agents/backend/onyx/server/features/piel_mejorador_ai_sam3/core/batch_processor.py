"""
Batch Processor for Piel Mejorador AI SAM3
===========================================

Process multiple files in batch with progress tracking.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Item in a batch processing job."""
    file_path: str
    enhancement_level: str = "medium"
    realism_level: Optional[float] = None
    custom_instructions: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchResult:
    """Result of batch processing."""
    total_items: int
    completed: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.completed / self.total_items
    
    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BatchProcessor:
    """
    Processes multiple files in batch.
    
    Features:
    - Parallel batch processing
    - Progress tracking
    - Error handling per item
    - Result aggregation
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize batch processor.
        
        Args:
            max_concurrent: Maximum concurrent processing tasks
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[BatchItem],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process a batch of items.
        
        Args:
            items: List of batch items to process
            process_func: Async function to process each item
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with aggregated results
        """
        result = BatchResult(
            total_items=len(items),
            completed=0,
            failed=0,
            results=[],
            errors=[],
            start_time=datetime.now()
        )
        
        logger.info(f"Starting batch processing: {len(items)} items")
        
        # Process items with concurrency limit
        tasks = []
        for item in items:
            task = self._process_item(item, process_func, result, progress_callback)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        result.end_time = datetime.now()
        
        logger.info(
            f"Batch processing completed: {result.completed} succeeded, "
            f"{result.failed} failed in {result.duration:.2f}s"
        )
        
        return result
    
    async def _process_item(
        self,
        item: BatchItem,
        process_func: Callable,
        result: BatchResult,
        progress_callback: Optional[Callable]
    ):
        """Process a single item with semaphore."""
        async with self._semaphore:
            try:
                # Process item
                item_result = await process_func(
                    file_path=item.file_path,
                    enhancement_level=item.enhancement_level,
                    realism_level=item.realism_level,
                    custom_instructions=item.custom_instructions,
                    priority=item.priority
                )
                
                result.results.append({
                    "file_path": item.file_path,
                    "success": True,
                    "result": item_result,
                    "metadata": item.metadata
                })
                
                result.completed += 1
                
                logger.debug(f"Processed item: {item.file_path}")
                
            except Exception as e:
                result.errors.append({
                    "file_path": item.file_path,
                    "error": str(e),
                    "metadata": item.metadata
                })
                
                result.failed += 1
                
                logger.error(f"Error processing item {item.file_path}: {e}")
            
            finally:
                # Call progress callback if provided
                if progress_callback:
                    try:
                        await progress_callback(
                            completed=result.completed,
                            failed=result.failed,
                            total=result.total_items
                        )
                    except Exception as e:
                        logger.warning(f"Error in progress callback: {e}")




