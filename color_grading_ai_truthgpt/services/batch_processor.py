"""
Batch Processor for Color Grading AI
====================================

Processes multiple videos/images in batch with progress tracking.
"""

import logging
import asyncio
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Item in batch processing."""
    id: str
    input_path: str
    output_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class BatchJob:
    """Batch processing job."""
    id: str
    items: List[BatchItem]
    status: str = "pending"
    total: int = 0
    completed: int = 0
    failed: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        self.total = len(self.items)


class BatchProcessor:
    """
    Processes multiple media files in batch.
    
    Features:
    - Process multiple files in parallel
    - Progress tracking
    - Error handling per item
    - Resume capability
    """
    
    def __init__(self, max_parallel: int = 3):
        """
        Initialize batch processor.
        
        Args:
            max_parallel: Maximum parallel processing
        """
        self.max_parallel = max_parallel
        self._jobs: Dict[str, BatchJob] = {}
        self._semaphore = asyncio.Semaphore(max_parallel)
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        processor_func: Callable,
        job_id: Optional[str] = None
    ) -> BatchJob:
        """
        Process batch of items.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            job_id: Optional job ID
            
        Returns:
            BatchJob with results
        """
        import uuid
        job_id = job_id or str(uuid.uuid4())
        
        # Create batch items
        batch_items = [
            BatchItem(
                id=str(uuid.uuid4()),
                input_path=item["input_path"],
                output_path=item.get("output_path"),
                parameters=item.get("parameters", {})
            )
            for item in items
        ]
        
        # Create job
        job = BatchJob(id=job_id, items=batch_items, status="processing")
        job.started_at = datetime.now()
        self._jobs[job_id] = job
        
        # Process items
        tasks = [
            self._process_item(item, processor_func, job)
            for item in batch_items
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update job status
        job.status = "completed" if job.failed == 0 else "completed_with_errors"
        job.completed_at = datetime.now()
        
        return job
    
    async def _process_item(
        self,
        item: BatchItem,
        processor_func: Callable,
        job: BatchJob
    ):
        """Process single item."""
        async with self._semaphore:
            item.status = "processing"
            try:
                result = await processor_func(
                    item.input_path,
                    item.output_path,
                    **item.parameters
                )
                item.status = "completed"
                item.result = result
                item.completed_at = datetime.now()
                job.completed += 1
            except Exception as e:
                item.status = "failed"
                item.error = str(e)
                item.completed_at = datetime.now()
                job.failed += 1
                logger.error(f"Error processing item {item.id}: {e}")
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        return {
            "id": job.id,
            "status": job.status,
            "total": job.total,
            "completed": job.completed,
            "failed": job.failed,
            "progress": (job.completed + job.failed) / job.total * 100 if job.total > 0 else 0,
            "items": [
                {
                    "id": item.id,
                    "status": item.status,
                    "error": item.error
                }
                for item in job.items
            ]
        }




