"""
Batch processing system for content analysis
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from types import ContentInput, SimilarityInput
from services import analyze_content, detect_similarity, assess_quality

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Individual batch job"""
    id: str
    input_data: Union[ContentInput, SimilarityInput]
    operation: str
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class BatchRequest:
    """Batch processing request"""
    id: str
    jobs: List[BatchJob]
    status: BatchStatus = BatchStatus.PENDING
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0


class BatchProcessor:
    """Batch processing system"""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self._active_batches: Dict[str, BatchRequest] = {}
        self._completed_batches: Dict[str, BatchRequest] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
    
    async def process_batch(self, batch_request: BatchRequest) -> BatchRequest:
        """Process a batch of jobs"""
        batch_request.status = BatchStatus.PROCESSING
        batch_request.started_at = time.time()
        batch_request.total_jobs = len(batch_request.jobs)
        
        self._active_batches[batch_request.id] = batch_request
        
        try:
            # Process jobs concurrently with semaphore
            tasks = [
                self._process_job(job, batch_request.id)
                for job in batch_request.jobs
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update batch status
            batch_request.completed_at = time.time()
            batch_request.progress = 100.0
            
            if batch_request.failed_jobs == 0:
                batch_request.status = BatchStatus.COMPLETED
            elif batch_request.completed_jobs > 0:
                batch_request.status = BatchStatus.COMPLETED  # Partial success
            else:
                batch_request.status = BatchStatus.FAILED
            
            # Move to completed batches
            self._completed_batches[batch_request.id] = batch_request
            if batch_request.id in self._active_batches:
                del self._active_batches[batch_request.id]
            
            logger.info(f"Batch {batch_request.id} completed: {batch_request.completed_jobs}/{batch_request.total_jobs} successful")
            
        except Exception as e:
            batch_request.status = BatchStatus.FAILED
            batch_request.completed_at = time.time()
            logger.error(f"Batch {batch_request.id} failed: {e}")
        
        return batch_request
    
    async def _process_job(self, job: BatchJob, batch_id: str) -> None:
        """Process individual job"""
        async with self._semaphore:
            try:
                job.status = BatchStatus.PROCESSING
                
                # Execute operation based on type
                if job.operation == "analyze":
                    result = analyze_content(job.input_data.content)
                elif job.operation == "similarity":
                    result = detect_similarity(
                        job.input_data.text1,
                        job.input_data.text2,
                        job.input_data.threshold
                    )
                elif job.operation == "quality":
                    result = assess_quality(job.input_data.content)
                else:
                    raise ValueError(f"Unknown operation: {job.operation}")
                
                job.result = result
                job.status = BatchStatus.COMPLETED
                job.completed_at = time.time()
                
                # Update batch progress
                if batch_id in self._active_batches:
                    batch = self._active_batches[batch_id]
                    batch.completed_jobs += 1
                    batch.progress = (batch.completed_jobs / batch.total_jobs) * 100
                
                logger.debug(f"Job {job.id} completed successfully")
                
            except Exception as e:
                job.status = BatchStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                
                # Update batch progress
                if batch_id in self._active_batches:
                    batch = self._active_batches[batch_id]
                    batch.failed_jobs += 1
                    batch.completed_jobs += 1
                    batch.progress = (batch.completed_jobs / batch.total_jobs) * 100
                
                logger.error(f"Job {job.id} failed: {e}")
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchRequest]:
        """Get batch status"""
        if batch_id in self._active_batches:
            return self._active_batches[batch_id]
        elif batch_id in self._completed_batches:
            return self._completed_batches[batch_id]
        return None
    
    def get_all_batches(self) -> Dict[str, BatchRequest]:
        """Get all batches (active and completed)"""
        all_batches = {}
        all_batches.update(self._active_batches)
        all_batches.update(self._completed_batches)
        return all_batches
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch"""
        if batch_id in self._active_batches:
            batch = self._active_batches[batch_id]
            batch.status = BatchStatus.CANCELLED
            batch.completed_at = time.time()
            
            # Cancel pending jobs
            for job in batch.jobs:
                if job.status == BatchStatus.PENDING:
                    job.status = BatchStatus.CANCELLED
                    job.completed_at = time.time()
            
            # Move to completed
            self._completed_batches[batch_id] = batch
            del self._active_batches[batch_id]
            
            logger.info(f"Batch {batch_id} cancelled")
            return True
        
        return False
    
    def cleanup_old_batches(self, max_age_hours: int = 24) -> int:
        """Clean up old completed batches"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for batch_id, batch in self._completed_batches.items():
            if current_time - batch.completed_at > max_age_seconds:
                to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self._completed_batches[batch_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old batches")
        return len(to_remove)


# Global batch processor
batch_processor = BatchProcessor(max_concurrent_jobs=10)


def create_batch_job(job_id: str, input_data: Union[ContentInput, SimilarityInput], operation: str) -> BatchJob:
    """Create a new batch job"""
    return BatchJob(
        id=job_id,
        input_data=input_data,
        operation=operation
    )


def create_batch_request(batch_id: str, jobs: List[BatchJob]) -> BatchRequest:
    """Create a new batch request"""
    return BatchRequest(
        id=batch_id,
        jobs=jobs
    )


async def process_batch_async(batch_request: BatchRequest) -> BatchRequest:
    """Process batch asynchronously"""
    return await batch_processor.process_batch(batch_request)


def get_batch_status(batch_id: str) -> Optional[BatchRequest]:
    """Get batch status"""
    return batch_processor.get_batch_status(batch_id)


def get_all_batches() -> Dict[str, BatchRequest]:
    """Get all batches"""
    return batch_processor.get_all_batches()


def cancel_batch(batch_id: str) -> bool:
    """Cancel batch"""
    return batch_processor.cancel_batch(batch_id)


def cleanup_old_batches(max_age_hours: int = 24) -> int:
    """Clean up old batches"""
    return batch_processor.cleanup_old_batches(max_age_hours)


