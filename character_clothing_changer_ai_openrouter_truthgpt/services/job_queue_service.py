"""
Job Queue Service
=================
Service for managing job queues with priorities and workers
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from heapq import heappush, heappop
import threading

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class JobStatus(Enum):
    """Job status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class Job:
    """Job definition"""
    id: str
    job_type: str
    payload: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare jobs by priority (lower priority value = higher priority)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class JobQueueService:
    """
    Service for managing job queues with priorities.
    
    Features:
    - Priority-based job queue
    - Multiple workers
    - Retry logic
    - Job status tracking
    - Statistics
    """
    
    def __init__(self, num_workers: int = 3):
        """
        Initialize job queue service.
        
        Args:
            num_workers: Number of worker threads
        """
        self.num_workers = num_workers
        self._queue: List[Job] = []
        self._queue_lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}
        self._stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'cancelled_jobs': 0
        }
    
    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ):
        """Register handler for job type"""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        job_id: Optional[str] = None
    ) -> Job:
        """
        Enqueue a job.
        
        Args:
            job_type: Type of job
            payload: Job payload
            priority: Job priority
            max_retries: Maximum retry attempts
            job_id: Optional job ID (auto-generated if not provided)
        
        Returns:
            Job object
        """
        if job_id is None:
            job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        job = Job(
            id=job_id,
            job_type=job_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries
        )
        
        with self._queue_lock:
            self._jobs[job_id] = job
            heappush(self._queue, job)
            job.status = JobStatus.QUEUED
            self._stats['total_jobs'] += 1
        
        logger.info(f"Job {job_id} enqueued (type: {job_type}, priority: {priority.name})")
        
        # Notify workers if queue was empty
        if len(self._queue) == 1:
            self._notify_workers()
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self._stats['cancelled_jobs'] += 1
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    async def start(self):
        """Start worker threads"""
        if self._running:
            return
        
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.num_workers)
        ]
        logger.info(f"Job queue service started with {self.num_workers} workers")
    
    async def stop(self):
        """Stop worker threads"""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for all workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers = []
        logger.info("Job queue service stopped")
    
    def _notify_workers(self):
        """Notify workers that a job is available"""
        # In a real implementation, use asyncio.Event or similar
        pass
    
    async def _worker(self, worker_name: str):
        """Worker thread that processes jobs"""
        logger.info(f"Worker {worker_name} started")
        
        while self._running:
            job = None
            
            # Get job from queue
            with self._queue_lock:
                if self._queue:
                    job = heappop(self._queue)
            
            if job is None:
                await asyncio.sleep(0.1)
                continue
            
            # Skip cancelled jobs
            if job.status == JobStatus.CANCELLED:
                continue
            
            # Process job
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now()
            
            try:
                handler = self._handlers.get(job.job_type)
                if not handler:
                    raise ValueError(f"No handler registered for job type: {job.job_type}")
                
                result = await handler(job.payload)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                self._stats['completed_jobs'] += 1
                
                logger.info(f"Job {job.id} completed by {worker_name}")
            
            except Exception as e:
                job.error = str(e)
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    logger.warning(
                        f"Job {job.id} failed (attempt {job.retry_count}/{job.max_retries}): {e}. Retrying..."
                    )
                    # Re-queue with same priority
                    with self._queue_lock:
                        heappush(self._queue, job)
                else:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    self._stats['failed_jobs'] += 1
                    logger.error(f"Job {job.id} failed after {job.retry_count} attempts: {e}")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        with self._queue_lock:
            return len(self._queue)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._queue_lock:
            pending = len([j for j in self._queue if j.status == JobStatus.QUEUED])
            processing = len([j for j in self._jobs.values() if j.status == JobStatus.PROCESSING])
        
        return {
            'queue_size': self.get_queue_size(),
            'pending_jobs': pending,
            'processing_jobs': processing,
            'total_jobs': self._stats['total_jobs'],
            'completed_jobs': self._stats['completed_jobs'],
            'failed_jobs': self._stats['failed_jobs'],
            'cancelled_jobs': self._stats['cancelled_jobs'],
            'num_workers': self.num_workers,
            'running': self._running,
            'registered_handlers': list(self._handlers.keys())
        }


# Global job queue service instance
_job_queue_service: Optional[JobQueueService] = None


def get_job_queue_service() -> JobQueueService:
    """Get or create job queue service instance"""
    global _job_queue_service
    if _job_queue_service is None:
        _job_queue_service = JobQueueService()
    return _job_queue_service

