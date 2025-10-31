"""
Job Manager for Refactored Opus Clip

Advanced job management with:
- Async job processing
- Priority queues
- Resource management
- Progress tracking
- Error handling and retries
- Job persistence
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Union
import asyncio
import uuid
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import PriorityQueue, Empty
import pickle
import os
from pathlib import Path

logger = structlog.get_logger("job_manager")

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Job:
    """Job data structure."""
    id: str
    type: str
    data: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Priority queue comparison."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at

@dataclass
class JobResult:
    """Job result data structure."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class JobManager:
    """
    Advanced job manager for Opus Clip.
    
    Features:
    - Async job processing
    - Priority queues
    - Resource management
    - Progress tracking
    - Error handling and retries
    - Job persistence
    """
    
    def __init__(self, max_workers: int = 4, 
                 persistence_path: Optional[str] = None,
                 enable_persistence: bool = True):
        """Initialize job manager."""
        self.max_workers = max_workers
        self.persistence_path = persistence_path or "jobs.pkl"
        self.enable_persistence = enable_persistence
        self.logger = structlog.get_logger("job_manager")
        
        # Job storage
        self.jobs: Dict[str, Job] = {}
        self.job_queue = PriorityQueue()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Job processors
        self.processors: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Load persisted jobs
        if self.enable_persistence:
            self._load_persisted_jobs()
        
        self.logger.info(f"Initialized JobManager with {max_workers} workers")
    
    def register_processor(self, job_type: str, processor: Callable):
        """Register a job processor."""
        self.processors[job_type] = processor
        self.logger.info(f"Registered processor for job type: {job_type}")
    
    async def submit_job(self, job_type: str, data: Dict[str, Any],
                        priority: JobPriority = JobPriority.NORMAL,
                        max_retries: int = 3,
                        timeout_seconds: float = 300.0,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new job for processing."""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            type=job_type,
            data=data,
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        self.job_queue.put(job)
        self.stats["total_jobs"] += 1
        
        # Persist job
        if self.enable_persistence:
            self._persist_jobs()
        
        self.logger.info(f"Submitted job {job_id} of type {job_type}")
        
        # Start processing if not running
        if not self.running:
            asyncio.create_task(self._start_processing())
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "type": job.type,
            "status": job.status.value,
            "priority": job.priority.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "error": job.error,
            "metadata": job.metadata
        }
    
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job result if completed."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None
        
        return JobResult(
            success=True,
            data=job.result,
            processing_time=(job.completed_at - job.started_at).total_seconds() if job.started_at and job.completed_at else 0.0,
            metadata=job.metadata
        )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Cancel running task
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            del self.running_jobs[job_id]
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.stats["cancelled_jobs"] += 1
        
        # Persist changes
        if self.enable_persistence:
            self._persist_jobs()
        
        self.logger.info(f"Cancelled job {job_id}")
        return True
    
    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.FAILED:
            return False
        
        if job.retry_count >= job.max_retries:
            self.logger.warning(f"Job {job_id} exceeded max retries")
            return False
        
        # Reset job for retry
        job.status = JobStatus.PENDING
        job.started_at = None
        job.completed_at = None
        job.progress = 0.0
        job.error = None
        job.retry_count += 1
        
        # Re-queue job
        self.job_queue.put(job)
        
        # Persist changes
        if self.enable_persistence:
            self._persist_jobs()
        
        self.logger.info(f"Retrying job {job_id} (attempt {job.retry_count + 1})")
        return True
    
    async def _start_processing(self):
        """Start job processing loop."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Started job processing")
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Get next job from queue
                    job = self.job_queue.get_nowait()
                    
                    # Check if we have capacity
                    if len(self.running_jobs) >= self.max_workers:
                        # Re-queue job and wait
                        self.job_queue.put(job)
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Start processing job
                    asyncio.create_task(self._process_job(job))
                    
                except Empty:
                    # No jobs in queue, wait a bit
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error in job processing loop: {e}")
                    await asyncio.sleep(1)
        
        finally:
            self.running = False
            self.logger.info("Stopped job processing")
    
    async def _process_job(self, job: Job):
        """Process a single job."""
        job_id = job.id
        
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Get processor
            processor = self.processors.get(job.type)
            if not processor:
                raise ValueError(f"No processor registered for job type: {job.type}")
            
            # Create processing task
            task = asyncio.create_task(
                self._run_processor(processor, job)
            )
            self.running_jobs[job_id] = task
            
            # Wait for completion with timeout
            try:
                result = await asyncio.wait_for(task, timeout=job.timeout_seconds)
                
                # Update job with result
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.progress = 100.0
                job.result = result.data if result.success else None
                job.error = result.error
                
                # Update statistics
                self.stats["completed_jobs"] += 1
                processing_time = result.processing_time
                self.stats["total_processing_time"] += processing_time
                self.stats["average_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["completed_jobs"]
                )
                
                self.logger.info(f"Completed job {job_id} in {processing_time:.2f}s")
                
            except asyncio.TimeoutError:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = f"Job timeout after {job.timeout_seconds}s"
                self.stats["failed_jobs"] += 1
                self.logger.error(f"Job {job_id} timed out")
            
            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
                self.stats["failed_jobs"] += 1
                self.logger.error(f"Job {job_id} failed: {e}")
            
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            # Persist changes
            if self.enable_persistence:
                self._persist_jobs()
    
    async def _run_processor(self, processor: Callable, job: Job) -> JobResult:
        """Run job processor with progress tracking."""
        start_time = time.time()
        
        try:
            # Check if processor is async
            if asyncio.iscoroutinefunction(processor):
                result = await processor(job.data, job.metadata)
            else:
                # Run in thread pool for sync processors
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, 
                    processor, 
                    job.data, 
                    job.metadata
                )
            
            processing_time = time.time() - start_time
            
            return JobResult(
                success=True,
                data=result,
                processing_time=processing_time,
                metadata=job.metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return JobResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata=job.metadata
            )
    
    def _persist_jobs(self):
        """Persist jobs to disk."""
        try:
            # Only persist pending and running jobs
            jobs_to_persist = {
                job_id: job for job_id, job in self.jobs.items()
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(jobs_to_persist, f)
                
        except Exception as e:
            self.logger.error(f"Failed to persist jobs: {e}")
    
    def _load_persisted_jobs(self):
        """Load persisted jobs from disk."""
        try:
            if not Path(self.persistence_path).exists():
                return
            
            with open(self.persistence_path, 'rb') as f:
                persisted_jobs = pickle.load(f)
            
            # Restore jobs
            for job_id, job in persisted_jobs.items():
                # Reset running jobs to pending
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.PENDING
                    job.started_at = None
                
                self.jobs[job_id] = job
                self.job_queue.put(job)
            
            self.logger.info(f"Loaded {len(persisted_jobs)} persisted jobs")
            
        except Exception as e:
            self.logger.error(f"Failed to load persisted jobs: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get job manager statistics."""
        return {
            "stats": self.stats.copy(),
            "active_jobs": len(self.running_jobs),
            "queued_jobs": self.job_queue.qsize(),
            "total_jobs": len(self.jobs),
            "registered_processors": list(self.processors.keys()),
            "max_workers": self.max_workers,
            "running": self.running
        }
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = [
            job_id for job_id, job in self.jobs.items()
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            and job.completed_at and job.completed_at < cutoff_time
        ]
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
            
            if self.enable_persistence:
                self._persist_jobs()
    
    async def shutdown(self):
        """Shutdown job manager."""
        self.logger.info("Shutting down job manager...")
        
        # Stop accepting new jobs
        self.running = False
        self.shutdown_event.set()
        
        # Wait for running jobs to complete
        if self.running_jobs:
            self.logger.info(f"Waiting for {len(self.running_jobs)} running jobs to complete...")
            await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Final persistence
        if self.enable_persistence:
            self._persist_jobs()
        
        self.logger.info("Job manager shutdown complete")


