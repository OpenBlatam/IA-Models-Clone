"""
Batch Processor for Ultimate Opus Clip

Advanced batch processing system that efficiently processes multiple videos
in parallel with intelligent resource management and progress tracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import torch
from queue import Queue, Empty
import threading

logger = structlog.get_logger("batch_processor")

class BatchStatus(Enum):
    """Status of batch processing jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingPriority(Enum):
    """Priority levels for batch processing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class BatchJob:
    """A single job in a batch."""
    job_id: str
    video_path: str
    config: Dict[str, Any]
    priority: ProcessingPriority
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent_jobs: int = 4
    max_queue_size: int = 1000
    timeout_seconds: int = 3600  # 1 hour
    retry_attempts: int = 3
    retry_delay: float = 5.0
    enable_gpu: bool = True
    enable_caching: bool = True
    progress_update_interval: float = 1.0
    cleanup_on_completion: bool = True

class BatchProcessor:
    """Advanced batch processing system."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.progress_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        logger.info("Batch processor initialized")
    
    def start(self):
        """Start the batch processor."""
        if self.running:
            logger.warning("Batch processor is already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.config.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start progress monitoring thread
        self.progress_thread = threading.Thread(
            target=self._progress_loop,
            name="ProgressMonitor",
            daemon=True
        )
        self.progress_thread.start()
        
        logger.info(f"Batch processor started with {self.config.max_concurrent_jobs} workers")
    
    def stop(self):
        """Stop the batch processor."""
        if not self.running:
            logger.warning("Batch processor is not running")
            return
        
        self.running = False
        
        # Wait for worker threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        # Wait for progress thread
        if self.progress_thread:
            self.progress_thread.join(timeout=5)
        
        logger.info("Batch processor stopped")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                # Get next job from queue
                job = self.job_queue.get(timeout=1)
                
                # Process the job
                self._process_job(job)
                
            except Empty:
                # No jobs available, continue
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def _process_job(self, job: BatchJob):
        """Process a single job."""
        try:
            with self.lock:
                job.status = BatchStatus.PROCESSING
                job.started_at = time.time()
                self.active_jobs[job.job_id] = job
            
            logger.info(f"Processing job {job.job_id}: {job.video_path}")
            
            # Simulate processing (replace with actual processing logic)
            result = self._execute_processing(job)
            
            with self.lock:
                job.status = BatchStatus.COMPLETED
                job.completed_at = time.time()
                job.result = result
                job.progress = 100.0
                
                # Move to completed jobs
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                self.completed_jobs[job.job_id] = job
                
                # Update statistics
                processing_time = job.completed_at - job.started_at
                self.stats["completed_jobs"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["average_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["completed_jobs"]
                )
            
            logger.info(f"Completed job {job.job_id} in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {e}")
            
            with self.lock:
                job.status = BatchStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                
                # Move to failed jobs
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                self.failed_jobs[job.job_id] = job
                
                self.stats["failed_jobs"] += 1
    
    def _execute_processing(self, job: BatchJob) -> Any:
        """Execute the actual processing for a job."""
        # This is a placeholder - replace with actual processing logic
        # For now, simulate processing time based on video length
        
        # Simulate processing time
        processing_time = min(30, max(5, np.random.normal(15, 5)))
        
        # Simulate progress updates
        for progress in range(0, 101, 10):
            if not self.running:
                break
            
            with self.lock:
                if job.job_id in self.active_jobs:
                    job.progress = progress
            
            time.sleep(processing_time / 10)
        
        # Return mock result
        return {
            "job_id": job.job_id,
            "video_path": job.video_path,
            "clips_generated": np.random.randint(1, 6),
            "processing_time": processing_time,
            "quality_score": np.random.uniform(0.7, 1.0),
            "viral_potential": np.random.uniform(0.6, 0.95)
        }
    
    def _progress_loop(self):
        """Progress monitoring loop."""
        while self.running:
            try:
                with self.lock:
                    active_count = len(self.active_jobs)
                    completed_count = len(self.completed_jobs)
                    failed_count = len(self.failed_jobs)
                    total_count = len(self.jobs)
                
                if active_count > 0 or completed_count > 0 or failed_count > 0:
                    logger.info(
                        f"Batch progress: {completed_count}/{total_count} completed, "
                        f"{active_count} active, {failed_count} failed"
                    )
                
                time.sleep(self.config.progress_update_interval)
                
            except Exception as e:
                logger.error(f"Error in progress loop: {e}")
                time.sleep(5)
    
    def add_job(self, video_path: str, config: Dict[str, Any], 
                priority: ProcessingPriority = ProcessingPriority.NORMAL) -> str:
        """Add a job to the batch queue."""
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            video_path=video_path,
            config=config,
            priority=priority,
            created_at=time.time()
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.stats["total_jobs"] += 1
        
        try:
            self.job_queue.put(job, timeout=1)
            logger.info(f"Added job {job_id} to queue")
        except Exception as e:
            logger.error(f"Failed to add job {job_id} to queue: {e}")
            with self.lock:
                job.status = BatchStatus.FAILED
                job.error = "Failed to add to queue"
                self.failed_jobs[job_id] = job
        
        return job_id
    
    def add_batch(self, video_paths: List[str], config: Dict[str, Any],
                  priority: ProcessingPriority = ProcessingPriority.NORMAL) -> List[str]:
        """Add multiple jobs to the batch queue."""
        job_ids = []
        
        for video_path in video_paths:
            job_id = self.add_job(video_path, config, priority)
            job_ids.append(job_id)
        
        logger.info(f"Added batch of {len(video_paths)} jobs")
        return job_ids
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific job."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "error": job.error,
                    "result": job.result
                }
        return None
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get the status of the entire batch."""
        with self.lock:
            return {
                "total_jobs": len(self.jobs),
                "pending_jobs": len([j for j in self.jobs.values() if j.status == BatchStatus.PENDING]),
                "active_jobs": len(self.active_jobs),
                "completed_jobs": len(self.completed_jobs),
                "failed_jobs": len(self.failed_jobs),
                "cancelled_jobs": len([j for j in self.jobs.values() if j.status == BatchStatus.CANCELLED]),
                "stats": self.stats.copy(),
                "running": self.running
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == BatchStatus.PENDING:
                    job.status = BatchStatus.CANCELLED
                    self.stats["cancelled_jobs"] += 1
                    logger.info(f"Cancelled job {job_id}")
                    return True
                elif job.status == BatchStatus.PROCESSING:
                    # Mark for cancellation (actual cancellation would need more complex logic)
                    job.status = BatchStatus.CANCELLED
                    self.stats["cancelled_jobs"] += 1
                    logger.info(f"Marked job {job_id} for cancellation")
                    return True
        return False
    
    def cancel_all_jobs(self) -> int:
        """Cancel all pending jobs."""
        cancelled_count = 0
        
        with self.lock:
            for job in self.jobs.values():
                if job.status == BatchStatus.PENDING:
                    job.status = BatchStatus.CANCELLED
                    cancelled_count += 1
        
        self.stats["cancelled_jobs"] += cancelled_count
        logger.info(f"Cancelled {cancelled_count} jobs")
        return cancelled_count
    
    def get_completed_results(self) -> List[Dict[str, Any]]:
        """Get results from all completed jobs."""
        with self.lock:
            return [
                {
                    "job_id": job.job_id,
                    "video_path": job.video_path,
                    "result": job.result,
                    "processing_time": job.completed_at - job.started_at if job.started_at and job.completed_at else 0
                }
                for job in self.completed_jobs.values()
            ]
    
    def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get information about failed jobs."""
        with self.lock:
            return [
                {
                    "job_id": job.job_id,
                    "video_path": job.video_path,
                    "error": job.error,
                    "failed_at": job.completed_at
                }
                for job in self.failed_jobs.values()
            ]
    
    def retry_failed_jobs(self) -> List[str]:
        """Retry all failed jobs."""
        retry_job_ids = []
        
        with self.lock:
            for job in list(self.failed_jobs.values()):
                # Reset job status
                job.status = BatchStatus.PENDING
                job.error = None
                job.started_at = None
                job.completed_at = None
                job.progress = 0.0
                
                # Move back to jobs
                self.jobs[job.job_id] = job
                del self.failed_jobs[job.job_id]
                
                # Add to queue
                try:
                    self.job_queue.put(job, timeout=1)
                    retry_job_ids.append(job.job_id)
                except Exception as e:
                    logger.error(f"Failed to retry job {job.job_id}: {e}")
        
        logger.info(f"Retrying {len(retry_job_ids)} failed jobs")
        return retry_job_ids
    
    def cleanup(self):
        """Cleanup batch processor resources."""
        self.stop()
        
        with self.lock:
            self.jobs.clear()
            self.active_jobs.clear()
            self.completed_jobs.clear()
            self.failed_jobs.clear()
            
            # Clear queue
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                except Empty:
                    break
        
        logger.info("Batch processor cleaned up")

# Global batch processor instance
_global_batch_processor: Optional[BatchProcessor] = None

def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    return _global_batch_processor

def start_batch_processing():
    """Start the global batch processor."""
    processor = get_batch_processor()
    processor.start()

def stop_batch_processing():
    """Stop the global batch processor."""
    processor = get_batch_processor()
    processor.stop()


