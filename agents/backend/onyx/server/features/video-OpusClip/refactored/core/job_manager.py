"""
Job Manager

Advanced job management system with queuing, scheduling, monitoring,
and distributed processing capabilities.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = structlog.get_logger("job_manager")

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Job:
    """Job representation."""
    job_id: str
    job_type: str
    status: JobStatus
    priority: JobPriority
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    callback_url: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""

@dataclass
class JobResult:
    """Job execution result."""
    job_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class JobQueue:
    """Priority-based job queue."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues = {
            JobPriority.URGENT: asyncio.Queue(maxsize=max_size // 4),
            JobPriority.HIGH: asyncio.Queue(maxsize=max_size // 4),
            JobPriority.NORMAL: asyncio.Queue(maxsize=max_size // 2),
            JobPriority.LOW: asyncio.Queue(maxsize=max_size // 4)
        }
        self._lock = asyncio.Lock()
    
    async def put(self, job: Job) -> bool:
        """Add job to queue."""
        try:
            async with self._lock:
                if self._is_full():
                    return False
                
                await self._queues[job.priority].put(job)
                return True
                
        except Exception as e:
            logger.error(f"Failed to add job to queue: {e}")
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Get next job from queue."""
        try:
            # Check queues in priority order
            for priority in [JobPriority.URGENT, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
                try:
                    job = await asyncio.wait_for(
                        self._queues[priority].get(), 
                        timeout=timeout or 0.1
                    )
                    return job
                except asyncio.TimeoutError:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job from queue: {e}")
            return None
    
    def _is_full(self) -> bool:
        """Check if queue is full."""
        return all(queue.full() for queue in self._queues.values())
    
    def size(self) -> int:
        """Get total queue size."""
        return sum(queue.qsize() for queue in self._queues.values())
    
    def size_by_priority(self, priority: JobPriority) -> int:
        """Get queue size for specific priority."""
        return self._queues[priority].qsize()

class JobDatabase:
    """SQLite-based job persistence."""
    
    def __init__(self, db_path: str = "/tmp/jobs.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize job database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    timeout REAL DEFAULT 300.0,
                    metadata TEXT,
                    dependencies TEXT,
                    callback_url TEXT,
                    progress REAL DEFAULT 0.0,
                    progress_message TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize job database: {e}")
            raise
    
    def save_job(self, job: Job) -> bool:
        """Save job to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO jobs (
                    job_id, job_type, status, priority, input_data, output_data,
                    error_message, created_at, started_at, completed_at,
                    retry_count, max_retries, timeout, metadata, dependencies,
                    callback_url, progress, progress_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id, job.job_type, job.status.value, job.priority.value,
                json.dumps(job.input_data), 
                json.dumps(job.output_data) if job.output_data else None,
                job.error_message, job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.retry_count, job.max_retries, job.timeout,
                json.dumps(job.metadata), json.dumps(job.dependencies),
                job.callback_url, job.progress, job.progress_message
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save job: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return self._row_to_job(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            return None
    
    def update_job(self, job: Job) -> bool:
        """Update job in database."""
        return self.save_job(job)
    
    def list_jobs(self, 
                  status: Optional[JobStatus] = None,
                  job_type: Optional[str] = None,
                  limit: int = 100,
                  offset: int = 0) -> List[Job]:
        """List jobs with filters."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM jobs WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if job_type:
                query += " AND job_type = ?"
                params.append(job_type)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            return [self._row_to_job(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def _row_to_job(self, row: tuple) -> Job:
        """Convert database row to Job object."""
        return Job(
            job_id=row[0],
            job_type=row[1],
            status=JobStatus(row[2]),
            priority=JobPriority(row[3]),
            input_data=json.loads(row[4]),
            output_data=json.loads(row[5]) if row[5] else None,
            error_message=row[6],
            created_at=datetime.fromisoformat(row[7]),
            started_at=datetime.fromisoformat(row[8]) if row[8] else None,
            completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
            retry_count=row[10],
            max_retries=row[11],
            timeout=row[12],
            metadata=json.loads(row[13]),
            dependencies=json.loads(row[14]),
            callback_url=row[15],
            progress=row[16],
            progress_message=row[17]
        )

class JobManager:
    """Advanced job management system."""
    
    def __init__(self, 
                 max_workers: int = 4,
                 db_path: str = "/tmp/jobs.db",
                 cleanup_interval: float = 3600.0):  # 1 hour
        self.max_workers = max_workers
        self.job_queue = JobQueue()
        self.job_db = JobDatabase(db_path)
        self.cleanup_interval = cleanup_interval
        
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.processors: Dict[str, Callable] = {}
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self.logger = structlog.get_logger("job_manager")
    
    async def start(self):
        """Start the job manager."""
        try:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Job manager started")
            
        except Exception as e:
            self.logger.error(f"Failed to start job manager: {e}")
            raise
    
    async def stop(self):
        """Stop the job manager."""
        try:
            self._running = False
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            self._executor.shutdown(wait=True)
            self.logger.info("Job manager stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop job manager: {e}")
    
    def register_processor(self, job_type: str, processor: Callable):
        """Register a job processor."""
        self.processors[job_type] = processor
        self.logger.info(f"Registered processor for job type: {job_type}")
    
    async def submit_job(self, 
                        job_type: str,
                        input_data: Dict[str, Any],
                        priority: JobPriority = JobPriority.NORMAL,
                        timeout: float = 300.0,
                        max_retries: int = 3,
                        dependencies: List[str] = None,
                        callback_url: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """Submit a new job."""
        try:
            job_id = str(uuid.uuid4())
            
            job = Job(
                job_id=job_id,
                job_type=job_type,
                status=JobStatus.PENDING,
                priority=priority,
                input_data=input_data,
                timeout=timeout,
                max_retries=max_retries,
                dependencies=dependencies or [],
                callback_url=callback_url,
                metadata=metadata or {}
            )
            
            # Save to database
            if not self.job_db.save_job(job):
                raise Exception("Failed to save job to database")
            
            # Add to queue
            if not await self.job_queue.put(job):
                raise Exception("Job queue is full")
            
            job.status = JobStatus.QUEUED
            self.job_db.update_job(job)
            
            self.logger.info(f"Job submitted: {job_id} ({job_type})")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        try:
            job = self.job_db.get_job(job_id)
            if not job:
                return None
            
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "priority": job.priority.value,
                "progress": job.progress,
                "progress_message": job.progress_message,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "retry_count": job.retry_count,
                "error_message": job.error_message,
                "metadata": job.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            job = self.job_db.get_job(job_id)
            if not job:
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self.job_db.update_job(job)
            self.logger.info(f"Job cancelled: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job: {e}")
            return False
    
    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        try:
            job = self.job_db.get_job(job_id)
            if not job:
                return False
            
            if job.status != JobStatus.FAILED:
                return False
            
            if job.retry_count >= job.max_retries:
                return False
            
            job.status = JobStatus.PENDING
            job.retry_count += 1
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            job.progress = 0.0
            job.progress_message = ""
            
            if not await self.job_queue.put(job):
                return False
            
            job.status = JobStatus.QUEUED
            self.job_db.update_job(job)
            
            self.logger.info(f"Job retried: {job_id} (attempt {job.retry_count})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retry job: {e}")
            return False
    
    async def process_jobs(self):
        """Process jobs from queue."""
        while self._running:
            try:
                job = await self.job_queue.get(timeout=1.0)
                if not job:
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(job):
                    # Re-queue job for later
                    await asyncio.sleep(5.0)
                    await self.job_queue.put(job)
                    continue
                
                # Process job
                asyncio.create_task(self._process_job(job))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in job processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_dependencies(self, job: Job) -> bool:
        """Check if job dependencies are satisfied."""
        if not job.dependencies:
            return True
        
        for dep_id in job.dependencies:
            dep_job = self.job_db.get_job(dep_id)
            if not dep_job or dep_job.status != JobStatus.COMPLETED:
                return False
        
        return True
    
    async def _process_job(self, job: Job):
        """Process a single job."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self.active_jobs[job.job_id] = job
            self.job_db.update_job(job)
            
            self.logger.info(f"Processing job: {job.job_id} ({job.job_type})")
            
            # Get processor
            processor = self.processors.get(job.job_type)
            if not processor:
                raise Exception(f"No processor registered for job type: {job.job_type}")
            
            # Process with timeout
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    processor(job.input_data),
                    timeout=job.timeout
                )
                
                processing_time = time.time() - start_time
                
                # Update job with result
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.progress = 100.0
                job.progress_message = "Completed successfully"
                job.output_data = result if isinstance(result, dict) else {"result": result}
                
                self.logger.info(f"Job completed: {job.job_id} in {processing_time:.2f}s")
                
            except asyncio.TimeoutError:
                raise Exception(f"Job timeout after {job.timeout} seconds")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            job.progress_message = f"Failed: {str(e)}"
            
            self.logger.error(f"Job failed: {job.job_id} - {e}")
            
            # Retry if possible
            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                await asyncio.sleep(2 ** job.retry_count)  # Exponential backoff
                await self.retry_job(job.job_id)
                return
        
        finally:
            # Update job in database
            self.job_db.update_job(job)
            
            # Move to completed jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            self.completed_jobs[job.job_id] = job
            
            # Call callback if provided
            if job.callback_url:
                await self._call_callback(job)
    
    async def _call_callback(self, job: Job):
        """Call job completion callback."""
        try:
            import aiohttp
            
            callback_data = {
                "job_id": job.job_id,
                "status": job.status.value,
                "result": job.output_data,
                "error": job.error_message,
                "processing_time": (job.completed_at - job.started_at).total_seconds() if job.started_at and job.completed_at else 0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(job.callback_url, json=callback_data) as response:
                    if response.status == 200:
                        self.logger.info(f"Callback called successfully for job: {job.job_id}")
                    else:
                        self.logger.warning(f"Callback failed for job: {job.job_id} - Status: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to call callback for job {job.job_id}: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old completed jobs."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Remove old completed jobs from memory
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_jobs = [
                    job_id for job_id, job in self.completed_jobs.items()
                    if job.completed_at and job.completed_at < cutoff_time
                ]
                
                for job_id in old_jobs:
                    del self.completed_jobs[job_id]
                
                if old_jobs:
                    self.logger.info(f"Cleaned up {len(old_jobs)} old jobs")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get job manager statistics."""
        return {
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "queue_size": self.job_queue.size(),
            "queue_by_priority": {
                priority.name: self.job_queue.size_by_priority(priority)
                for priority in JobPriority
            },
            "registered_processors": list(self.processors.keys()),
            "running": self._running
        }

# Export classes
__all__ = [
    "JobManager",
    "Job",
    "JobStatus", 
    "JobPriority",
    "JobResult",
    "JobQueue",
    "JobDatabase"
]


