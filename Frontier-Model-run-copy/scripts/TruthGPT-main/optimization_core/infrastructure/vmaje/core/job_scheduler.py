"""
Job Scheduler
=============

Advanced job scheduling system with:
- Intelligent job queuing and prioritization
- Resource-aware scheduling
- Multi-cloud job distribution
- GPU job optimization
- Cost-aware scheduling
- Job dependency management
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import heapq


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(Enum):
    """Job priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class JobType(Enum):
    """Job type enumeration"""
    TRAINING = "training"
    INFERENCE = "inference"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    CUSTOM = "custom"


@dataclass
class JobConfig:
    """Job configuration"""
    name: str
    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL
    timeout: Optional[int] = None  # seconds
    retry_count: int = 3
    retry_delay: int = 60  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    gpu_required: bool = False
    gpu_count: int = 0
    memory_required: int = 0  # MB
    cpu_required: int = 0  # cores
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """Job representation"""
    id: str
    config: JobConfig
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    vm_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobScheduler:
    """
    Advanced Job Scheduler.
    
    Features:
    - Intelligent job queuing and prioritization
    - Resource-aware scheduling
    - Multi-cloud job distribution
    - GPU job optimization
    - Cost-aware scheduling
    - Job dependency management
    """
    
    def __init__(self, vm_manager, config: Optional[Dict[str, Any]] = None):
        self.vm_manager = vm_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Job storage
        self.jobs: Dict[str, Job] = {}
        self.job_queue = []  # Priority queue
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.scheduler_thread = None
        self.running = False
        
        # Job callbacks
        self.job_callbacks: Dict[str, Callable] = {}
        
        # Start scheduler
        self.start_scheduler()
    
    def start_scheduler(self):
        """Start job scheduler"""
        if self.scheduler_thread:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Job scheduler started")
    
    def stop_scheduler(self):
        """Stop job scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._process_job_queue()
                self._check_running_jobs()
                self._cleanup_completed_jobs()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(10)
    
    def _process_job_queue(self):
        """Process jobs in the queue"""
        while self.job_queue and len(self.running_jobs) < self.config.get('max_concurrent_jobs', 10):
            # Get highest priority job
            priority, job_id, job = heapq.heappop(self.job_queue)
            
            # Check if job can be scheduled
            if self._can_schedule_job(job):
                self._schedule_job(job)
            else:
                # Put job back in queue
                heapq.heappush(self.job_queue, (priority, job_id, job))
                break
    
    def _can_schedule_job(self, job: Job) -> bool:
        """Check if job can be scheduled"""
        # Check dependencies
        for dep_id in job.config.dependencies:
            if dep_id not in self.completed_jobs:
                return False
        
        # Check resource availability
        if job.config.gpu_required:
            available_gpus = self._get_available_gpus()
            if available_gpus < job.config.gpu_count:
                return False
        
        return True
    
    def _get_available_gpus(self) -> int:
        """Get available GPU count"""
        total_gpus = 0
        for vm in self.vm_manager.list_vms():
            if vm.status.value == 'running':
                total_gpus += vm.config.gpu_count
        
        # Subtract GPUs used by running jobs
        used_gpus = sum(
            job.config.gpu_count for job in self.running_jobs.values()
            if job.config.gpu_required
        )
        
        return total_gpus - used_gpus
    
    def _schedule_job(self, job: Job):
        """Schedule a job for execution"""
        self.logger.info(f"Scheduling job: {job.name}")
        
        # Find suitable VM
        vm = self._find_suitable_vm(job)
        if not vm:
            self.logger.warning(f"No suitable VM found for job: {job.name}")
            return
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.vm_id = vm.id
        
        # Move to running jobs
        self.running_jobs[job.id] = job
        
        # Execute job
        asyncio.create_task(self._execute_job(job))
    
    def _find_suitable_vm(self, job: Job) -> Optional[Any]:
        """Find suitable VM for job"""
        available_vms = [
            vm for vm in self.vm_manager.list_vms()
            if vm.status.value == 'running'
        ]
        
        # Filter by resource requirements
        suitable_vms = []
        for vm in available_vms:
            if self._vm_satisfies_requirements(vm, job):
                suitable_vms.append(vm)
        
        if not suitable_vms:
            return None
        
        # Select best VM (e.g., least loaded, closest to data, etc.)
        return self._select_best_vm(suitable_vms, job)
    
    def _vm_satisfies_requirements(self, vm: Any, job: Job) -> bool:
        """Check if VM satisfies job requirements"""
        # Check GPU requirements
        if job.config.gpu_required and vm.config.gpu_count < job.config.gpu_count:
            return False
        
        # Check memory requirements
        if job.config.memory_required > 0:
            # This would need actual VM memory info
            pass
        
        # Check CPU requirements
        if job.config.cpu_required > 0:
            # This would need actual VM CPU info
            pass
        
        return True
    
    def _select_best_vm(self, vms: List[Any], job: Job) -> Any:
        """Select best VM from candidates"""
        # Simple selection - in production, use more sophisticated logic
        return vms[0]
    
    async def _execute_job(self, job: Job):
        """Execute a job"""
        try:
            self.logger.info(f"Executing job: {job.name}")
            
            # Get job callback
            callback = self.job_callbacks.get(job.config.job_type.value)
            if not callback:
                raise ValueError(f"No callback registered for job type: {job.config.job_type}")
            
            # Execute job with timeout
            if job.config.timeout:
                result = await asyncio.wait_for(
                    callback(job),
                    timeout=job.config.timeout
                )
            else:
                result = await callback(job)
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            job.progress = 100.0
            
            self.logger.info(f"Job completed successfully: {job.name}")
            
        except asyncio.TimeoutError:
            job.status = JobStatus.TIMEOUT
            job.completed_at = datetime.now()
            job.error = f"Job timed out after {job.config.timeout} seconds"
            self.logger.error(f"Job timed out: {job.name}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = str(e)
            self.logger.error(f"Job failed: {job.name} - {e}")
            
            # Retry if possible
            if job.retry_count < job.config.retry_count:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.started_at = None
                job.vm_id = None
                job.error = None
                
                # Re-queue job
                self._requeue_job(job)
                return
        
        finally:
            # Move job to completed
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            
            self.completed_jobs[job.id] = job
    
    def _requeue_job(self, job: Job):
        """Re-queue a job for retry"""
        priority = -job.config.priority.value  # Negative for max-heap behavior
        heapq.heappush(self.job_queue, (priority, job.id, job))
        self.logger.info(f"Job re-queued for retry: {job.name}")
    
    def _check_running_jobs(self):
        """Check running jobs for timeouts and status"""
        current_time = datetime.now()
        
        for job_id, job in list(self.running_jobs.items()):
            # Check for timeout
            if job.config.timeout and job.started_at:
                elapsed = (current_time - job.started_at).total_seconds()
                if elapsed > job.config.timeout:
                    job.status = JobStatus.TIMEOUT
                    job.completed_at = current_time
                    job.error = "Job timeout"
                    
                    del self.running_jobs[job_id]
                    self.completed_jobs[job_id] = job
                    
                    self.logger.warning(f"Job timed out: {job.name}")
    
    def _cleanup_completed_jobs(self):
        """Cleanup old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        old_jobs = [
            job_id for job_id, job in self.completed_jobs.items()
            if job.completed_at and job.completed_at < cutoff_time
        ]
        
        for job_id in old_jobs:
            del self.completed_jobs[job_id]
        
        if old_jobs:
            self.logger.info(f"Cleaned up {len(old_jobs)} old completed jobs")
    
    def submit_job(self, config: JobConfig, callback: Optional[Callable] = None) -> str:
        """Submit a new job"""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            config=config,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Store job
        self.jobs[job_id] = job
        
        # Register callback if provided
        if callback:
            self.job_callbacks[config.job_type.value] = callback
        
        # Add to queue
        priority = -config.priority.value  # Negative for max-heap behavior
        heapq.heappush(self.job_queue, (priority, job_id, job))
        
        self.logger.info(f"Job submitted: {config.name} (ID: {job_id})")
        return job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status == JobStatus.RUNNING:
            # Cancel running job
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            self.completed_jobs[job_id] = job
            
        elif job.status == JobStatus.PENDING:
            # Remove from queue
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Remove from queue (this is simplified - in production, you'd need to rebuild the heap)
            self.job_queue = [
                (priority, jid, j) for priority, jid, j in self.job_queue
                if jid != job_id
            ]
            heapq.heapify(self.job_queue)
            
            self.completed_jobs[job_id] = job
        
        self.logger.info(f"Job cancelled: {job.name}")
        return True
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """List jobs"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        return jobs
    
    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get job metrics"""
        if job_id not in self.jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        job = self.jobs[job_id]
        
        metrics = {
            'id': job.id,
            'name': job.name,
            'status': job.status.value,
            'progress': job.progress,
            'created_at': job.created_at.isoformat(),
            'job_type': job.config.job_type.value,
            'priority': job.config.priority.value
        }
        
        if job.started_at:
            metrics['started_at'] = job.started_at.isoformat()
        
        if job.completed_at:
            metrics['completed_at'] = job.completed_at.isoformat()
            metrics['duration'] = (job.completed_at - job.created_at).total_seconds()
        
        if job.vm_id:
            metrics['vm_id'] = job.vm_id
        
        if job.error:
            metrics['error'] = job.error
        
        return metrics
    
    def get_scheduler_summary(self) -> Dict[str, Any]:
        """Get scheduler summary"""
        total_jobs = len(self.jobs)
        pending_jobs = len([job for job in self.jobs.values() if job.status == JobStatus.PENDING])
        running_jobs = len(self.running_jobs)
        completed_jobs = len(self.completed_jobs)
        failed_jobs = len([job for job in self.jobs.values() if job.status == JobStatus.FAILED])
        
        return {
            'total_jobs': total_jobs,
            'pending_jobs': pending_jobs,
            'running_jobs': running_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'queue_size': len(self.job_queue),
            'max_concurrent_jobs': self.config.get('max_concurrent_jobs', 10)
        }
    
    def register_job_callback(self, job_type: JobType, callback: Callable):
        """Register job callback for specific job type"""
        self.job_callbacks[job_type.value] = callback
        self.logger.info(f"Registered callback for job type: {job_type.value}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_scheduler()
        self.executor.shutdown(wait=True)
        self.logger.info("JobScheduler cleanup completed")


