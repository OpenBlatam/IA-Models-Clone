"""
Refactored Job Manager for Final Ultimate AI System

Enhanced job management with:
- Advanced job scheduling and prioritization
- Resource allocation and management
- Job dependency management
- Retry mechanisms and error handling
- Performance monitoring and optimization
- Load balancing and auto-scaling
- Job queuing and batching
- Real-time job tracking and notifications
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import heapq
import threading
from collections import defaultdict, deque
import weakref
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = structlog.get_logger("refactored_job_manager")

T = TypeVar('T')

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    PAUSED = "paused"

class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class JobType(Enum):
    """Job type enumeration."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    GPU_INTENSIVE = "gpu_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED = "mixed"

class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class JobResource:
    """Job resource requirements."""
    resource_type: ResourceType
    required: float
    allocated: float = 0.0
    unit: str = "cores"  # cores, MB, GB, etc.

@dataclass
class JobDependency:
    """Job dependency information."""
    job_id: str
    dependency_type: str  # "must_complete", "must_succeed", "must_fail"
    timeout: Optional[int] = None

@dataclass
class JobRetryConfig:
    """Job retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[str] = field(default_factory=list)

@dataclass
class Job:
    """Job data structure."""
    job_id: str
    job_type: JobType
    priority: JobPriority
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    function: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    resources: List[JobResource] = field(default_factory=list)
    dependencies: List[JobDependency] = field(default_factory=list)
    retry_config: Optional[JobRetryConfig] = None
    timeout: Optional[int] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    progress_message: str = ""

@dataclass
class JobMetrics:
    """Job metrics data structure."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    average_execution_time: float = 0.0
    average_wait_time: float = 0.0
    throughput: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class ResourceManager:
    """Resource management and allocation system."""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: mp.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            ResourceType.GPU: 0,  # Will be detected
            ResourceType.DISK: psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
            ResourceType.NETWORK: 1000  # Mbps (placeholder)
        }
        
        self.allocated_resources: Dict[ResourceType, float] = {
            resource_type: 0.0 for resource_type in ResourceType
        }
        
        self._lock = threading.Lock()
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get available resources."""
        with self._lock:
            return {
                resource_type: self.available_resources[resource_type] - self.allocated_resources[resource_type]
                for resource_type in ResourceType
            }
    
    def can_allocate(self, resources: List[JobResource]) -> bool:
        """Check if resources can be allocated."""
        with self._lock:
            available = self.get_available_resources()
            
            for resource in resources:
                if available[resource.resource_type] < resource.required:
                    return False
            
            return True
    
    def allocate_resources(self, resources: List[JobResource]) -> bool:
        """Allocate resources for a job."""
        with self._lock:
            if not self.can_allocate(resources):
                return False
            
            for resource in resources:
                self.allocated_resources[resource.resource_type] += resource.required
                resource.allocated = resource.required
            
            return True
    
    def deallocate_resources(self, resources: List[JobResource]) -> None:
        """Deallocate resources from a job."""
        with self._lock:
            for resource in resources:
                self.allocated_resources[resource.resource_type] -= resource.allocated
                resource.allocated = 0.0
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get resource utilization percentage."""
        with self._lock:
            return {
                resource_type: (
                    self.allocated_resources[resource_type] / 
                    self.available_resources[resource_type] * 100
                )
                for resource_type in ResourceType
            }

class JobScheduler:
    """Advanced job scheduler with priority and dependency management."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.job_queue: List[Job] = []
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.failed_jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
    
    def add_job(self, job: Job) -> bool:
        """Add a job to the scheduler."""
        with self._lock:
            # Check dependencies
            if not self._check_dependencies(job):
                return False
            
            # Add to queue
            self.job_queue.append(job)
            
            # Sort by priority (highest first)
            self.job_queue.sort(key=lambda j: j.priority.value, reverse=True)
            
            logger.info(f"Job {job.job_id} added to scheduler")
            return True
    
    def _check_dependencies(self, job: Job) -> bool:
        """Check if job dependencies are satisfied."""
        for dependency in job.dependencies:
            if dependency.dependency_type == "must_complete":
                if dependency.job_id not in self.completed_jobs:
                    return False
            elif dependency.dependency_type == "must_succeed":
                if (dependency.job_id not in self.completed_jobs or 
                    self.completed_jobs[dependency.job_id].status != JobStatus.COMPLETED):
                    return False
            elif dependency.dependency_type == "must_fail":
                if (dependency.job_id not in self.failed_jobs or 
                    self.failed_jobs[dependency.job_id].status != JobStatus.FAILED):
                    return False
        
        return True
    
    def get_next_job(self) -> Optional[Job]:
        """Get the next job to execute."""
        with self._lock:
            for i, job in enumerate(self.job_queue):
                # Check if resources are available
                if self.resource_manager.can_allocate(job.resources):
                    # Allocate resources
                    if self.resource_manager.allocate_resources(job.resources):
                        # Remove from queue and add to running
                        self.job_queue.pop(i)
                        self.running_jobs[job.job_id] = job
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now()
                        return job
            
            return None
    
    def complete_job(self, job_id: str, result: Any = None) -> None:
        """Mark a job as completed."""
        with self._lock:
            if job_id in self.running_jobs:
                job = self.running_jobs.pop(job_id)
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                self.completed_jobs[job_id] = job
                
                # Deallocate resources
                self.resource_manager.deallocate_resources(job.resources)
                
                logger.info(f"Job {job_id} completed")
    
    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        with self._lock:
            if job_id in self.running_jobs:
                job = self.running_jobs.pop(job_id)
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = error
                self.failed_jobs[job_id] = job
                
                # Deallocate resources
                self.resource_manager.deallocate_resources(job.resources)
                
                logger.info(f"Job {job_id} failed: {error}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            # Check if job is in queue
            for i, job in enumerate(self.job_queue):
                if job.job_id == job_id:
                    job = self.job_queue.pop(i)
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    return True
            
            # Check if job is running
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                
                # Deallocate resources
                self.resource_manager.deallocate_resources(job.resources)
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.running_jobs[job_id]
                
                return True
            
            return False
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        with self._lock:
            # Check running jobs
            if job_id in self.running_jobs:
                return self.running_jobs[job_id].status
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id].status
            
            # Check failed jobs
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id].status
            
            # Check queue
            for job in self.job_queue:
                if job.job_id == job_id:
                    return job.status
            
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information."""
        with self._lock:
            return {
                "queue_size": len(self.job_queue),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len(self.completed_jobs),
                "failed_jobs": len(self.failed_jobs),
                "resource_utilization": self.resource_manager.get_resource_utilization()
            }

class JobExecutor:
    """Job execution engine with retry and error handling."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()
    
    async def execute_job(self, job: Job) -> Any:
        """Execute a job with retry logic."""
        try:
            # Check if job has retry configuration
            retry_config = job.retry_config or JobRetryConfig()
            
            # Execute with retry
            result = await self._execute_with_retry(job, retry_config)
            return result
            
        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            raise e
    
    async def _execute_with_retry(self, job: Job, retry_config: JobRetryConfig) -> Any:
        """Execute job with retry logic."""
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Update retry count
                job.retry_count = attempt
                
                if attempt > 0:
                    job.status = JobStatus.RETRYING
                    delay = self._calculate_retry_delay(attempt, retry_config)
                    await asyncio.sleep(delay)
                
                # Execute job
                result = await self._execute_single_attempt(job)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e, retry_config):
                    break
                
                logger.warning(f"Job {job.job_id} attempt {attempt + 1} failed: {e}")
        
        # All retries failed
        raise last_exception
    
    def _calculate_retry_delay(self, attempt: int, retry_config: JobRetryConfig) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = min(
            retry_config.base_delay * (retry_config.exponential_base ** attempt),
            retry_config.max_delay
        )
        
        if retry_config.jitter:
            jitter = delay * 0.1 * (2 * np.random.random() - 1)  # ±10% jitter
            delay += jitter
        
        return delay
    
    def _should_retry(self, exception: Exception, retry_config: JobRetryConfig) -> bool:
        """Check if exception should trigger a retry."""
        if not retry_config.retry_on_exceptions:
            return True
        
        exception_type = type(exception).__name__
        return exception_type in retry_config.retry_on_exceptions
    
    async def _execute_single_attempt(self, job: Job) -> Any:
        """Execute a single job attempt."""
        try:
            # Set timeout if specified
            if job.timeout:
                result = await asyncio.wait_for(
                    self._run_job_function(job),
                    timeout=job.timeout
                )
            else:
                result = await self._run_job_function(job)
            
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Job {job.job_id} timed out after {job.timeout} seconds")
    
    async def _run_job_function(self, job: Job) -> Any:
        """Run the job function."""
        if job.function is None:
            raise Exception("No function specified for job")
        
        # Choose execution method based on job type
        if job.job_type == JobType.CPU_INTENSIVE:
            # Use process pool for CPU-intensive tasks
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_pool,
                job.function,
                *job.args,
                **job.kwargs
            )
        else:
            # Use thread pool for I/O-intensive tasks
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                job.function,
                *job.args,
                **job.kwargs
            )
        
        return result
    
    async def shutdown(self) -> None:
        """Shutdown the job executor."""
        try:
            # Cancel all active tasks
            for task in self.active_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
            
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("Job executor shutdown complete")
            
        except Exception as e:
            logger.error(f"Job executor shutdown error: {e}")

class RefactoredJobManager:
    """Refactored job manager with advanced features."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        
        # Initialize components
        self.resource_manager = ResourceManager()
        self.scheduler = JobScheduler(self.resource_manager)
        self.executor = JobExecutor(self.max_workers)
        
        # Job tracking
        self.jobs: Dict[str, Job] = {}
        self.job_metrics = JobMetrics()
        
        # Execution control
        self._running = False
        self._execution_task = None
        self._shutdown_event = asyncio.Event()
        
        # Notifications
        self._job_listeners: List[Callable] = []
        
        # Performance monitoring
        self._execution_times = deque(maxlen=1000)
        self._wait_times = deque(maxlen=1000)
        self._last_metrics_update = datetime.now()
    
    async def initialize(self) -> bool:
        """Initialize the job manager."""
        try:
            self._running = True
            self._execution_task = asyncio.create_task(self._execution_loop())
            
            logger.info("Refactored job manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Job manager initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the job manager gracefully."""
        try:
            self._running = False
            self._shutdown_event.set()
            
            # Wait for execution loop to complete
            if self._execution_task:
                await asyncio.wait_for(self._execution_task, timeout=30)
            
            # Shutdown executor
            await self.executor.shutdown()
            
            logger.info("Refactored job manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Job manager shutdown error: {e}")
    
    async def submit_job(self, job_id: str, function: Callable, job_type: JobType,
                        priority: JobPriority = JobPriority.NORMAL, 
                        resources: List[JobResource] = None,
                        dependencies: List[JobDependency] = None,
                        retry_config: JobRetryConfig = None,
                        timeout: int = None,
                        *args, **kwargs) -> str:
        """Submit a job for execution."""
        try:
            # Create job
            job = Job(
                job_id=job_id,
                job_type=job_type,
                priority=priority,
                function=function,
                args=args,
                kwargs=kwargs,
                resources=resources or [],
                dependencies=dependencies or [],
                retry_config=retry_config,
                timeout=timeout
            )
            
            # Add to job manager
            self.jobs[job_id] = job
            
            # Add to scheduler
            success = self.scheduler.add_job(job)
            if not success:
                raise Exception(f"Failed to schedule job {job_id}")
            
            # Notify listeners
            await self._notify_job_listeners(job, "submitted")
            
            logger.info(f"Job {job_id} submitted")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            raise e
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        return self.scheduler.get_job_status(job_id)
    
    async def get_job_result(self, job_id: str) -> Any:
        """Get job result."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status == JobStatus.COMPLETED:
                return job.result
            elif job.status == JobStatus.FAILED:
                raise Exception(f"Job {job_id} failed: {job.error}")
            else:
                raise Exception(f"Job {job_id} is not completed")
        else:
            raise Exception(f"Job {job_id} not found")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            success = self.scheduler.cancel_job(job_id)
            if success:
                await self._notify_job_listeners(self.jobs.get(job_id), "cancelled")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return self.scheduler.get_queue_status()
    
    async def get_metrics(self) -> JobMetrics:
        """Get job metrics."""
        # Update metrics
        await self._update_metrics()
        return self.job_metrics
    
    async def _execution_loop(self) -> None:
        """Main execution loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next job
                job = self.scheduler.get_next_job()
                
                if job:
                    # Execute job
                    await self._execute_job(job)
                else:
                    # No jobs available, wait a bit
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_job(self, job: Job) -> None:
        """Execute a single job."""
        try:
            start_time = time.time()
            
            # Execute job
            result = await self.executor.execute_job(job)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)
            
            # Calculate wait time
            if job.started_at and job.created_at:
                wait_time = (job.started_at - job.created_at).total_seconds()
                self._wait_times.append(wait_time)
            
            # Mark job as completed
            self.scheduler.complete_job(job.job_id, result)
            
            # Update job
            job.result = result
            job.progress = 100.0
            job.progress_message = "Completed"
            
            # Notify listeners
            await self._notify_job_listeners(job, "completed")
            
            logger.info(f"Job {job.job_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            # Mark job as failed
            self.scheduler.fail_job(job.job_id, str(e))
            
            # Update job
            job.error = str(e)
            job.progress_message = f"Failed: {str(e)}"
            
            # Notify listeners
            await self._notify_job_listeners(job, "failed")
            
            logger.error(f"Job {job.job_id} failed: {e}")
    
    async def _update_metrics(self) -> None:
        """Update job metrics."""
        try:
            now = datetime.now()
            
            # Count jobs by status
            total_jobs = len(self.jobs)
            completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED])
            failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
            cancelled_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.CANCELLED])
            
            # Update metrics
            self.job_metrics.total_jobs = total_jobs
            self.job_metrics.completed_jobs = completed_jobs
            self.job_metrics.failed_jobs = failed_jobs
            self.job_metrics.cancelled_jobs = cancelled_jobs
            
            # Calculate averages
            if self._execution_times:
                self.job_metrics.average_execution_time = sum(self._execution_times) / len(self._execution_times)
            
            if self._wait_times:
                self.job_metrics.average_wait_time = sum(self._wait_times) / len(self._wait_times)
            
            # Calculate throughput (jobs per second)
            time_diff = (now - self._last_metrics_update).total_seconds()
            if time_diff > 0:
                self.job_metrics.throughput = completed_jobs / time_diff
            
            # Calculate error rate
            if total_jobs > 0:
                self.job_metrics.error_rate = failed_jobs / total_jobs
            
            # Get resource utilization
            self.job_metrics.resource_utilization = self.resource_manager.get_resource_utilization()
            
            self.job_metrics.last_updated = now
            self._last_metrics_update = now
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    async def _notify_job_listeners(self, job: Job, event: str) -> None:
        """Notify job event listeners."""
        for listener in self._job_listeners:
            try:
                await listener(job, event)
            except Exception as e:
                logger.error(f"Job listener error: {e}")
    
    def add_job_listener(self, listener: Callable) -> None:
        """Add a job event listener."""
        self._job_listeners.append(listener)
    
    def remove_job_listener(self, listener: Callable) -> None:
        """Remove a job event listener."""
        if listener in self._job_listeners:
            self._job_listeners.remove(listener)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get job manager health status."""
        try:
            queue_status = await self.get_queue_status()
            metrics = await self.get_metrics()
            
            return {
                "healthy": True,
                "running": self._running,
                "queue_status": queue_status,
                "metrics": {
                    "total_jobs": metrics.total_jobs,
                    "completed_jobs": metrics.completed_jobs,
                    "failed_jobs": metrics.failed_jobs,
                    "error_rate": metrics.error_rate,
                    "throughput": metrics.throughput
                },
                "resource_utilization": metrics.resource_utilization
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}

# Example usage
async def example_job_function(data: str, multiplier: int = 2) -> str:
    """Example job function."""
    await asyncio.sleep(1)  # Simulate work
    return f"Processed: {data} * {multiplier} = {data * multiplier}"

async def main():
    """Example usage of refactored job manager."""
    job_manager = RefactoredJobManager(max_workers=4)
    
    # Initialize job manager
    success = await job_manager.initialize()
    if not success:
        print("Failed to initialize job manager")
        return
    
    # Add job listener
    async def job_listener(job: Job, event: str):
        print(f"Job {job.job_id} {event}")
    
    job_manager.add_job_listener(job_listener)
    
    # Submit some jobs
    job1_id = await job_manager.submit_job(
        job_id="job_1",
        function=example_job_function,
        job_type=JobType.CPU_INTENSIVE,
        priority=JobPriority.HIGH,
        resources=[
            JobResource(ResourceType.CPU, 1.0),
            JobResource(ResourceType.MEMORY, 100.0, unit="MB")
        ],
        args=("Hello",),
        kwargs={"multiplier": 3}
    )
    
    job2_id = await job_manager.submit_job(
        job_id="job_2",
        function=example_job_function,
        job_type=JobType.IO_INTENSIVE,
        priority=JobPriority.NORMAL,
        args=("World",),
        kwargs={"multiplier": 2}
    )
    
    # Wait for jobs to complete
    while True:
        status1 = await job_manager.get_job_status(job1_id)
        status2 = await job_manager.get_job_status(job2_id)
        
        if status1 == JobStatus.COMPLETED and status2 == JobStatus.COMPLETED:
            break
        
        await asyncio.sleep(0.1)
    
    # Get results
    result1 = await job_manager.get_job_result(job1_id)
    result2 = await job_manager.get_job_result(job2_id)
    
    print(f"Job 1 result: {result1}")
    print(f"Job 2 result: {result2}")
    
    # Get metrics
    metrics = await job_manager.get_metrics()
    print(f"Job metrics: {metrics}")
    
    # Get health status
    health = await job_manager.get_health_status()
    print(f"Health status: {health}")
    
    # Shutdown job manager
    await job_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


