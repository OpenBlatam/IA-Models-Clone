"""
Background Job Scheduler
Schedules and manages background jobs
"""

import asyncio
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """Scheduled job definition"""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    schedule_time: datetime = None
    interval: Optional[timedelta] = None  # For recurring jobs
    max_runs: Optional[int] = None  # Limit runs for recurring jobs
    run_count: int = 0
    status: JobStatus = JobStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.next_run is None and self.schedule_time:
            self.next_run = self.schedule_time


class JobScheduler:
    """Schedules and executes background jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start scheduler"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job scheduler started")
    
    async def stop(self):
        """Stop scheduler"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Job scheduler stopped")
    
    def schedule(
        self,
        job_id: str,
        func: Callable,
        schedule_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        max_runs: Optional[int] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Schedule a job
        
        Args:
            job_id: Unique job identifier
            func: Function to execute
            schedule_time: When to run (defaults to now)
            interval: Recurring interval (for recurring jobs)
            max_runs: Maximum runs for recurring jobs
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Job ID
        """
        if schedule_time is None:
            schedule_time = datetime.utcnow()
        
        job = ScheduledJob(
            id=job_id,
            func=func,
            args=args,
            kwargs=kwargs or {},
            schedule_time=schedule_time,
            interval=interval,
            max_runs=max_runs,
            next_run=schedule_time,
            status=JobStatus.SCHEDULED
        )
        
        self.jobs[job_id] = job
        logger.info(f"Scheduled job: {job_id} for {schedule_time}")
        
        return job_id
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.CANCELLED
            logger.info(f"Cancelled job: {job_id}")
            return True
        return False
    
    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _check_and_run_jobs(self):
        """Check for jobs that need to run"""
        now = datetime.utcnow()
        
        for job_id, job in list(self.jobs.items()):
            if job.status in [JobStatus.CANCELLED, JobStatus.COMPLETED]:
                continue
            
            if job.next_run and now >= job.next_run:
                # Run job
                asyncio.create_task(self._run_job(job))
    
    async def _run_job(self, job: ScheduledJob):
        """Execute a job"""
        job.status = JobStatus.RUNNING
        job.last_run = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(job.func):
                await job.func(*job.args, **job.kwargs)
            else:
                job.func(*job.args, **job.kwargs)
            
            job.run_count += 1
            
            # Check if recurring
            if job.interval:
                # Check max runs
                if job.max_runs and job.run_count >= job.max_runs:
                    job.status = JobStatus.COMPLETED
                    logger.info(f"Job {job.id} completed after {job.run_count} runs")
                else:
                    # Schedule next run
                    job.next_run = datetime.utcnow() + job.interval
                    job.status = JobStatus.SCHEDULED
                    logger.debug(f"Job {job.id} rescheduled for {job.next_run}")
            else:
                job.status = JobStatus.COMPLETED
                logger.info(f"Job {job.id} completed")
                
        except Exception as e:
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)


# Global scheduler instance
_scheduler: Optional[JobScheduler] = None


def get_scheduler() -> JobScheduler:
    """Get global job scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobScheduler()
    return _scheduler















