"""
Video Scheduler Service
Schedules video generation for future execution
"""

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
import logging
import asyncio

from ..core.models import VideoGenerationRequest

logger = logging.getLogger(__name__)


class ScheduledJob:
    """Represents a scheduled video generation job"""
    
    def __init__(
        self,
        job_id: str,
        video_id: UUID,
        request: VideoGenerationRequest,
        scheduled_at: datetime,
        timezone: str = "UTC",
        repeat: Optional[str] = None,  # daily, weekly, monthly
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.job_id = job_id
        self.video_id = video_id
        self.request = request
        self.scheduled_at = scheduled_at
        self.timezone = timezone
        self.repeat = repeat
        self.metadata = metadata or {}
        self.status = "scheduled"
        self.executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "video_id": str(self.video_id),
            "scheduled_at": self.scheduled_at.isoformat(),
            "timezone": self.timezone,
            "repeat": self.repeat,
            "status": self.status,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "metadata": self.metadata,
        }


class SchedulerService:
    """Manages scheduled video generation"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
    
    def schedule_video(
        self,
        video_id: UUID,
        request: VideoGenerationRequest,
        scheduled_at: datetime,
        timezone: str = "UTC",
        repeat: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScheduledJob:
        """
        Schedule video generation
        
        Args:
            video_id: Video ID
            request: Video generation request
            scheduled_at: When to generate video
            timezone: Timezone
            repeat: Repeat pattern (daily, weekly, monthly)
            metadata: Additional metadata
            
        Returns:
            Scheduled job
        """
        job_id = f"scheduled_{len(self.jobs) + 1}"
        
        job = ScheduledJob(
            job_id=job_id,
            video_id=video_id,
            request=request,
            scheduled_at=scheduled_at,
            timezone=timezone,
            repeat=repeat,
            metadata=metadata
        )
        
        self.jobs[job_id] = job
        logger.info(f"Scheduled video generation: {job_id} for {scheduled_at}")
        
        return job
    
    def get_scheduled_jobs(
        self,
        video_id: Optional[UUID] = None,
        status: Optional[str] = None
    ) -> List[ScheduledJob]:
        """Get scheduled jobs"""
        jobs = list(self.jobs.values())
        
        if video_id:
            jobs = [j for j in jobs if j.video_id == video_id]
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return jobs
    
    def cancel_scheduled_job(self, job_id: str) -> bool:
        """Cancel scheduled job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == "executed":
            return False
        
        job.status = "cancelled"
        logger.info(f"Cancelled scheduled job: {job_id}")
        return True
    
    async def start_worker(self, orchestrator):
        """Start scheduler worker"""
        if self.running:
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._worker(orchestrator))
        logger.info("Scheduler worker started")
    
    async def stop_worker(self):
        """Stop scheduler worker"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler worker stopped")
    
    async def _worker(self, orchestrator):
        """Worker process"""
        while self.running:
            try:
                now = datetime.utcnow()
                
                for job in self.jobs.values():
                    if job.status == "scheduled" and now >= job.scheduled_at:
                        # Execute job
                        logger.info(f"Executing scheduled job: {job.job_id}")
                        job.status = "executing"
                        
                        try:
                            await orchestrator.process_generation(
                                job.video_id,
                                job.request
                            )
                            
                            job.status = "executed"
                            job.executed_at = datetime.utcnow()
                            
                            # Handle repeat
                            if job.repeat:
                                self._schedule_next(job)
                                
                        except Exception as e:
                            logger.error(f"Scheduled job failed: {str(e)}")
                            job.status = "failed"
                            job.metadata["error"] = str(e)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler worker error: {str(e)}")
                await asyncio.sleep(60)
    
    def _schedule_next(self, job: ScheduledJob):
        """Schedule next occurrence for repeating job"""
        if job.repeat == "daily":
            next_time = job.scheduled_at + timedelta(days=1)
        elif job.repeat == "weekly":
            next_time = job.scheduled_at + timedelta(weeks=1)
        elif job.repeat == "monthly":
            next_time = job.scheduled_at + timedelta(days=30)
        else:
            return
        
        # Create new job for next occurrence
        new_job = ScheduledJob(
            job_id=f"{job.job_id}_next",
            video_id=job.video_id,
            request=job.request,
            scheduled_at=next_time,
            timezone=job.timezone,
            repeat=job.repeat,
            metadata=job.metadata
        )
        
        self.jobs[new_job.job_id] = new_job


_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """Get scheduler service instance (singleton)"""
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service

