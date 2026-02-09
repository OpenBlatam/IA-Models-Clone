"""
Job Queue Service - Cola de trabajos
=====================================

Sistema de cola de trabajos para procesamiento asíncrono.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Estados de trabajo"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Prioridades de trabajo"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Job:
    """Trabajo en cola"""
    id: str
    job_type: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class JobQueueService:
    """Servicio de cola de trabajos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.jobs: Dict[str, Job] = {}
        self.queue: List[str] = []  # IDs de trabajos en cola
        self.processors: Dict[str, Callable] = {}  # job_type -> processor
        self.is_processing = False
        logger.info("JobQueueService initialized")
    
    def register_processor(self, job_type: str, processor: Callable):
        """Registrar procesador para tipo de trabajo"""
        self.processors[job_type] = processor
        logger.info(f"Processor registered for job type: {job_type}")
    
    def enqueue_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL
    ) -> Job:
        """Agregar trabajo a la cola"""
        job_id = f"job_{int(datetime.now().timestamp())}_{len(self.jobs)}"
        
        job = Job(
            id=job_id,
            job_type=job_type,
            payload=payload,
            status=JobStatus.QUEUED,
            priority=priority,
        )
        
        self.jobs[job_id] = job
        
        # Agregar a cola según prioridad
        if priority == JobPriority.URGENT:
            self.queue.insert(0, job_id)
        elif priority == JobPriority.HIGH:
            insert_pos = next(
                (i for i, jid in enumerate(self.queue) 
                 if self.jobs[jid].priority in [JobPriority.NORMAL, JobPriority.LOW]),
                len(self.queue)
            )
            self.queue.insert(insert_pos, job_id)
        else:
            self.queue.append(job_id)
        
        logger.info(f"Job enqueued: {job_id}")
        return job
    
    async def process_job(self, job_id: str) -> Job:
        """Procesar trabajo"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status != JobStatus.QUEUED:
            raise ValueError(f"Job {job_id} is not queued")
        
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        
        processor = self.processors.get(job.job_type)
        if not processor:
            job.status = JobStatus.FAILED
            job.error = f"No processor registered for job type: {job.job_type}"
            return job
        
        try:
            # Procesar trabajo
            if asyncio.iscoroutinefunction(processor):
                result = await processor(job.payload)
            else:
                result = processor(job.payload)
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            logger.info(f"Job completed: {job_id}")
        
        except Exception as e:
            job.error = str(e)
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                job.status = JobStatus.QUEUED
                # Re-agregar a cola
                self.queue.append(job_id)
                logger.warning(f"Job {job_id} failed, retrying ({job.retry_count}/{job.max_retries})")
            else:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job_id} failed after {job.max_retries} retries: {e}")
        
        return job
    
    async def process_queue(self):
        """Procesar cola de trabajos"""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            while self.queue:
                job_id = self.queue.pop(0)
                await self.process_job(job_id)
                await asyncio.sleep(0.1)  # Pequeña pausa entre trabajos
        finally:
            self.is_processing = False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de trabajo"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count,
            "error": job.error,
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar trabajo"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
            job.status = JobStatus.CANCELLED
            if job_id in self.queue:
                self.queue.remove(job_id)
            return True
        
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola"""
        status_counts = {}
        for job in self.jobs.values():
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_jobs": len(self.jobs),
            "queued": len(self.queue),
            "processing": sum(1 for j in self.jobs.values() if j.status == JobStatus.PROCESSING),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
            "status_breakdown": status_counts,
        }




