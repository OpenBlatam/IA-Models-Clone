"""
Queue Manager - Gestor de Colas Avanzado
========================================

Gestión avanzada de colas:
- Priority queues
- Delayed jobs
- Scheduled jobs
- Job retries
- Job status tracking
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Estados de job"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Prioridades de job"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class Job:
    """Job en cola"""
    
    def __init__(
        self,
        job_id: str,
        task: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        delay: Optional[float] = None,
        scheduled_at: Optional[datetime] = None
    ) -> None:
        self.job_id = job_id
        self.task = task
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.max_retries = max_retries
        self.retry_count = 0
        self.delay = delay
        self.scheduled_at = scheduled_at or datetime.now()
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[str] = None
    
    def is_ready(self) -> bool:
        """Verifica si el job está listo para ejecutarse"""
        if self.status != JobStatus.PENDING:
            return False
        
        if self.delay:
            elapsed = (datetime.now() - self.created_at).total_seconds()
            return elapsed >= self.delay
        
        if self.scheduled_at:
            return datetime.now() >= self.scheduled_at
        
        return True


class QueueManager:
    """
    Gestor de colas avanzado.
    """
    
    def __init__(self, max_workers: int = 5) -> None:
        self.max_workers = max_workers
        self.queues: Dict[str, List[Job]] = {}
        self.jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    def create_queue(self, queue_name: str) -> None:
        """Crea una cola"""
        if queue_name not in self.queues:
            self.queues[queue_name] = []
            logger.info(f"Queue created: {queue_name}")
    
    def enqueue(
        self,
        queue_name: str,
        task: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        delay: Optional[float] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Agrega job a la cola"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            task=task,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            delay=delay,
            scheduled_at=scheduled_at
        )
        
        self.jobs[job_id] = job
        self.queues[queue_name].append(job)
        
        # Ordenar por prioridad
        self.queues[queue_name].sort(key=lambda j: j.priority.value, reverse=True)
        
        logger.info(f"Job {job_id} enqueued in {queue_name}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "priority": job.priority.value,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error
        }
    
    async def process_queue(self, queue_name: str) -> None:
        """Procesa jobs de una cola"""
        if queue_name not in self.queues:
            return
        
        while self.running:
            ready_jobs = [
                job for job in self.queues[queue_name]
                if job.is_ready() and job.status == JobStatus.PENDING
            ]
            
            if not ready_jobs:
                await asyncio.sleep(1)
                continue
            
            # Tomar job de mayor prioridad
            job = ready_jobs[0]
            self.queues[queue_name].remove(job)
            
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            try:
                # Ejecutar job
                if asyncio.iscoroutinefunction(job.task):
                    job.result = await job.task(*job.args, **job.kwargs)
                else:
                    job.result = job.task(*job.args, **job.kwargs)
                
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                logger.info(f"Job {job.job_id} completed")
                
            except Exception as e:
                job.error = str(e)
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    # Re-enqueue con delay
                    await asyncio.sleep(2 ** job.retry_count)  # Exponential backoff
                    self.queues[queue_name].append(job)
                    logger.info(f"Job {job.job_id} retrying ({job.retry_count}/{job.max_retries})")
                else:
                    job.status = JobStatus.FAILED
                    logger.error(f"Job {job.job_id} failed after {job.max_retries} retries")
    
    async def start(self) -> None:
        """Inicia workers"""
        self.running = True
        
        for queue_name in self.queues.keys():
            for _ in range(self.max_workers):
                worker = asyncio.create_task(self.process_queue(queue_name))
                self.workers.append(worker)
        
        logger.info(f"Queue manager started with {len(self.workers)} workers")
    
    async def stop(self) -> None:
        """Detiene workers"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Queue manager stopped")
    
    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Obtiene estadísticas de una cola"""
        if queue_name not in self.queues:
            return {}
        
        jobs = self.queues[queue_name]
        return {
            "queue": queue_name,
            "total_jobs": len(jobs),
            "pending": sum(1 for j in jobs if j.status == JobStatus.PENDING),
            "running": sum(1 for j in jobs if j.status == JobStatus.RUNNING),
            "completed": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED)
        }


def get_queue_manager(max_workers: int = 5) -> QueueManager:
    """Obtiene gestor de colas"""
    return QueueManager(max_workers=max_workers)















