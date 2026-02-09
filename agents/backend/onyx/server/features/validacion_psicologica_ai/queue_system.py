"""
Sistema de Colas Asíncronas
===========================
Procesamiento asíncrono con colas
"""

from typing import Dict, Any, List, Optional, Callable
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import structlog
import asyncio
from dataclasses import dataclass, field
from collections import deque, defaultdict

logger = structlog.get_logger()


class JobStatus(str, Enum):
    """Estado de trabajo"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Prioridad de trabajo"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class QueueJob:
    """Trabajo en cola"""
    id: UUID
    job_type: str
    data: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class QueueManager:
    """Gestor de colas"""
    
    def __init__(self, max_workers: int = 5):
        """
        Inicializar gestor
        
        Args:
            max_workers: Máximo de workers concurrentes
        """
        self.max_workers = max_workers
        self._queues: Dict[str, deque] = defaultdict(deque)
        self._jobs: Dict[UUID, QueueJob] = {}
        self._workers: List[asyncio.Task] = []
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        logger.info("QueueManager initialized", max_workers=max_workers)
    
    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Registrar handler para tipo de trabajo
        
        Args:
            job_type: Tipo de trabajo
            handler: Función handler
        """
        self._handlers[job_type] = handler
        logger.info("Handler registered", job_type=job_type)
    
    async def enqueue(
        self,
        job_type: str,
        data: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL
    ) -> UUID:
        """
        Agregar trabajo a la cola
        
        Args:
            job_type: Tipo de trabajo
            data: Datos del trabajo
            priority: Prioridad
            
        Returns:
            ID del trabajo
        """
        job = QueueJob(
            id=uuid4(),
            job_type=job_type,
            data=data,
            priority=priority
        )
        
        self._jobs[job.id] = job
        self._queues[job_type].append(job)
        job.status = JobStatus.QUEUED
        
        logger.info(
            "Job enqueued",
            job_id=str(job.id),
            job_type=job_type,
            priority=priority.value
        )
        
        return job.id
    
    async def start_workers(self) -> None:
        """Iniciar workers"""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info("Workers started", count=self.max_workers)
    
    async def stop_workers(self) -> None:
        """Detener workers"""
        self._running = False
        
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Workers stopped")
    
    async def _worker(self, worker_name: str) -> None:
        """Worker para procesar trabajos"""
        logger.info("Worker started", name=worker_name)
        
        while self._running:
            try:
                # Buscar trabajo en todas las colas
                job = None
                for queue in self._queues.values():
                    if queue:
                        job = queue.popleft()
                        break
                
                if not job:
                    await asyncio.sleep(0.1)
                    continue
                
                # Procesar trabajo
                await self._process_job(job, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", worker=worker_name, error=str(e))
                await asyncio.sleep(1)
    
    async def _process_job(self, job: QueueJob, worker_name: str) -> None:
        """Procesar trabajo individual"""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        
        logger.info(
            "Job processing started",
            job_id=str(job.id),
            worker=worker_name
        )
        
        try:
            handler = self._handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler for job type: {job.job_type}")
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(job.data)
            else:
                result = handler(job.data)
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            logger.info(
                "Job completed",
                job_id=str(job.id),
                worker=worker_name
            )
            
        except Exception as e:
            job.error = str(e)
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                job.status = JobStatus.PENDING
                self._queues[job.job_type].append(job)
                logger.warning(
                    "Job failed, retrying",
                    job_id=str(job.id),
                    retry_count=job.retry_count
                )
            else:
                job.status = JobStatus.FAILED
                logger.error(
                    "Job failed permanently",
                    job_id=str(job.id),
                    error=str(e)
                )
    
    def get_job(self, job_id: UUID) -> Optional[QueueJob]:
        """
        Obtener trabajo por ID
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Trabajo o None
        """
        return self._jobs.get(job_id)
    
    def get_queue_stats(self, job_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de cola
        
        Args:
            job_type: Tipo de trabajo (opcional)
            
        Returns:
            Estadísticas
        """
        if job_type:
            queue = self._queues.get(job_type, deque())
            jobs = [j for j in self._jobs.values() if j.job_type == job_type]
        else:
            queue = None
            jobs = list(self._jobs.values())
        
        status_counts = defaultdict(int)
        for job in jobs:
            status_counts[job.status.value] += 1
        
        return {
            "queue_size": len(queue) if queue else sum(len(q) for q in self._queues.values()),
            "total_jobs": len(jobs),
            "status_distribution": dict(status_counts),
            "workers": len(self._workers),
            "running": self._running
        }


# Instancia global del gestor de colas
queue_manager = QueueManager()

