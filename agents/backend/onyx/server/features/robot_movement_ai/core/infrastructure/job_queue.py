"""
Job Queue System
================

Sistema de cola de trabajos.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Estado del trabajo."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Trabajo."""
    job_id: str
    job_type: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: int = 5  # 1-10
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobQueue:
    """
    Cola de trabajos.
    
    Gestiona trabajos con estados y prioridades.
    """
    
    def __init__(self, name: str):
        """
        Inicializar cola de trabajos.
        
        Args:
            name: Nombre de la cola
        """
        self.name = name
        self.jobs: Dict[str, Job] = {}
        self.pending_jobs: List[str] = []  # job_ids ordenados por prioridad
        self.workers: Dict[str, Callable] = {}  # job_type -> handler
    
    def register_worker(
        self,
        job_type: str,
        handler: Callable
    ) -> None:
        """
        Registrar worker para tipo de trabajo.
        
        Args:
            job_type: Tipo de trabajo
            handler: Función manejadora
        """
        self.workers[job_type] = handler
        logger.info(f"Registered worker for job type: {job_type}")
    
    async def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        max_attempts: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Agregar trabajo a la cola.
        
        Args:
            job_type: Tipo de trabajo
            payload: Datos del trabajo
            priority: Prioridad (1-10)
            max_attempts: Intentos máximos
            metadata: Metadata adicional
            
        Returns:
            Trabajo creado
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        self.pending_jobs.append(job_id)
        
        # Ordenar por prioridad (mayor primero)
        self.pending_jobs.sort(
            key=lambda jid: self.jobs[jid].priority,
            reverse=True
        )
        
        logger.info(f"Enqueued job: {job_type} ({job_id})")
        
        return job
    
    async def dequeue(self) -> Optional[Job]:
        """
        Obtener trabajo de la cola.
        
        Returns:
            Trabajo o None si no hay pendientes
        """
        if not self.pending_jobs:
            return None
        
        job_id = self.pending_jobs.pop(0)
        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        job.attempts += 1
        
        return job
    
    async def process_job(self, job: Job) -> Any:
        """
        Procesar trabajo.
        
        Args:
            job: Trabajo a procesar
            
        Returns:
            Resultado del trabajo
        """
        if job.job_type not in self.workers:
            raise ValueError(f"No worker registered for job type: {job.job_type}")
        
        handler = self.workers[job.job_type]
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(job.payload)
            else:
                result = handler(job.payload)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            job.result = result
            
            return result
        except Exception as e:
            job.error = str(e)
            
            if job.attempts >= job.max_attempts:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job.job_id} failed after {job.attempts} attempts: {e}")
            else:
                # Reencolar
                job.status = JobStatus.PENDING
                self.pending_jobs.append(job.job_id)
                self.pending_jobs.sort(
                    key=lambda jid: self.jobs[jid].priority,
                    reverse=True
                )
                logger.warning(f"Job {job.job_id} will be retried (attempt {job.attempts}/{job.max_attempts})")
            
            raise
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Obtener trabajo por ID."""
        return self.jobs.get(job_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola."""
        status_counts = {}
        for job in self.jobs.values():
            status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
        
        return {
            "queue_name": self.name,
            "total_jobs": len(self.jobs),
            "pending_jobs": len(self.pending_jobs),
            "status_counts": status_counts,
            "registered_workers": list(self.workers.keys())
        }


class JobQueueManager:
    """
    Gestor de colas de trabajos.
    
    Gestiona múltiples colas de trabajos.
    """
    
    def __init__(self):
        """Inicializar gestor de colas."""
        self.queues: Dict[str, JobQueue] = {}
    
    def create_queue(self, name: str) -> JobQueue:
        """
        Crear cola de trabajos.
        
        Args:
            name: Nombre de la cola
            
        Returns:
            Cola creada
        """
        queue = JobQueue(name)
        self.queues[name] = queue
        logger.info(f"Created job queue: {name}")
        return queue
    
    def get_queue(self, name: str) -> Optional[JobQueue]:
        """Obtener cola por nombre."""
        return self.queues.get(name)


# Instancia global
_job_queue_manager: Optional[JobQueueManager] = None


def get_job_queue_manager() -> JobQueueManager:
    """Obtener instancia global del gestor de colas de trabajos."""
    global _job_queue_manager
    if _job_queue_manager is None:
        _job_queue_manager = JobQueueManager()
    return _job_queue_manager






