"""
Async Queue - Sistema de cola para procesamiento asíncrono
===========================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Estados de un job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job en la cola"""
    id: str
    job_type: str
    data: Dict[str, Any]
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    priority: int = 0  # Mayor número = mayor prioridad


class AsyncQueue:
    """Sistema de cola asíncrona"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        self.processors: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}
        self.running = False
    
    def register_processor(self, job_type: str, processor: Callable[[Dict[str, Any]], Awaitable[Any]]):
        """Registra un procesador para un tipo de job"""
        self.processors[job_type] = processor
        logger.info(f"Procesador registrado para tipo: {job_type}")
    
    async def enqueue(self, job_type: str, data: Dict[str, Any],
                     priority: int = 0, max_retries: int = 3) -> str:
        """Agrega un job a la cola"""
        job_id = str(uuid4())
        
        job = Job(
            id=job_id,
            job_type=job_type,
            data=data,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            priority=priority,
            max_retries=max_retries
        )
        
        self.jobs[job_id] = job
        
        # Agregar a cola con prioridad
        await self.queue.put((priority, job_id))
        
        logger.info(f"Job {job_id} agregado a la cola (tipo: {job_type})")
        return job_id
    
    async def start_workers(self):
        """Inicia los workers"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Iniciados {self.max_workers} workers")
    
    async def stop_workers(self):
        """Detiene los workers"""
        self.running = False
        
        # Esperar a que terminen los workers
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers = []
        logger.info("Workers detenidos")
    
    async def _worker(self, worker_id: str):
        """Worker que procesa jobs"""
        logger.info(f"Worker {worker_id} iniciado")
        
        while self.running:
            try:
                # Obtener job de la cola (con timeout)
                try:
                    priority, job_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                job = self.jobs.get(job_id)
                if not job:
                    continue
                
                # Verificar si hay procesador
                processor = self.processors.get(job.job_type)
                if not processor:
                    job.status = JobStatus.FAILED
                    job.error = f"No hay procesador para tipo {job.job_type}"
                    logger.error(f"Job {job_id}: {job.error}")
                    continue
                
                # Procesar job
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                
                try:
                    result = await processor(job.data)
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.completed_at = datetime.now()
                    logger.info(f"Job {job_id} completado exitosamente")
                
                except Exception as e:
                    job.retries += 1
                    job.error = str(e)
                    
                    if job.retries >= job.max_retries:
                        job.status = JobStatus.FAILED
                        logger.error(f"Job {job_id} falló después de {job.retries} intentos: {e}")
                    else:
                        job.status = JobStatus.PENDING
                        # Re-agregar a la cola
                        await self.queue.put((job.priority, job_id))
                        logger.warning(f"Job {job_id} reintentando ({job.retries}/{job.max_retries})")
            
            except Exception as e:
                logger.error(f"Error en worker {worker_id}: {e}")
        
        logger.info(f"Worker {worker_id} detenido")
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Obtiene un job por ID"""
        return self.jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de un job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "type": job.job_type,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retries": job.retries,
            "error": job.error
        }
    
    def list_jobs(self, status: Optional[JobStatus] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Lista jobs"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Ordenar por fecha de creación (más reciente primero)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return [self.get_job_status(j.id) for j in jobs[:limit]]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancela un job"""
        job = self.jobs.get(job_id)
        if not job or job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            return False
        
        job.status = JobStatus.CANCELLED
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la cola"""
        jobs_by_status = {}
        for status in JobStatus:
            jobs_by_status[status.value] = sum(
                1 for j in self.jobs.values() if j.status == status
            )
        
        return {
            "total_jobs": len(self.jobs),
            "queue_size": self.queue.qsize(),
            "active_workers": len([w for w in self.workers if not w.done()]),
            "jobs_by_status": jobs_by_status,
            "registered_processors": list(self.processors.keys())
        }




