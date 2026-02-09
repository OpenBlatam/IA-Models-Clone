"""
Sistema de cola de trabajos para Robot Movement AI v2.0
Job queue con prioridades y retry
"""

from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
from queue import PriorityQueue
import threading


class JobStatus(str, Enum):
    """Estados de trabajo"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Prioridades de trabajo"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class Job:
    """Representa un trabajo en la cola"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def __lt__(self, other):
        """Comparar por prioridad y tiempo de creación"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class JobQueue:
    """Cola de trabajos con prioridades"""
    
    def __init__(self, max_workers: int = 4):
        """
        Inicializar cola de trabajos
        
        Args:
            max_workers: Número máximo de workers concurrentes
        """
        self.max_workers = max_workers
        self.queue: PriorityQueue = PriorityQueue()
        self.jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Iniciar workers"""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self):
        """Detener workers"""
        self.running = False
        
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def _worker(self, name: str):
        """Worker que procesa trabajos"""
        while self.running:
            try:
                # Obtener trabajo de la cola
                try:
                    job = await asyncio.wait_for(
                        asyncio.to_thread(self.queue.get, timeout=1),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Ejecutar trabajo
                await self._execute_job(job)
                
            except Exception as e:
                print(f"Error in worker {name}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_job(self, job: Job):
        """Ejecutar trabajo"""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            import inspect
            
            if inspect.iscoroutinefunction(job.func):
                result = await job.func(*job.args, **job.kwargs)
            else:
                result = await asyncio.to_thread(job.func, *job.args, **job.kwargs)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            
        except Exception as e:
            job.error = e
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                await asyncio.sleep(job.retry_delay * (2 ** job.retry_count))  # Exponential backoff
                await asyncio.to_thread(self.queue.put, job)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
    
    def enqueue(
        self,
        func: Callable,
        *args,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Agregar trabajo a la cola
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            priority: Prioridad del trabajo
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos
            **kwargs: Argumentos nombrados
            
        Returns:
            ID del trabajo
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self.jobs[job_id] = job
        self.queue.put(job)
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Obtener trabajo por ID"""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar trabajo"""
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola"""
        status_counts = {}
        for job in self.jobs.values():
            status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
        
        return {
            "queue_size": self.queue.qsize(),
            "total_jobs": len(self.jobs),
            "status_counts": status_counts,
            "workers": len(self.workers),
            "running": self.running
        }


# Instancia global
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Obtener instancia global de la cola de trabajos"""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


async def start_job_queue():
    """Iniciar cola de trabajos"""
    queue = get_job_queue()
    await queue.start()


async def stop_job_queue():
    """Detener cola de trabajos"""
    queue = get_job_queue()
    await queue.stop()


def enqueue_job(
    func: Callable,
    *args,
    priority: JobPriority = JobPriority.NORMAL,
    **kwargs
) -> str:
    """Helper para agregar trabajo a la cola"""
    queue = get_job_queue()
    return queue.enqueue(func, *args, priority=priority, **kwargs)




