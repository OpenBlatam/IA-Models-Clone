"""
Task Queue - Sistema de Colas de Tareas
========================================

Sistema de colas para procesamiento asíncrono de tareas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Tarea en la cola."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = 0  # Mayor número = mayor prioridad


class TaskQueue:
    """Cola de tareas asíncrona."""
    
    def __init__(self, max_workers: int = 5):
        """
        Inicializar cola de tareas.
        
        Args:
            max_workers: Número máximo de workers concurrentes
        """
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.workers: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        """Iniciar workers."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Task queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Detener workers."""
        self._running = False
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Task queue stopped")
    
    async def _worker(self, name: str):
        """Worker que procesa tareas."""
        while self._running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now()
                
                try:
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        result = task.func(*task.args, **task.kwargs)
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    
                    logger.debug(f"Task {task.id} completed by {name}")
                
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    logger.error(f"Task {task.id} failed: {e}")
                
                self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in worker {name}: {e}")
    
    async def enqueue(
        self,
        func: Callable,
        *args,
        name: str = "",
        priority: int = 0,
        **kwargs
    ) -> str:
        """
        Encolar tarea.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            name: Nombre de la tarea
            priority: Prioridad (mayor = más prioritario)
            **kwargs: Argumentos nombrados
        
        Returns:
            ID de la tarea
        """
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )
        
        async with self._lock:
            self.tasks[task.id] = task
        
        # Encolar (implementación simple, en producción usar priority queue)
        await self.queue.put(task)
        
        logger.debug(f"Task {task.id} enqueued: {task.name}")
        return task.id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Obtener tarea por ID."""
        async with self._lock:
            return self.tasks.get(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Task:
        """Esperar a que una tarea se complete."""
        start_time = datetime.now()
        
        while True:
            task = await self.get_task(task_id)
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
    
    def get_queue_size(self) -> int:
        """Obtener tamaño de la cola."""
        return self.queue.qsize()
    
    def get_active_tasks(self) -> List[Task]:
        """Obtener tareas activas."""
        async with self._lock:
            return [
                task
                for task in self.tasks.values()
                if task.status == TaskStatus.PROCESSING
            ]
    
    def get_pending_tasks(self) -> List[Task]:
        """Obtener tareas pendientes."""
        async with self._lock:
            return [
                task
                for task in self.tasks.values()
                if task.status == TaskStatus.PENDING
            ]
































