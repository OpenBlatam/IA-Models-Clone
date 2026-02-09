"""
MCP Queue - Cola de tareas asíncronas
======================================
"""

import asyncio
import logging
from typing import Callable, Any, Dict, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Estados de tarea"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Tarea en la cola"""
    task_id: str = Field(..., description="ID único de la tarea")
    func: str = Field(..., description="Nombre de la función a ejecutar")
    args: list = Field(default_factory=list, description="Argumentos posicionales")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Argumentos nombrados")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Estado de la tarea")
    result: Optional[Any] = Field(None, description="Resultado de la tarea")
    error: Optional[str] = Field(None, description="Error si falló")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = Field(default=0, description="Prioridad (mayor = más prioritario)")
    retries: int = Field(default=0, description="Número de reintentos")
    max_retries: int = Field(default=3, description="Máximo de reintentos")


class AsyncTaskQueue:
    """
    Cola de tareas asíncronas
    
    Permite encolar tareas para ejecución asíncrona
    con control de prioridad y reintentos.
    """
    
    def __init__(self, max_workers: int = 5):
        """
        Args:
            max_workers: Número máximo de workers concurrentes
        """
        self.max_workers = max_workers
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._function_registry: Dict[str, Callable] = {}
    
    def register_function(self, name: str, func: Callable):
        """
        Registra una función para ejecutar en la cola
        
        Args:
            name: Nombre de la función
            func: Función a registrar
        """
        self._function_registry[name] = func
        logger.info(f"Registered function: {name}")
    
    async def enqueue(
        self,
        func_name: str,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Encola una tarea
        
        Args:
            func_name: Nombre de la función registrada
            *args: Argumentos posicionales
            task_id: ID de la tarea (opcional, se genera si no se proporciona)
            priority: Prioridad de la tarea
            max_retries: Máximo de reintentos
            
        Returns:
            ID de la tarea
        """
        import uuid
        
        if func_name not in self._function_registry:
            raise MCPError(f"Function {func_name} not registered")
        
        task_id = task_id or str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            func=func_name,
            args=list(args),
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
        )
        
        self._tasks[task_id] = task
        
        # Encolar con prioridad (negativo para que mayor = primero)
        await self._queue.put((-priority, task_id))
        
        logger.info(f"Enqueued task {task_id} with priority {priority}")
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Obtiene estado de una tarea
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Task o None si no existe
        """
        return self._tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            True si se canceló, False si no existe o ya está ejecutándose
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.RUNNING:
            return False  # No se puede cancelar si ya está ejecutándose
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        logger.info(f"Cancelled task {task_id}")
        
        return True
    
    async def start(self):
        """Inicia los workers"""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} workers")
    
    async def stop(self):
        """Detiene los workers"""
        if not self._running:
            return
        
        self._running = False
        
        # Esperar a que terminen los workers
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Stopped workers")
    
    async def _worker(self, worker_name: str):
        """Worker que procesa tareas"""
        logger.info(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Obtener tarea de la cola (con timeout)
                try:
                    priority, task_id = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self._tasks.get(task_id)
                if not task:
                    continue
                
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Ejecutar tarea
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                try:
                    func = self._function_registry[task.func]
                    
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*task.args, **task.kwargs)
                    else:
                        result = func(*task.args, **task.kwargs)
                    
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    
                    logger.info(f"Task {task_id} completed")
                    
                except Exception as e:
                    task.retries += 1
                    
                    if task.retries >= task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = datetime.utcnow()
                        logger.error(f"Task {task_id} failed after {task.retries} retries: {e}")
                    else:
                        # Reintentar
                        task.status = TaskStatus.PENDING
                        await self._queue.put((priority, task_id))
                        logger.warning(f"Task {task_id} retry {task.retries}/{task.max_retries}")
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la cola
        
        Returns:
            Diccionario con estadísticas
        """
        status_counts = {}
        for task in self._tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self._tasks),
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "running": self._running,
            "status_counts": status_counts,
        }

