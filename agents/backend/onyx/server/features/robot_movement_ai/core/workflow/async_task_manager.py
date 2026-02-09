"""
Async Task Manager System
==========================

Sistema de gestión de tareas asíncronas.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """Tarea asíncrona."""
    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:
    """
    Gestor de tareas asíncronas.
    
    Gestiona ejecución de tareas asíncronas con prioridades.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Inicializar gestor de tareas.
        
        Args:
            max_workers: Número máximo de workers
        """
        self.max_workers = max_workers
        self.tasks: Dict[str, AsyncTask] = {}
        self.pending_tasks: List[str] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self.worker_task: Optional[asyncio.Task] = None
    
    def submit(
        self,
        name: str,
        func: Callable,
        *args,
        priority: int = 5,
        **kwargs
    ) -> str:
        """
        Enviar tarea para ejecución.
        
        Args:
            name: Nombre de la tarea
            func: Función a ejecutar
            *args: Argumentos
            priority: Prioridad (1-10)
            **kwargs: Keyword arguments
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        task = AsyncTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        self.tasks[task_id] = task
        self.pending_tasks.append(task_id)
        
        # Ordenar por prioridad (mayor primero)
        self.pending_tasks.sort(
            key=lambda tid: self.tasks[tid].priority,
            reverse=True
        )
        
        # Iniciar worker si no está corriendo
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())
        
        logger.info(f"Submitted task: {name} ({task_id})")
        
        return task_id
    
    async def _worker_loop(self) -> None:
        """Loop de worker para procesar tareas."""
        while self.pending_tasks or self.running_tasks:
            if self.pending_tasks:
                task_id = self.pending_tasks.pop(0)
                task = self.tasks[task_id]
                
                # Ejecutar tarea
                asyncio.create_task(self._execute_task(task))
            
            await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: AsyncTask) -> None:
        """Ejecutar tarea."""
        async with self.semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self.running_tasks[task.task_id] = asyncio.current_task()
            
            try:
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        task.func,
                        *task.args,
                        **task.kwargs
                    )
                
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now().isoformat()
                
                logger.info(f"Task {task.name} completed: {task.task_id}")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now().isoformat()
                logger.error(f"Task {task.name} failed: {e}", exc_info=True)
            finally:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
    
    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """Obtener tarea por ID."""
        return self.tasks.get(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> Optional[AsyncTask]:
        """
        Esperar a que tarea complete.
        
        Args:
            task_id: ID de la tarea
            timeout: Timeout en segundos
            
        Returns:
            Tarea completada o None si timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            task = self.get_task(task_id)
            if not task:
                return None
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return None
            
            await asyncio.sleep(0.1)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancelar tarea."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                if task_id in self.pending_tasks:
                    self.pending_tasks.remove(task_id)
                return True
            elif task_id in self.running_tasks:
                running_task = self.running_tasks[task_id]
                running_task.cancel()
                task.status = TaskStatus.CANCELLED
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de tareas."""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "status_counts": status_counts,
            "max_workers": self.max_workers
        }


# Instancia global
_async_task_manager: Optional[AsyncTaskManager] = None


def get_async_task_manager(max_workers: int = 10) -> AsyncTaskManager:
    """Obtener instancia global del gestor de tareas asíncronas."""
    global _async_task_manager
    if _async_task_manager is None:
        _async_task_manager = AsyncTaskManager(max_workers=max_workers)
    return _async_task_manager


