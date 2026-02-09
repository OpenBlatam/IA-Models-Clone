"""
Task Manager Mixin
==================
Mixin para sistemas que gestionan tareas
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import uuid


class TaskStatus(Enum):
    """Estados de tarea"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Tarea genérica"""
    id: str
    task_type: str
    input_data: Dict[str, Any]
    status: TaskStatus
    priority: int = 0
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class TaskManagerMixin:
    """
    Mixin para gestión de tareas
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tasks: Dict[str, Task] = {}
        self._task_queue: List[str] = []  # IDs ordenados por prioridad
        self._task_handlers: Dict[str, Callable] = {}
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Registrar handler para tipo de tarea"""
        self._task_handlers[task_type] = handler
    
    def create_task(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = 0
    ) -> Task:
        """Crear tarea"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            task_type=task_type,
            input_data=input_data,
            status=TaskStatus.PENDING,
            priority=priority
        )
        
        self._tasks[task_id] = task
        self._task_queue.append(task_id)
        
        # Ordenar por prioridad (mayor primero)
        self._task_queue.sort(
            key=lambda tid: self._tasks[tid].priority,
            reverse=True
        )
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtener tarea"""
        return self._tasks.get(task_id)
    
    def execute_task(self, task_id: str) -> Any:
        """Ejecutar tarea"""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.status != TaskStatus.PENDING and task.status != TaskStatus.QUEUED:
            raise ValueError(f"Task {task_id} is not in executable state")
        
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type {task.task_type}")
            
            result = handler(task.input_data)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            return result
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            raise
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancelar tarea"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)
        
        return True
    
    def get_pending_tasks(self) -> List[Task]:
        """Obtener tareas pendientes"""
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.PENDING
        ]
    
    def get_running_tasks(self) -> List[Task]:
        """Obtener tareas en ejecución"""
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.RUNNING
        ]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de tareas"""
        status_counts = {}
        for task in self._tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_tasks': len(self._tasks),
            'pending_tasks': len(self.get_pending_tasks()),
            'running_tasks': len(self.get_running_tasks()),
            'completed_tasks': status_counts.get('completed', 0),
            'failed_tasks': status_counts.get('failed', 0),
            'cancelled_tasks': status_counts.get('cancelled', 0),
            'queued_tasks': len(self._task_queue),
            'registered_handlers': len(self._task_handlers)
        }

