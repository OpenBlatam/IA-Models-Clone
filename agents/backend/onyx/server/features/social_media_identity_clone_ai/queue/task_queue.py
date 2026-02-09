"""
Sistema de colas para procesamiento asíncrono
"""

import logging
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Estados de tarea"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Tarea en la cola"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        data = asdict(self)
        # Convertir datetime a string
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Crea desde diccionario"""
        # Convertir strings a datetime
        for key in ["created_at", "started_at", "completed_at"]:
            if key in data and data[key] and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class TaskQueue:
    """Cola de tareas para procesamiento asíncrono"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./storage/queue")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, Task] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[Callable] = []
        self.lock = asyncio.Lock()
        
        # Cargar tareas pendientes
        self._load_pending_tasks()
    
    def _load_pending_tasks(self):
        """Carga tareas pendientes desde disco"""
        try:
            for task_file in self.storage_path.glob("*.json"):
                try:
                    with open(task_file, "r") as f:
                        data = json.load(f)
                        task = Task.from_dict(data)
                        if task.status == TaskStatus.PENDING:
                            self.tasks[task.task_id] = task
                            asyncio.create_task(self._enqueue_task(task.task_id))
                except Exception as e:
                    logger.error(f"Error cargando tarea {task_file}: {e}")
        except Exception as e:
            logger.error(f"Error cargando tareas pendientes: {e}")
    
    async def _enqueue_task(self, task_id: str):
        """Agrega tarea a la cola"""
        await self.queue.put(task_id)
    
    async def add_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        max_retries: int = 3
    ) -> str:
        """
        Agrega tarea a la cola
        
        Args:
            task_type: Tipo de tarea
            payload: Datos de la tarea
            max_retries: Máximo de reintentos
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            max_retries=max_retries
        )
        
        async with self.lock:
            self.tasks[task_id] = task
            self._save_task(task)
            await self._enqueue_task(task_id)
        
        logger.info(f"Tarea agregada: {task_id} ({task_type})")
        return task_id
    
    async def get_task(self) -> Optional[Task]:
        """Obtiene siguiente tarea de la cola"""
        try:
            task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                return task
        except asyncio.TimeoutError:
            pass
        return None
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Actualiza estado de tarea"""
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            
            task.status = status
            if status == TaskStatus.PROCESSING:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = datetime.utcnow()
                if result:
                    task.result = result
                if error:
                    task.error = error
            
            self._save_task(task)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtiene tarea por ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Task]:
        """Lista tareas con filtros"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        # Ordenar por fecha de creación (más recientes primero)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def _save_task(self, task: Task):
        """Guarda tarea en disco"""
        try:
            task_file = self.storage_path / f"{task.task_id}.json"
            with open(task_file, "w") as f:
                json.dump(task.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error guardando tarea {task.task_id}: {e}")
    
    def _delete_task(self, task_id: str):
        """Elimina tarea del disco"""
        try:
            task_file = self.storage_path / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
        except Exception as e:
            logger.error(f"Error eliminando tarea {task_id}: {e}")


# Singleton global
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Obtiene instancia singleton de la cola"""
    global _task_queue
    if _task_queue is None:
        from ..config import get_settings
        settings = get_settings()
        _task_queue = TaskQueue(storage_path=f"{settings.storage_path}/queue")
    return _task_queue




