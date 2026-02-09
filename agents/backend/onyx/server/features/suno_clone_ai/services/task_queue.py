"""
Sistema de Colas de Tareas Avanzado

Soporta:
- Celery para tareas asíncronas
- Redis como broker
- Prioridades de tareas
- Retry automático
- Monitoreo de tareas
"""

import logging
import json
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridades de tareas"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Estados de tareas"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Representa una tarea en la cola"""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la tarea a diccionario"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


class TaskQueue:
    """Gestor de cola de tareas"""
    
    def __init__(self, use_celery: bool = True, redis_url: Optional[str] = None):
        """
        Args:
            use_celery: Usar Celery si está disponible
            redis_url: URL de Redis para el broker
        """
        self.use_celery = use_celery
        self.redis_url = redis_url
        self.tasks: Dict[str, Task] = {}
        self._celery_app = None
        
        if use_celery:
            try:
                from celery import Celery
                self._celery_app = Celery(
                    'suno_clone_ai',
                    broker=redis_url or 'redis://localhost:6379/0',
                    backend=redis_url or 'redis://localhost:6379/0'
                )
                logger.info("Celery initialized")
            except ImportError:
                logger.warning("Celery not available, using in-memory queue")
                self.use_celery = False
    
    def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        task_id: Optional[str] = None
    ) -> str:
        """
        Agrega una tarea a la cola
        
        Args:
            task_type: Tipo de tarea
            payload: Datos de la tarea
            priority: Prioridad
            max_retries: Número máximo de reintentos
            task_id: ID opcional de la tarea
        
        Returns:
            ID de la tarea
        """
        import uuid
        task_id = task_id or str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            status=TaskStatus.QUEUED
        )
        
        self.tasks[task_id] = task
        
        if self.use_celery and self._celery_app:
            # Enviar a Celery
            try:
                celery_task = self._celery_app.send_task(
                    task_type,
                    args=[payload],
                    task_id=task_id,
                    priority=priority.value
                )
                logger.info(f"Task {task_id} enqueued in Celery")
            except Exception as e:
                logger.error(f"Error enqueueing task in Celery: {e}")
                task.status = TaskStatus.FAILED
                task.error = str(e)
        else:
            # Procesar en memoria (para desarrollo)
            logger.info(f"Task {task_id} enqueued in memory")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtiene una tarea por ID"""
        return self.tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ):
        """Actualiza el estado de una tarea"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            
            if status == TaskStatus.PROCESSING:
                task.started_at = datetime.now()
            elif status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now()
                task.result = result
            elif status == TaskStatus.FAILED:
                task.error = error
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    logger.info(f"Task {task_id} will retry ({task.retry_count}/{task.max_retries})")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancela una tarea"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                task.status = TaskStatus.CANCELLED
                return True
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la cola"""
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "tasks_by_status": status_counts,
            "pending": status_counts.get(TaskStatus.PENDING.value, 0),
            "queued": status_counts.get(TaskStatus.QUEUED.value, 0),
            "processing": status_counts.get(TaskStatus.PROCESSING.value, 0),
            "completed": status_counts.get(TaskStatus.COMPLETED.value, 0),
            "failed": status_counts.get(TaskStatus.FAILED.value, 0)
        }
    
    def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> list[Task]:
        """Obtiene tareas por estado"""
        tasks = [task for task in self.tasks.values() if task.status == status]
        return sorted(tasks, key=lambda x: x.created_at, reverse=True)[:limit]


# Instancia global
_task_queue: Optional[TaskQueue] = None


def get_task_queue(use_celery: bool = True, redis_url: Optional[str] = None) -> TaskQueue:
    """Obtiene la instancia global de la cola de tareas"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(use_celery=use_celery, redis_url=redis_url)
    return _task_queue


def process_sqs_message(message_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa un mensaje de SQS
    
    Args:
        message_body: Cuerpo del mensaje de SQS
        
    Returns:
        Resultado del procesamiento
    """
    try:
        task_type = message_body.get("task_type", "unknown")
        payload = message_body.get("payload", {})
        task_id = message_body.get("task_id")
        
        logger.info(f"Processing SQS message: task_type={task_type}, task_id={task_id}")
        
        # Obtener cola de tareas
        queue = get_task_queue(use_celery=False)  # No usar Celery en Lambda
        
        # Procesar según tipo de tarea
        if task_type == "generate_music":
            from core.music_generator import get_music_generator
            generator = get_music_generator()
            result = generator.generate(
                prompt=payload.get("prompt"),
                duration=payload.get("duration", 30),
                **payload.get("params", {})
            )
            return {"status": "success", "result": result}
        
        elif task_type == "process_audio":
            from core.audio_processor import AudioProcessor
            processor = AudioProcessor()
            result = processor.process(payload.get("audio_path"))
            return {"status": "success", "result": result}
        
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"status": "error", "error": f"Unknown task type: {task_type}"}
            
    except Exception as e:
        logger.error(f"Error processing SQS message: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
