"""
Servicio de Cola de Tareas con Prioridades y Scheduling.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import heapq

from config.logging_config import get_logger

logger = get_logger(__name__)


class TaskPriority(int, Enum):
    """Prioridades de tareas."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class QueuedTask:
    """Tarea en cola."""
    task_id: str
    task_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_at: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparar por prioridad y tiempo."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Mayor prioridad primero
        if self.scheduled_at and other.scheduled_at:
            return self.scheduled_at < other.scheduled_at  # Más temprano primero
        return self.created_at < other.created_at


class QueueService:
    """
    Servicio de cola de tareas con prioridades y mejoras.
    
    Attributes:
        queue: Cola de tareas pendientes
        processing: Tareas en procesamiento
        completed: Tareas completadas
        failed: Tareas fallidas
        max_size: Tamaño máximo de la cola
        stats: Estadísticas del servicio
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Inicializar servicio de cola con validaciones.
        
        Args:
            max_size: Tamaño máximo de la cola (opcional, debe ser entero positivo si se proporciona)
            
        Raises:
            ValueError: Si max_size es inválido
        """
        # Validación
        if max_size is not None:
            if not isinstance(max_size, int) or max_size < 1:
                raise ValueError(f"max_size debe ser un entero positivo si se proporciona, recibido: {max_size}")
        
        self.queue: List[QueuedTask] = []
        self.processing: Dict[str, QueuedTask] = {}
        self.completed: Dict[str, QueuedTask] = {}
        self.failed: Dict[str, QueuedTask] = {}
        self.max_size = max_size
        self.stats = {
            "total_enqueued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "total_completed": 0,
            "current_queue_size": 0,
            "current_processing": 0
        }
        
        logger.info(f"✅ QueueService inicializado: max_size={max_size or 'unlimited'}")
    
    def enqueue(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Agregar tarea a la cola con validaciones.
        
        Args:
            task_id: ID único de la tarea (debe ser string no vacío)
            task_data: Datos de la tarea (debe ser diccionario)
            priority: Prioridad (debe ser TaskPriority)
            scheduled_at: Fecha programada (opcional, debe ser datetime si se proporciona)
            max_retries: Número máximo de reintentos (debe ser entero no negativo)
            metadata: Metadata adicional (opcional, debe ser diccionario si se proporciona)
            
        Returns:
            True si se agregó exitosamente, False si la cola está llena
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task_id debe ser un string no vacío, recibido: {task_id}")
        
        if not isinstance(task_data, dict):
            raise ValueError(f"task_data debe ser un diccionario, recibido: {type(task_data)}")
        
        if not isinstance(priority, TaskPriority):
            raise ValueError(f"priority debe ser un TaskPriority, recibido: {type(priority)}")
        
        if scheduled_at is not None:
            if not isinstance(scheduled_at, datetime):
                raise ValueError(f"scheduled_at debe ser un datetime si se proporciona, recibido: {type(scheduled_at)}")
        
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries debe ser un entero no negativo, recibido: {max_retries}")
        
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata debe ser un diccionario si se proporciona, recibido: {type(metadata)}")
        
        task_id = task_id.strip()
        
        if self.max_size and len(self.queue) >= self.max_size:
            logger.warning(f"⚠️  Cola llena ({len(self.queue)}/{self.max_size}), rechazando tarea {task_id}")
            return False
        
        try:
            queued_task = QueuedTask(
                task_id=task_id,
                task_data=task_data,
                priority=priority,
                scheduled_at=scheduled_at,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            heapq.heappush(self.queue, queued_task)
            self.stats["total_enqueued"] += 1
            self.stats["current_queue_size"] = len(self.queue)
            
            logger.info(
                f"✅ Tarea {task_id} agregada a la cola "
                f"(prioridad: {priority.name}, scheduled_at: {scheduled_at or 'now'}, "
                f"max_retries: {max_retries}, queue_size: {len(self.queue)})"
            )
            return True
        except Exception as e:
            logger.error(f"Error al agregar tarea {task_id} a la cola: {e}", exc_info=True)
            raise ValueError(f"Error al agregar tarea: {e}") from e
    
    def dequeue(self) -> Optional[QueuedTask]:
        """
        Obtener siguiente tarea de la cola.
        
        Returns:
            Tarea o None si la cola está vacía o no hay tareas listas
        """
        if not self.queue:
            return None
        
        now = datetime.now()
        
        # Buscar tarea lista (sin scheduled_at o scheduled_at <= now)
        ready_tasks = []
        while self.queue:
            task = heapq.heappop(self.queue)
            if not task.scheduled_at or task.scheduled_at <= now:
                self.stats["current_queue_size"] = len(self.queue)
                return task
            ready_tasks.append(task)
        
        # Re-agregar tareas no listas
        for task in ready_tasks:
            heapq.heappush(self.queue, task)
        
        return None
    
    def mark_processing(self, task_id: str) -> bool:
        """
        Marcar tarea como en procesamiento.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            True si se marcó exitosamente
        """
        task = self.dequeue()
        if task and task.task_id == task_id:
            self.processing[task_id] = task
            self.stats["current_processing"] = len(self.processing)
            return True
        return False
    
    def mark_completed(self, task_id: str) -> None:
        """
        Marcar tarea como completada.
        
        Args:
            task_id: ID de la tarea
        """
        if task_id in self.processing:
            task = self.processing.pop(task_id)
            self.completed[task_id] = task
            self.stats["total_processed"] += 1
            self.stats["total_completed"] += 1
            self.stats["current_processing"] = len(self.processing)
            logger.info(f"Tarea {task_id} completada")
    
    def mark_failed(self, task_id: str, retry: bool = True) -> bool:
        """
        Marcar tarea como fallida con validaciones.
        
        Args:
            task_id: ID de la tarea (debe ser string no vacío)
            retry: Si reintentar (debe ser bool)
            
        Returns:
            True si se reintentará, False si se agotaron los reintentos o no existe
            
        Raises:
            ValueError: Si task_id es inválido
        """
        # Validación
        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task_id debe ser un string no vacío, recibido: {task_id}")
        
        if not isinstance(retry, bool):
            raise ValueError(f"retry debe ser un bool, recibido: {type(retry)}")
        
        task_id = task_id.strip()
        
        if task_id not in self.processing:
            logger.warning(f"Tarea {task_id} no está en procesamiento")
            return False
        
        task = self.processing.pop(task_id)
        self.stats["current_processing"] = len(self.processing)
        
        if retry and task.retry_count < task.max_retries:
            task.retry_count += 1
            # Re-agregar a la cola con delay exponencial
            delay = timedelta(seconds=2 ** task.retry_count)
            task.scheduled_at = datetime.now() + delay
            heapq.heappush(self.queue, task)
            self.stats["current_queue_size"] = len(self.queue)
            logger.info(
                f"🔄 Tarea {task_id} falló, reintentando "
                f"(intento {task.retry_count}/{task.max_retries}, delay: {delay.total_seconds()}s)"
            )
            return True
        else:
            self.failed[task_id] = task
            self.stats["total_failed"] += 1
            logger.warning(
                f"❌ Tarea {task_id} falló definitivamente después de {task.retry_count} intentos "
                f"(max_retries: {task.max_retries})"
            )
            return False
    
    def get_queue_size(self) -> int:
        """Obtener tamaño de la cola."""
        return len(self.queue)
    
    def get_processing_count(self) -> int:
        """Obtener número de tareas en procesamiento."""
        return len(self.processing)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "queue_size": len(self.queue),
            "processing_count": len(self.processing),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "tasks_by_priority": {
                priority.name: len([t for t in self.queue if t.priority == priority])
                for priority in TaskPriority
            }
        }
    
    def clear(self) -> None:
        """Limpiar cola."""
        self.queue.clear()
        self.processing.clear()
        self.completed.clear()
        self.failed.clear()
        self.stats["current_queue_size"] = 0
        self.stats["current_processing"] = 0
        logger.info("Cola limpiada")

