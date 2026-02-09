"""
Sistema de colas asíncronas para procesamiento
"""

import asyncio
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    """Estado de tarea"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    """Prioridad de tarea"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Task:
    """Tarea en cola"""
    id: str
    task_type: str
    data: Dict
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


class AsyncQueue:
    """Cola asíncrona de tareas"""
    
    def __init__(self, max_workers: int = 5):
        """
        Inicializa la cola
        
        Args:
            max_workers: Número máximo de workers
        """
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.workers: List[asyncio.Task] = []
        self.max_workers = max_workers
        self.processors: Dict[str, Callable] = {}
        self.running = False
    
    def register_processor(self, task_type: str, processor: Callable):
        """
        Registra un procesador para un tipo de tarea
        
        Args:
            task_type: Tipo de tarea
            processor: Función procesadora
        """
        self.processors[task_type] = processor
    
    async def enqueue(self, task_type: str, data: Dict,
                     priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Agrega tarea a la cola
        
        Args:
            task_type: Tipo de tarea
            data: Datos de la tarea
            priority: Prioridad
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            task_type=task_type,
            data=data,
            priority=priority
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task)
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Obtiene una tarea por ID"""
        return self.tasks.get(task_id)
    
    async def start_workers(self):
        """Inicia los workers"""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop_workers(self):
        """Detiene los workers"""
        self.running = False
        
        # Esperar a que terminen las tareas pendientes
        await self.queue.join()
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def _worker(self, name: str):
        """Worker que procesa tareas"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                if task.task_type not in self.processors:
                    task.status = TaskStatus.FAILED
                    task.error = f"Procesador no encontrado: {task.task_type}"
                    self.queue.task_done()
                    continue
                
                processor = self.processors[task.task_type]
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now().isoformat()
                
                try:
                    result = await processor(task.data)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now().isoformat()
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now().isoformat()
                
                self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error en worker {name}: {e}")
                continue
    
    def get_queue_stats(self) -> Dict:
        """Obtiene estadísticas de la cola"""
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        processing = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            "queue_size": self.queue.qsize(),
            "total_tasks": len(self.tasks),
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "workers": len(self.workers),
            "running": self.running
        }

