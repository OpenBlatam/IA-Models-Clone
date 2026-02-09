"""
Queue - Sistema de colas para procesamiento asíncrono
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridades de tareas"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueueTask:
    """Tarea en cola"""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    retries: int = 0
    max_retries: int = 3


class TaskQueue:
    """Cola de tareas"""

    def __init__(self, max_workers: int = 5):
        """
        Inicializar cola de tareas.

        Args:
            max_workers: Número máximo de workers
        """
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_workers = max_workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.processors: Dict[str, Callable] = {}

    def register_processor(self, task_type: str, processor: Callable):
        """
        Registrar procesador para un tipo de tarea.

        Args:
            task_type: Tipo de tarea
            processor: Función procesadora
        """
        self.processors[task_type] = processor
        logger.info(f"Procesador registrado para: {task_type}")

    async def enqueue(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Agregar tarea a la cola.

        Args:
            task_type: Tipo de tarea
            data: Datos de la tarea
            priority: Prioridad

        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        task = QueueTask(
            id=task_id,
            task_type=task_type,
            data=data,
            priority=priority
        )
        
        await self.queue.put(task)
        logger.info(f"Tarea encolada: {task_id} ({task_type})")
        return task_id

    async def start(self):
        """Iniciar workers"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Cola iniciada con {self.max_workers} workers")

    async def stop(self):
        """Detener workers"""
        self.running = False
        
        # Esperar a que se completen las tareas
        await self.queue.join()
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Cola detenida")

    async def _worker(self, worker_id: str):
        """Worker que procesa tareas"""
        logger.info(f"Worker iniciado: {worker_id}")
        
        while self.running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                processor = self.processors.get(task.task_type)
                if not processor:
                    logger.warning(f"No hay procesador para: {task.task_type}")
                    self.queue.task_done()
                    continue
                
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(task.data)
                    else:
                        processor(task.data)
                    
                    logger.info(f"Tarea completada: {task.id}")
                except Exception as e:
                    logger.error(f"Error procesando tarea {task.id}: {e}")
                    task.retries += 1
                    
                    if task.retries < task.max_retries:
                        await self.queue.put(task)
                    else:
                        logger.error(f"Tarea fallida después de {task.max_retries} intentos: {task.id}")
                
                self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en worker {worker_id}: {e}")

    def get_queue_size(self) -> int:
        """Obtener tamaño de la cola"""
        return self.queue.qsize()

