"""
Procesador Paralelo de Tareas - Procesamiento concurrente avanzado.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import deque

from config.logging_config import get_logger
from core.constants import TaskStatus

logger = get_logger(__name__)


class TaskQueue:
    """Cola de tareas con prioridades."""
    
    def __init__(self):
        """Inicializar cola."""
        self.queue: deque = deque()
        self.priority_queue: deque = deque()
        self.processing: Dict[str, asyncio.Task] = {}
    
    def add_task(self, task: Dict[str, Any], priority: int = 0) -> None:
        """
        Agregar tarea a la cola.
        
        Args:
            task: Tarea a agregar
            priority: Prioridad (mayor = más importante)
        """
        task_item = {
            "task": task,
            "priority": priority,
            "added_at": datetime.now()
        }
        
        if priority > 0:
            self.priority_queue.append(task_item)
            # Mantener ordenado por prioridad
            self.priority_queue = deque(sorted(self.priority_queue, key=lambda x: x["priority"], reverse=True))
        else:
            self.queue.append(task_item)
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Obtener siguiente tarea.
        
        Returns:
            Tarea o None si la cola está vacía
        """
        if self.priority_queue:
            return self.priority_queue.popleft()["task"]
        if self.queue:
            return self.queue.popleft()["task"]
        return None
    
    def is_empty(self) -> bool:
        """Verificar si la cola está vacía."""
        return len(self.queue) == 0 and len(self.priority_queue) == 0
    
    def size(self) -> int:
        """Obtener tamaño de la cola."""
        return len(self.queue) + len(self.priority_queue)


class ParallelTaskProcessor:
    """Procesador paralelo de tareas."""
    
    def __init__(
        self,
        max_workers: int = 5,
        task_processor_func: Optional[Callable] = None
    ):
        """
        Inicializar procesador paralelo.
        
        Args:
            max_workers: Número máximo de workers paralelos
            task_processor_func: Función para procesar tareas
        """
        self.max_workers = max_workers
        self.task_processor_func = task_processor_func
        self.queue = TaskQueue()
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.stats = {
            "total_processed": 0,
            "total_succeeded": 0,
            "total_failed": 0,
            "active_workers": 0
        }
    
    async def start(self) -> None:
        """Iniciar procesador."""
        if self.running:
            return
        
        self.running = True
        
        # Iniciar workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Procesador paralelo iniciado con {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Detener procesador."""
        self.running = False
        
        # Esperar a que terminen los workers
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Procesador paralelo detenido")
    
    async def add_task(self, task: Dict[str, Any], priority: int = 0) -> None:
        """
        Agregar tarea para procesar.
        
        Args:
            task: Tarea a procesar
            priority: Prioridad
        """
        self.queue.add_task(task, priority)
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Loop de worker."""
        while self.running:
            try:
                task = self.queue.get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                self.stats["active_workers"] += 1
                self.stats["total_processed"] += 1
                
                try:
                    if self.task_processor_func:
                        await self.task_processor_func(task)
                    self.stats["total_succeeded"] += 1
                except Exception as e:
                    logger.error(f"Error procesando tarea en {worker_id}: {e}", exc_info=True)
                    self.stats["total_failed"] += 1
                finally:
                    self.stats["active_workers"] -= 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en worker loop {worker_id}: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "queue_size": self.queue.size(),
            "max_workers": self.max_workers,
            "running": self.running
        }



