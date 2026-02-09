"""
Sistema de background tasks para Robot Movement AI v2.0
Tareas en segundo plano con gestión de cola y prioridades
"""

import asyncio
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue
import threading


class TaskPriority(int, Enum):
    """Prioridades de tareas"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class BackgroundTask:
    """Representa una tarea en segundo plano"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    max_retries: int = 0
    retry_count: int = 0
    error_handler: Optional[Callable] = None
    
    def __lt__(self, other):
        """Comparar por prioridad y tiempo de creación"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class BackgroundTaskManager:
    """Gestor de tareas en segundo plano"""
    
    def __init__(self, max_workers: int = 4):
        """
        Inicializar gestor de tareas
        
        Args:
            max_workers: Número máximo de workers concurrentes
        """
        self.max_workers = max_workers
        self.task_queue: PriorityQueue = PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[BackgroundTask] = []
        self.failed_tasks: List[BackgroundTask] = []
        self.workers: List[asyncio.Task] = []
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Iniciar workers"""
        if self.running:
            return
        
        self.running = True
        
        # Crear workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self):
        """Detener workers"""
        self.running = False
        
        # Esperar a que terminen las tareas actuales
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def _worker(self, name: str):
        """Worker que procesa tareas"""
        while self.running:
            try:
                # Obtener tarea de la cola (con timeout)
                try:
                    task = await asyncio.wait_for(
                        asyncio.to_thread(self.task_queue.get, timeout=1),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Ejecutar tarea
                await self._execute_task(task)
                
            except Exception as e:
                print(f"Error in worker {name}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: BackgroundTask):
        """Ejecutar una tarea"""
        async with self._lock:
            self.running_tasks[task.id] = asyncio.create_task(
                self._run_task(task)
            )
        
        try:
            await self.running_tasks[task.id]
        finally:
            async with self._lock:
                self.running_tasks.pop(task.id, None)
    
    async def _run_task(self, task: BackgroundTask):
        """Ejecutar función de tarea"""
        try:
            import inspect
            
            if inspect.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = await asyncio.to_thread(task.func, *task.args, **task.kwargs)
            
            self.completed_tasks.append(task)
            return result
            
        except Exception as e:
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Reintentar
                await asyncio.to_thread(self.task_queue.put, task)
            else:
                # Falló definitivamente
                self.failed_tasks.append(task)
                
                if task.error_handler:
                    try:
                        if inspect.iscoroutinefunction(task.error_handler):
                            await task.error_handler(task, e)
                        else:
                            task.error_handler(task, e)
                    except Exception as handler_error:
                        print(f"Error in error handler: {handler_error}")
    
    def add_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 0,
        error_handler: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Agregar tarea a la cola
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            priority: Prioridad de la tarea
            max_retries: Número máximo de reintentos
            error_handler: Handler de errores
            **kwargs: Argumentos nombrados
            
        Returns:
            ID de la tarea
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = BackgroundTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            error_handler=error_handler
        )
        
        self.task_queue.put(task)
        return task_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del gestor"""
        return {
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "workers": len(self.workers),
            "running": self.running
        }


# Instancia global
_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Obtener instancia global del gestor de tareas"""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


async def start_background_tasks():
    """Iniciar gestor de tareas en segundo plano"""
    manager = get_task_manager()
    await manager.start()


async def stop_background_tasks():
    """Detener gestor de tareas"""
    manager = get_task_manager()
    await manager.stop()


def add_background_task(
    func: Callable,
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs
) -> str:
    """Helper para agregar tarea en segundo plano"""
    manager = get_task_manager()
    return manager.add_task(func, *args, priority=priority, **kwargs)




