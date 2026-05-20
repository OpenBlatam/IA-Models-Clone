"""
Advanced Concurrency Utilities
===============================
Utilidades avanzadas para concurrencia.
"""

import asyncio
from typing import Any, Callable, List, Dict, Optional, Coroutine
from datetime import datetime
from collections import deque
import time

from .logger import get_logger

logger = get_logger(__name__)


class SemaphorePool:
    """Pool de semáforos para control de concurrencia."""
    
    def __init__(self, total_slots: int = 10):
        """
        Inicializar semaphore pool.
        
        Args:
            total_slots: Número total de slots disponibles
        """
        self.semaphore = asyncio.Semaphore(total_slots)
        self.total_slots = total_slots
        self.active_tasks = 0
        self.waiting_tasks = 0
    
    async def acquire(self):
        """Adquirir slot."""
        self.waiting_tasks += 1
        await self.semaphore.acquire()
        self.waiting_tasks -= 1
        self.active_tasks += 1
    
    def release(self):
        """Liberar slot."""
        self.active_tasks -= 1
        self.semaphore.release()
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_slots": self.total_slots,
            "active_tasks": self.active_tasks,
            "waiting_tasks": self.waiting_tasks,
            "available_slots": self.total_slots - self.active_tasks
        }


class TaskQueue:
    """Cola de tareas con prioridad."""
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Inicializar task queue.
        
        Args:
            max_size: Tamaño máximo (None = ilimitado)
        """
        self.queue = asyncio.Queue(maxsize=max_size or 0)
        self.pending_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=1000)
    
    async def enqueue(
        self,
        task_id: str,
        coro: Coroutine,
        priority: int = 0
    ):
        """
        Encolar tarea.
        
        Args:
            task_id: ID de la tarea
            coro: Coroutine a ejecutar
            priority: Prioridad (mayor = más prioritario)
        """
        await self.queue.put((priority, task_id, coro))
    
    async def process_next(self) -> Optional[Dict[str, Any]]:
        """
        Procesar siguiente tarea.
        
        Returns:
            Resultado de la tarea
        """
        try:
            priority, task_id, coro = await self.queue.get()
            
            task = asyncio.create_task(coro)
            self.pending_tasks[task_id] = task
            
            try:
                result = await task
                self.completed_tasks.append({
                    "task_id": task_id,
                    "priority": priority,
                    "completed_at": datetime.now().isoformat(),
                    "status": "success"
                })
                return {"task_id": task_id, "result": result, "status": "success"}
            except Exception as e:
                self.failed_tasks.append({
                    "task_id": task_id,
                    "priority": priority,
                    "failed_at": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "failed"
                })
                return {"task_id": task_id, "error": str(e), "status": "failed"}
            finally:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
        
        except asyncio.CancelledError:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "queue_size": self.queue.qsize(),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks)
        }


class RateLimiter:
    """Rate limiter avanzado con ventana deslizante."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Inicializar rate limiter.
        
        Args:
            max_calls: Número máximo de llamadas
            time_window: Ventana de tiempo en segundos
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Intentar adquirir permiso.
        
        Returns:
            True si se permite, False si se excede el límite
        """
        async with self._lock:
            now = time.time()
            
            # Eliminar llamadas fuera de la ventana
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            # Verificar si hay espacio
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    async def wait(self):
        """Esperar hasta que haya espacio disponible."""
        while not await self.acquire():
            # Calcular tiempo de espera
            if self.calls:
                oldest_call = self.calls[0]
                wait_time = self.time_window - (time.time() - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        now = time.time()
        recent_calls = [
            call for call in self.calls
            if call >= now - self.time_window
        ]
        
        return {
            "max_calls": self.max_calls,
            "time_window": self.time_window,
            "current_calls": len(recent_calls),
            "available_calls": self.max_calls - len(recent_calls)
        }


async def run_with_timeout(
    coro: Coroutine,
    timeout: float,
    default: Any = None
) -> Any:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        default: Valor por defecto si timeout
        
    Returns:
        Resultado o valor por defecto
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout}s")
        return default


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int = 10
) -> List[Any]:
    """
    Ejecutar coroutines con límite de concurrencia.
    
    Args:
        coros: Lista de coroutines
        limit: Límite de concurrencia
        
    Returns:
        Resultados
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Coroutine):
        async with semaphore:
            return await coro
    
    tasks = [bounded_coro(coro) for coro in coros]
    return await asyncio.gather(*tasks)



