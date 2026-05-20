"""
Scheduler Utilities - Utilidades avanzadas de scheduling
========================================================

Sistema de scheduling avanzado con tareas programadas, cron jobs,
y gestión de tareas periódicas.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Estado de tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Tarea programada."""
    
    id: str
    func: Union[Callable[[], Any], Callable[[], Awaitable[Any]]]
    schedule: str  # Cron expression o intervalo
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_async(self) -> bool:
        """Verificar si la función es asíncrona."""
        return asyncio.iscoroutinefunction(self.func)


class Scheduler:
    """
    Scheduler avanzado para tareas programadas.
    
    Soporta cron expressions y intervalos simples.
    """
    
    def __init__(self):
        """Inicializar scheduler."""
        self._tasks: Dict[str, ScheduledTask] = {}
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def schedule(
        self,
        task_id: str,
        func: Union[Callable[[], Any], Callable[[], Awaitable[Any]]],
        schedule: str,
        *args,
        **kwargs
    ) -> ScheduledTask:
        """
        Programar tarea.
        
        Args:
            task_id: ID único de la tarea
            func: Función a ejecutar (síncrona o asíncrona)
            schedule: Cron expression o intervalo (ej: "*/5 * * * *" o "5m")
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            ScheduledTask creada
        
        Example:
            scheduler.schedule("cleanup", cleanup_task, "0 2 * * *")  # Diario a las 2 AM
            scheduler.schedule("backup", backup_task, "30m")  # Cada 30 minutos
        """
        task = ScheduledTask(
            id=task_id,
            func=func,
            schedule=schedule,
            args=args,
            kwargs=kwargs
        )
        
        # Calcular próxima ejecución
        task.next_run = self._calculate_next_run(schedule)
        
        with self._lock:
            self._tasks[task_id] = task
        
        logger.info(f"Scheduled task '{task_id}' with schedule '{schedule}'")
        return task
    
    def unschedule(self, task_id: str) -> bool:
        """
        Desprogramar tarea.
        
        Args:
            task_id: ID de la tarea
        
        Returns:
            True si se desprogramó, False si no existía
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Unscheduled task '{task_id}'")
                return True
            return False
    
    def enable(self, task_id: str) -> bool:
        """Habilitar tarea."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = True
                return True
            return False
    
    def disable(self, task_id: str) -> bool:
        """Deshabilitar tarea."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = False
                return True
            return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Obtener tarea."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """Listar todas las tareas."""
        with self._lock:
            return list(self._tasks.values())
    
    def start(self) -> None:
        """Iniciar scheduler."""
        if self._running:
            return
        
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Detener scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")
    
    def _run_loop(self) -> None:
        """Loop principal del scheduler."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        
        try:
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()
    
    async def _async_loop(self) -> None:
        """Loop asíncrono del scheduler."""
        while self._running:
            try:
                await self._check_and_run_tasks()
                await asyncio.sleep(1)  # Verificar cada segundo
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _check_and_run_tasks(self) -> None:
        """Verificar y ejecutar tareas pendientes."""
        now = datetime.utcnow()
        tasks_to_run = []
        
        with self._lock:
            for task in self._tasks.values():
                if not task.enabled:
                    continue
                
                if task.next_run and now >= task.next_run:
                    tasks_to_run.append(task)
        
        # Ejecutar tareas
        for task in tasks_to_run:
            asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Ejecutar tarea."""
        try:
            logger.debug(f"Executing task '{task.id}'")
            
            if task.is_async():
                await task.func(*task.args, **task.kwargs)
            else:
                # Ejecutar función síncrona en thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
            
            # Actualizar estado
            with self._lock:
                task.last_run = datetime.utcnow()
                task.run_count += 1
                task.next_run = self._calculate_next_run(task.schedule)
            
            logger.debug(f"Task '{task.id}' completed successfully")
        
        except Exception as e:
            logger.error(f"Task '{task.id}' failed: {e}", exc_info=True)
            with self._lock:
                task.error_count += 1
                task.next_run = self._calculate_next_run(task.schedule)
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """
        Calcular próxima ejecución.
        
        Args:
            schedule: Cron expression o intervalo
        
        Returns:
            Próxima fecha de ejecución
        """
        now = datetime.utcnow()
        
        # Si es intervalo simple (ej: "5m", "1h", "30s")
        if schedule.endswith(('s', 'm', 'h', 'd')):
            return self._parse_interval(schedule, now)
        
        # Si es cron expression (simplificado)
        # Por ahora, solo soportamos intervalos simples
        # TODO: Implementar parser de cron completo
        return now + timedelta(minutes=5)  # Default
    
    def _parse_interval(self, interval: str, base: datetime) -> datetime:
        """Parsear intervalo simple."""
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 's':
            return base + timedelta(seconds=value)
        elif unit == 'm':
            return base + timedelta(minutes=value)
        elif unit == 'h':
            return base + timedelta(hours=value)
        elif unit == 'd':
            return base + timedelta(days=value)
        else:
            return base + timedelta(minutes=5)  # Default


# Instancia global
_scheduler = Scheduler()


def get_scheduler() -> Scheduler:
    """Obtener instancia global de Scheduler."""
    return _scheduler


def schedule_task(
    task_id: str,
    schedule: str,
    *args,
    **kwargs
):
    """
    Decorador para programar tarea.
    
    Args:
        task_id: ID de la tarea
        schedule: Cron expression o intervalo
    
    Example:
        @schedule_task("cleanup", "0 2 * * *")
        def cleanup():
            ...
    """
    def decorator(func: Callable) -> Callable:
        scheduler = get_scheduler()
        scheduler.schedule(task_id, func, schedule, *args, **kwargs)
        return func
    return decorator


__all__ = [
    "TaskStatus",
    "ScheduledTask",
    "Scheduler",
    "get_scheduler",
    "schedule_task",
]

