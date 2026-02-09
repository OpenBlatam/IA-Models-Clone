"""
Task Scheduler System
=====================

Sistema de programación de tareas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Tarea programada."""
    task_id: str
    name: str
    func: Callable
    schedule_type: str  # "once", "interval", "cron"
    schedule_value: Any
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0
    enabled: bool = True


class TaskScheduler:
    """
    Programador de tareas.
    
    Programa y ejecuta tareas de forma automática.
    """
    
    def __init__(self):
        """Inicializar programador de tareas."""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
    
    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        schedule_type: str = "interval",
        schedule_value: Any = 60,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> ScheduledTask:
        """
        Agregar tarea programada.
        
        Args:
            task_id: ID único de la tarea
            name: Nombre de la tarea
            func: Función a ejecutar
            schedule_type: Tipo de programación ("once", "interval", "cron")
            schedule_value: Valor de programación
            args: Argumentos posicionales
            kwargs: Argumentos con nombre
            enabled: Si está habilitada
            
        Returns:
            Tarea programada
        """
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled
        )
        
        # Calcular próxima ejecución
        task.next_run = self._calculate_next_run(task)
        
        self.tasks[task_id] = task
        logger.info(f"Added scheduled task: {name} ({task_id})")
        
        return task
    
    def _calculate_next_run(self, task: ScheduledTask) -> str:
        """Calcular próxima ejecución."""
        now = datetime.now()
        
        if task.schedule_type == "once":
            if task.run_count == 0:
                return now.isoformat()
            return None
        
        elif task.schedule_type == "interval":
            # schedule_value en segundos
            next_run = now + timedelta(seconds=task.schedule_value)
            return next_run.isoformat()
        
        elif task.schedule_type == "cron":
            # Implementación básica de cron
            # En producción, usar biblioteca como croniter
            return now.isoformat()
        
        return None
    
    async def start(self) -> None:
        """Iniciar programador."""
        if self.running:
            return
        
        self.running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Task scheduler started")
    
    async def stop(self) -> None:
        """Detener programador."""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")
    
    async def _run_loop(self) -> None:
        """Loop principal del programador."""
        while self.running:
            try:
                now = datetime.now()
                
                # Ejecutar tareas pendientes
                for task in self.tasks.values():
                    if not task.enabled or task.status == TaskStatus.RUNNING:
                        continue
                    
                    if task.next_run:
                        next_run_time = datetime.fromisoformat(task.next_run)
                        if now >= next_run_time:
                            asyncio.create_task(self._execute_task(task))
                
                await asyncio.sleep(1)  # Verificar cada segundo
            
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Ejecutar tarea."""
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now().isoformat()
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.run_count += 1
            
            # Calcular próxima ejecución
            task.next_run = self._calculate_next_run(task)
            
            logger.info(f"Task completed: {task.name}")
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_count += 1
            logger.error(f"Task failed: {task.name} - {e}")
        
        finally:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PENDING
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Obtener tarea por ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """Listar todas las tareas."""
        return list(self.tasks.values())
    
    def enable_task(self, task_id: str) -> bool:
        """Habilitar tarea."""
        task = self.tasks.get(task_id)
        if task:
            task.enabled = True
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Deshabilitar tarea."""
        task = self.tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False
    
    def remove_task(self, task_id: str) -> bool:
        """Eliminar tarea."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False


# Instancia global
_task_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    """Obtener instancia global del programador de tareas."""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = TaskScheduler()
    return _task_scheduler






