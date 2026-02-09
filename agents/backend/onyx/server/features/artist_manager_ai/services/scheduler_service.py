"""
Scheduler Service
=================

Servicio de programación avanzada de tareas.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Tarea programada."""
    id: str
    name: str
    func: Callable
    schedule: str  # Cron expression o timedelta
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "status": self.status.value,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count
        }


class SchedulerService:
    """Servicio de programación."""
    
    def __init__(self):
        """Inicializar servicio de programación."""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._logger = logger
    
    def add_task(
        self,
        name: str,
        func: Callable,
        schedule: str,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> ScheduledTask:
        """
        Agregar tarea programada.
        
        Args:
            name: Nombre de la tarea
            func: Función a ejecutar
            schedule: Expresión cron o timedelta
            args: Argumentos posicionales
            kwargs: Argumentos nombrados
        
        Returns:
            Tarea programada
        """
        import uuid
        
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            func=func,
            schedule=schedule,
            args=args,
            kwargs=kwargs or {}
        )
        
        # Calcular próxima ejecución
        task.next_run = self._calculate_next_run(schedule)
        task.status = TaskStatus.SCHEDULED
        
        self.tasks[task.id] = task
        self._logger.info(f"Added scheduled task: {name} (id: {task.id})")
        
        return task
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """
        Calcular próxima ejecución.
        
        Args:
            schedule: Expresión de programación
        
        Returns:
            Próxima ejecución
        """
        # Por ahora, implementación simple
        # En producción usaría croniter o similar
        if schedule.startswith("every_"):
            # Formato: "every_5_minutes", "every_1_hour", etc.
            parts = schedule.split("_")
            if len(parts) >= 3:
                value = int(parts[1])
                unit = parts[2]
                
                if unit.startswith("minute"):
                    return datetime.now() + timedelta(minutes=value)
                elif unit.startswith("hour"):
                    return datetime.now() + timedelta(hours=value)
                elif unit.startswith("day"):
                    return datetime.now() + timedelta(days=value)
        
        # Default: 1 hora
        return datetime.now() + timedelta(hours=1)
    
    async def start(self):
        """Iniciar scheduler."""
        if self.running:
            self._logger.warning("Scheduler already running")
            return
        
        self.running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._logger.info("Scheduler started")
    
    async def stop(self):
        """Detener scheduler."""
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Loop principal del scheduler."""
        while self.running:
            try:
                now = datetime.now()
                
                for task in self.tasks.values():
                    if task.status == TaskStatus.SCHEDULED and task.next_run and task.next_run <= now:
                        await self._execute_task(task)
                
                # Esperar 1 minuto antes de revisar de nuevo
                await asyncio.sleep(60)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _execute_task(self, task: ScheduledTask):
        """Ejecutar tarea."""
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.run_count += 1
            
            # Recalcular próxima ejecución
            task.next_run = self._calculate_next_run(task.schedule)
            task.status = TaskStatus.SCHEDULED
            
            self._logger.info(f"Task {task.name} completed successfully")
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_count += 1
            self._logger.error(f"Task {task.name} failed: {str(e)}")
            
            # Reintentar después de un tiempo
            task.next_run = datetime.now() + timedelta(minutes=5)
            task.status = TaskStatus.SCHEDULED
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Obtener tarea."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """Listar todas las tareas."""
        return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancelar tarea."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            return True
        return False




