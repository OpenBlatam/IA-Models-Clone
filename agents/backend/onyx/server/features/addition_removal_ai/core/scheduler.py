"""
Scheduler - Sistema de tareas programadas
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Tarea programada"""
    id: str
    name: str
    func: Callable
    schedule: str  # cron-like o interval
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Programador de tareas"""

    def __init__(self):
        """Inicializar el programador"""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    def schedule_task(
        self,
        name: str,
        func: Callable,
        schedule: str,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Programar una tarea.

        Args:
            name: Nombre de la tarea
            func: Función a ejecutar
            schedule: Horario (cron-like o interval en segundos)
            enabled: Si está habilitada
            metadata: Metadatos adicionales

        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        # Calcular próxima ejecución
        next_run = self._calculate_next_run(schedule)
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            func=func,
            schedule=schedule,
            enabled=enabled,
            next_run=next_run,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        logger.info(f"Tarea programada: {name} (ID: {task_id})")
        
        return task_id

    def _calculate_next_run(self, schedule: str) -> datetime:
        """
        Calcular próxima ejecución.

        Args:
            schedule: Horario

        Returns:
            Fecha de próxima ejecución
        """
        # Si es un número, es un intervalo en segundos
        try:
            interval = int(schedule)
            return datetime.utcnow() + timedelta(seconds=interval)
        except ValueError:
            # Aquí se podría implementar parsing de cron
            # Por ahora, asumimos intervalos
            return datetime.utcnow() + timedelta(hours=1)

    async def start(self):
        """Iniciar el programador"""
        if self.running:
            return
        
        self.running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Programador de tareas iniciado")

    async def stop(self):
        """Detener el programador"""
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Programador de tareas detenido")

    async def _scheduler_loop(self):
        """Loop principal del programador"""
        while self.running:
            try:
                now = datetime.utcnow()
                
                for task in self.tasks.values():
                    if not task.enabled:
                        continue
                    
                    if task.next_run and now >= task.next_run:
                        await self._execute_task(task)
                
                # Esperar un segundo antes de revisar de nuevo
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error en scheduler loop: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: ScheduledTask):
        """
        Ejecutar una tarea.

        Args:
            task: Tarea a ejecutar
        """
        try:
            logger.info(f"Ejecutando tarea: {task.name}")
            
            if asyncio.iscoroutinefunction(task.func):
                await task.func(**task.metadata)
            else:
                task.func(**task.metadata)
            
            task.last_run = datetime.utcnow()
            task.run_count += 1
            task.next_run = self._calculate_next_run(task.schedule)
            
            logger.info(f"Tarea completada: {task.name} (ejecuciones: {task.run_count})")
        
        except Exception as e:
            logger.error(f"Error ejecutando tarea {task.name}: {e}")

    def enable_task(self, task_id: str):
        """Habilitar una tarea"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            logger.info(f"Tarea habilitada: {task_id}")

    def disable_task(self, task_id: str):
        """Deshabilitar una tarea"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Tarea deshabilitada: {task_id}")

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las tareas.

        Returns:
            Lista de tareas
        """
        return [
            {
                "id": task.id,
                "name": task.name,
                "schedule": task.schedule,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "next_run": task.next_run.isoformat() if task.next_run else None,
                "run_count": task.run_count,
                "metadata": task.metadata
            }
            for task in self.tasks.values()
        ]

    def remove_task(self, task_id: str):
        """
        Eliminar una tarea.

        Args:
            task_id: ID de la tarea
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Tarea eliminada: {task_id}")






