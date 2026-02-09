"""
Scheduler Service - Sistema de tareas programadas
==================================================

Sistema para ejecutar tareas en segundo plano.
"""

import logging
import asyncio
from typing import Callable, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Tarea programada"""
    id: str
    name: str
    func: Callable
    interval_seconds: int
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True


class SchedulerService:
    """Servicio de scheduler"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        logger.info("SchedulerService initialized")
    
    def schedule_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        interval_seconds: int
    ) -> ScheduledTask:
        """Programar tarea"""
        task = ScheduledTask(
            id=task_id,
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            next_run=datetime.now() + timedelta(seconds=interval_seconds)
        )
        
        self.tasks[task_id] = task
        logger.info(f"Task scheduled: {name} (every {interval_seconds}s)")
        return task
    
    async def run_scheduler(self):
        """Ejecutar scheduler"""
        self.running = True
        logger.info("Scheduler started")
        
        while self.running:
            now = datetime.now()
            
            for task in self.tasks.values():
                if not task.enabled:
                    continue
                
                if task.next_run and now >= task.next_run:
                    try:
                        # Ejecutar tarea
                        if asyncio.iscoroutinefunction(task.func):
                            await task.func()
                        else:
                            task.func()
                        
                        task.last_run = now
                        task.next_run = now + timedelta(seconds=task.interval_seconds)
                        logger.info(f"Task executed: {task.name}")
                    except Exception as e:
                        logger.error(f"Error executing task {task.name}: {e}")
            
            # Esperar 1 segundo antes de revisar de nuevo
            await asyncio.sleep(1)
    
    def stop_scheduler(self):
        """Detener scheduler"""
        self.running = False
        logger.info("Scheduler stopped")
    
    def enable_task(self, task_id: str):
        """Habilitar tarea"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
    
    def disable_task(self, task_id: str):
        """Deshabilitar tarea"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False




