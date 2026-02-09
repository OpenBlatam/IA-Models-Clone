"""
Scheduler - Sistema de tareas programadas
==========================================

Programa tareas para ejecutarse en momentos específicos.
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Tipos de programación"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScheduledTask:
    """Tarea programada"""
    id: str
    name: str
    command: str
    schedule_type: ScheduleType
    schedule_value: str  # cron expression, interval seconds, etc.
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskScheduler:
    """Programador de tareas"""
    
    def __init__(self, agent):
        self.agent = agent
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Iniciar el scheduler"""
        if self.running:
            return
        
        self.running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("⏰ Task scheduler started")
    
    async def stop(self):
        """Detener el scheduler"""
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("⏰ Task scheduler stopped")
    
    def schedule_task(
        self,
        name: str,
        command: str,
        schedule_type: ScheduleType,
        schedule_value: str,
        max_runs: Optional[int] = None
    ) -> str:
        """Programar una tarea"""
        task_id = f"scheduled_{datetime.now().timestamp()}_{len(self.scheduled_tasks)}"
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            command=command,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            max_runs=max_runs
        )
        
        # Calcular próxima ejecución
        task.next_run = self._calculate_next_run(task)
        
        self.scheduled_tasks[task_id] = task
        logger.info(f"📅 Scheduled task: {name} (next run: {task.next_run})")
        
        return task_id
    
    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """Calcular próxima ejecución"""
        now = datetime.now()
        
        if task.schedule_type == ScheduleType.ONCE:
            # Parsear fecha/hora
            try:
                return datetime.fromisoformat(task.schedule_value)
            except:
                return now + timedelta(seconds=60)
        
        elif task.schedule_type == ScheduleType.INTERVAL:
            # Intervalo en segundos
            try:
                interval = int(task.schedule_value)
                return now + timedelta(seconds=interval)
            except:
                return now + timedelta(seconds=60)
        
        elif task.schedule_type == ScheduleType.DAILY:
            # Ejecutar diariamente a una hora específica
            try:
                hour, minute = map(int, task.schedule_value.split(':'))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run
            except:
                return now + timedelta(days=1)
        
        elif task.schedule_type == ScheduleType.WEEKLY:
            # Ejecutar semanalmente
            return now + timedelta(weeks=1)
        
        elif task.schedule_type == ScheduleType.MONTHLY:
            # Ejecutar mensualmente
            return now + timedelta(days=30)
        
        else:
            return now + timedelta(seconds=60)
    
    async def _scheduler_loop(self):
        """Loop principal del scheduler"""
        while self.running:
            try:
                now = datetime.now()
                
                # Verificar tareas programadas
                for task in list(self.scheduled_tasks.values()):
                    if not task.enabled:
                        continue
                    
                    # Verificar si debe ejecutarse
                    if task.next_run and task.next_run <= now:
                        await self._execute_scheduled_task(task)
                
                # Esperar un segundo antes de verificar de nuevo
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_scheduled_task(self, task: ScheduledTask):
        """Ejecutar tarea programada"""
        try:
            logger.info(f"⏰ Executing scheduled task: {task.name}")
            
            # Agregar tarea al agente
            await self.agent.add_task(task.command)
            
            # Actualizar estadísticas
            task.last_run = datetime.now()
            task.run_count += 1
            task.next_run = self._calculate_next_run(task)
            
            # Verificar si alcanzó el máximo de ejecuciones
            if task.max_runs and task.run_count >= task.max_runs:
                task.enabled = False
                logger.info(f"📅 Scheduled task {task.name} reached max runs ({task.max_runs})")
            
        except Exception as e:
            logger.error(f"Error executing scheduled task {task.name}: {e}")
    
    def get_scheduled_tasks(self) -> list:
        """Obtener lista de tareas programadas"""
        return [
            {
                "id": task.id,
                "name": task.name,
                "command": task.command[:100],
                "schedule_type": task.schedule_type.value,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "next_run": task.next_run.isoformat() if task.next_run else None,
                "run_count": task.run_count,
                "max_runs": task.max_runs
            }
            for task in self.scheduled_tasks.values()
        ]
    
    def enable_task(self, task_id: str) -> bool:
        """Habilitar tarea programada"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = True
            self.scheduled_tasks[task_id].next_run = self._calculate_next_run(
                self.scheduled_tasks[task_id]
            )
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Deshabilitar tarea programada"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = False
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Eliminar tarea programada"""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            return True
        return False


