"""
Scheduled Tasks - Sistema de tareas programadas (Cron-like)
===========================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Tarea programada"""
    id: str
    name: str
    task: Callable
    schedule: str  # Cron expression o intervalo
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "metadata": self.metadata
        }


class CronParser:
    """Parser de expresiones cron"""
    
    @staticmethod
    def parse(cron_expr: str) -> Dict[str, List[int]]:
        """Parsea una expresión cron"""
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("Cron expression debe tener 5 partes")
        
        return {
            "minute": CronParser._parse_part(parts[0], 0, 59),
            "hour": CronParser._parse_part(parts[1], 0, 23),
            "day": CronParser._parse_part(parts[2], 1, 31),
            "month": CronParser._parse_part(parts[3], 1, 12),
            "weekday": CronParser._parse_part(parts[4], 0, 6)
        }
    
    @staticmethod
    def _parse_part(part: str, min_val: int, max_val: int) -> List[int]:
        """Parsea una parte de cron"""
        if part == "*":
            return list(range(min_val, max_val + 1))
        
        if "/" in part:
            base, step = part.split("/")
            if base == "*":
                base_range = list(range(min_val, max_val + 1))
            else:
                base_range = CronParser._parse_part(base, min_val, max_val)
            return [v for v in base_range if (v - min_val) % int(step) == 0]
        
        if "-" in part:
            start, end = part.split("-")
            return list(range(int(start), int(end) + 1))
        
        if "," in part:
            return [int(v) for v in part.split(",")]
        
        return [int(part)]
    
    @staticmethod
    def get_next_run(cron_expr: str, from_time: Optional[datetime] = None) -> datetime:
        """Calcula el próximo tiempo de ejecución"""
        if from_time is None:
            from_time = datetime.now()
        
        cron_parts = CronParser.parse(cron_expr)
        current = from_time
        
        # Intentar encontrar el próximo tiempo válido (simplificado)
        # En producción, usar una librería como croniter
        for _ in range(10000):  # Límite de iteraciones
            current += timedelta(minutes=1)
            
            if current.minute in cron_parts["minute"] and \
               current.hour in cron_parts["hour"] and \
               current.day in cron_parts["day"] and \
               current.month in cron_parts["month"] and \
               current.weekday() in cron_parts["weekday"]:
                return current
        
        raise ValueError("No se pudo calcular próximo tiempo de ejecución")


class ScheduledTaskManager:
    """Gestor de tareas programadas"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._task_loop: Optional[asyncio.Task] = None
    
    def register_task(
        self,
        task_id: str,
        name: str,
        task: Callable,
        schedule: str,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScheduledTask:
        """Registra una tarea programada"""
        # Calcular próximo tiempo de ejecución
        next_run = None
        if schedule.startswith("every "):
            # Intervalo simple: "every 5 minutes"
            interval_str = schedule.replace("every ", "").strip()
            next_run = self._parse_interval(interval_str)
        else:
            # Expresión cron
            try:
                next_run = CronParser.get_next_run(schedule)
            except Exception as e:
                logger.error(f"Error parseando schedule {schedule}: {e}")
                next_run = datetime.now() + timedelta(hours=1)
        
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            task=task,
            schedule=schedule,
            enabled=enabled,
            next_run=next_run,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = scheduled_task
        logger.info(f"Tarea {task_id} registrada con schedule {schedule}")
        
        # Iniciar loop si no está corriendo
        if not self.running:
            asyncio.create_task(self._run_loop())
        
        return scheduled_task
    
    def _parse_interval(self, interval_str: str) -> datetime:
        """Parsea un intervalo simple"""
        now = datetime.now()
        
        if "minute" in interval_str or "minutes" in interval_str:
            minutes = int(re.search(r'\d+', interval_str).group())
            return now + timedelta(minutes=minutes)
        elif "hour" in interval_str or "hours" in interval_str:
            hours = int(re.search(r'\d+', interval_str).group())
            return now + timedelta(hours=hours)
        elif "day" in interval_str or "days" in interval_str:
            days = int(re.search(r'\d+', interval_str).group())
            return now + timedelta(days=days)
        else:
            return now + timedelta(hours=1)
    
    async def _run_loop(self):
        """Loop principal de ejecución de tareas"""
        if self.running:
            return
        
        self.running = True
        logger.info("Iniciando loop de tareas programadas")
        
        while self.running:
            try:
                now = datetime.now()
                
                for task in self.tasks.values():
                    if not task.enabled:
                        continue
                    
                    if task.next_run and task.next_run <= now:
                        # Ejecutar tarea
                        asyncio.create_task(self._execute_task(task))
                        
                        # Calcular próximo tiempo
                        if task.schedule.startswith("every "):
                            task.next_run = self._parse_interval(task.schedule)
                        else:
                            try:
                                task.next_run = CronParser.get_next_run(task.schedule, now)
                            except Exception as e:
                                logger.error(f"Error calculando próximo run para {task.id}: {e}")
                                task.next_run = now + timedelta(hours=1)
                
                # Esperar 1 minuto antes de verificar de nuevo
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error en loop de tareas: {e}")
                await asyncio.sleep(60)
    
    async def _execute_task(self, task: ScheduledTask):
        """Ejecuta una tarea"""
        try:
            logger.info(f"Ejecutando tarea {task.id}: {task.name}")
            
            if asyncio.iscoroutinefunction(task.task):
                await task.task()
            else:
                task.task()
            
            task.last_run = datetime.now()
            task.run_count += 1
            logger.info(f"Tarea {task.id} completada exitosamente")
        except Exception as e:
            task.error_count += 1
            logger.error(f"Error ejecutando tarea {task.id}: {e}")
    
    def enable_task(self, task_id: str) -> bool:
        """Habilita una tarea"""
        if task_id not in self.tasks:
            return False
        
        self.tasks[task_id].enabled = True
        return True
    
    def disable_task(self, task_id: str) -> bool:
        """Deshabilita una tarea"""
        if task_id not in self.tasks:
            return False
        
        self.tasks[task_id].enabled = False
        return True
    
    def run_task_now(self, task_id: str) -> bool:
        """Ejecuta una tarea inmediatamente"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        asyncio.create_task(self._execute_task(task))
        return True
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """Lista todas las tareas"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene una tarea"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].to_dict()
    
    def stop(self):
        """Detiene el gestor de tareas"""
        self.running = False
        logger.info("Gestor de tareas detenido")




