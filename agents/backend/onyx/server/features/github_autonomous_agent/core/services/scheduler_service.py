"""
Servicio de Scheduling de Tareas.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from config.logging_config import get_logger
from config.di_setup import get_service

logger = get_logger(__name__)


class ScheduleType(str, Enum):
    """Tipos de schedule."""
    ONCE = "once"  # Ejecutar una vez
    INTERVAL = "interval"  # Ejecutar cada X tiempo
    CRON = "cron"  # Ejecutar según expresión cron
    DAILY = "daily"  # Ejecutar diariamente
    WEEKLY = "weekly"  # Ejecutar semanalmente


@dataclass
class ScheduledTask:
    """Tarea programada."""
    task_id: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    task_data: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicializar después de creación."""
        if self.metadata is None:
            self.metadata = {}
        self._calculate_next_run()
    
    def _calculate_next_run(self) -> None:
        """Calcular próxima ejecución."""
        if not self.enabled:
            self.next_run = None
            return
        
        now = datetime.now()
        
        if self.schedule_type == ScheduleType.ONCE:
            if self.run_count > 0:
                self.next_run = None
            else:
                self.next_run = now + timedelta(seconds=self.schedule_config.get("delay_seconds", 0))
        
        elif self.schedule_type == ScheduleType.INTERVAL:
            interval_seconds = self.schedule_config.get("interval_seconds", 3600)
            if self.last_run:
                self.next_run = self.last_run + timedelta(seconds=interval_seconds)
            else:
                self.next_run = now + timedelta(seconds=interval_seconds)
        
        elif self.schedule_type == ScheduleType.DAILY:
            hour = self.schedule_config.get("hour", 0)
            minute = self.schedule_config.get("minute", 0)
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            self.next_run = next_run
        
        elif self.schedule_type == ScheduleType.WEEKLY:
            day_of_week = self.schedule_config.get("day_of_week", 0)  # 0 = Monday
            hour = self.schedule_config.get("hour", 0)
            minute = self.schedule_config.get("minute", 0)
            
            days_until = (day_of_week - now.weekday()) % 7
            if days_until == 0:
                # Si es hoy, verificar hora
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    days_until = 7
                else:
                    self.next_run = next_run
                    return
            
            next_run = now + timedelta(days=days_until)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            self.next_run = next_run


class SchedulerService:
    """
    Servicio para programar tareas con mejoras.
    
    Attributes:
        scheduled_tasks: Diccionario de tareas programadas
        running: Si el scheduler está corriendo
        scheduler_task: Tarea asyncio del scheduler
        task_handler: Handler para ejecutar tareas
    """
    
    def __init__(self):
        """Inicializar servicio de scheduling con logging."""
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.task_handler: Optional[Callable] = None
        
        logger.info("✅ SchedulerService inicializado")
    
    def register_task_handler(self, handler: Callable) -> None:
        """
        Registrar handler para ejecutar tareas.
        
        Args:
            handler: Función async que recibe task_data y ejecuta la tarea
        """
        self.task_handler = handler
        logger.info("Task handler registrado")
    
    def schedule_task(
        self,
        task_id: str,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any],
        task_data: Dict[str, Any],
        enabled: bool = True,
        max_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScheduledTask:
        """
        Programar tarea con validaciones.
        
        Args:
            task_id: ID único de la tarea programada (debe ser string no vacío)
            schedule_type: Tipo de schedule (debe ser ScheduleType)
            schedule_config: Configuración del schedule (debe ser diccionario)
            task_data: Datos de la tarea a ejecutar (debe ser diccionario)
            enabled: Si está habilitada (debe ser bool)
            max_runs: Número máximo de ejecuciones (opcional, debe ser entero positivo)
            metadata: Metadata adicional (opcional, debe ser diccionario si se proporciona)
            
        Returns:
            Tarea programada
            
        Raises:
            ValueError: Si los parámetros son inválidos o la tarea ya existe
        """
        # Validaciones
        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task_id debe ser un string no vacío, recibido: {task_id}")
        
        if not isinstance(schedule_type, ScheduleType):
            raise ValueError(f"schedule_type debe ser un ScheduleType, recibido: {type(schedule_type)}")
        
        if not isinstance(schedule_config, dict):
            raise ValueError(f"schedule_config debe ser un diccionario, recibido: {type(schedule_config)}")
        
        if not isinstance(task_data, dict):
            raise ValueError(f"task_data debe ser un diccionario, recibido: {type(task_data)}")
        
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled debe ser un bool, recibido: {type(enabled)}")
        
        if max_runs is not None:
            if not isinstance(max_runs, int) or max_runs < 1:
                raise ValueError(f"max_runs debe ser un entero positivo si se proporciona, recibido: {max_runs}")
        
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata debe ser un diccionario si se proporciona, recibido: {type(metadata)}")
        
        task_id = task_id.strip()
        
        # Verificar si la tarea ya existe
        if task_id in self.scheduled_tasks:
            logger.warning(f"Tarea {task_id} ya existe, será reemplazada")
        
        try:
            scheduled_task = ScheduledTask(
                task_id=task_id,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                task_data=task_data,
                enabled=enabled,
                max_runs=max_runs,
                metadata=metadata
            )
            
            self.scheduled_tasks[task_id] = scheduled_task
            
            logger.info(
                f"✅ Tarea programada: {task_id} - tipo: {schedule_type.value}, "
                f"enabled: {enabled}, next_run: {scheduled_task.next_run}, "
                f"max_runs: {max_runs or 'unlimited'}"
            )
            
            return scheduled_task
        except Exception as e:
            logger.error(f"Error al programar tarea {task_id}: {e}", exc_info=True)
            raise ValueError(f"Error al programar tarea: {e}") from e
    
    async def execute_task(self, scheduled_task: ScheduledTask) -> bool:
        """
        Ejecutar tarea programada.
        
        Args:
            scheduled_task: Tarea a ejecutar
            
        Returns:
            True si se ejecutó exitosamente
        """
        if not self.task_handler:
            logger.error("No hay task handler registrado")
            return False
        
        if not scheduled_task.enabled:
            return False
        
        # Verificar max_runs
        if scheduled_task.max_runs and scheduled_task.run_count >= scheduled_task.max_runs:
            logger.info(f"Tarea {scheduled_task.task_id} alcanzó max_runs, deshabilitando")
            scheduled_task.enabled = False
            return False
        
        try:
            logger.info(f"Ejecutando tarea programada: {scheduled_task.task_id}")
            
            if asyncio.iscoroutinefunction(self.task_handler):
                await self.task_handler(scheduled_task.task_data)
            else:
                self.task_handler(scheduled_task.task_data)
            
            scheduled_task.last_run = datetime.now()
            scheduled_task.run_count += 1
            scheduled_task._calculate_next_run()
            
            logger.info(f"Tarea {scheduled_task.task_id} ejecutada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error ejecutando tarea {scheduled_task.task_id}: {e}", exc_info=True)
            scheduled_task._calculate_next_run()
            return False
    
    async def start_scheduler(self, check_interval: float = 60.0) -> None:
        """
        Iniciar scheduler con validaciones.
        
        Args:
            check_interval: Intervalo entre verificaciones en segundos (debe ser float positivo)
            
        Raises:
            ValueError: Si check_interval es inválido
            RuntimeError: Si el scheduler ya está corriendo
        """
        # Validación
        if not isinstance(check_interval, (int, float)) or check_interval <= 0:
            raise ValueError(f"check_interval debe ser un número positivo, recibido: {check_interval}")
        
        if self.running:
            logger.warning("Scheduler ya está corriendo")
            return
        
        self.running = True
        
        async def scheduler_loop():
            logger.info(f"🔄 Scheduler loop iniciado (check_interval: {check_interval}s)")
            while self.running:
                try:
                    now = datetime.now()
                    tasks_to_run = 0
                    
                    # Verificar tareas que deben ejecutarse
                    for scheduled_task in self.scheduled_tasks.values():
                        if not scheduled_task.enabled:
                            continue
                        
                        if scheduled_task.next_run and scheduled_task.next_run <= now:
                            tasks_to_run += 1
                            await self.execute_task(scheduled_task)
                    
                    if tasks_to_run > 0:
                        logger.debug(f"Ejecutadas {tasks_to_run} tareas programadas")
                    
                    await asyncio.sleep(check_interval)
                except asyncio.CancelledError:
                    logger.info("Scheduler loop cancelado")
                    raise
                except Exception as e:
                    logger.error(f"Error en scheduler loop: {e}", exc_info=True)
                    await asyncio.sleep(check_interval)
        
        self.scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info(f"✅ Scheduler iniciado (check_interval: {check_interval}s, tasks: {len(self.scheduled_tasks)})")
    
    async def stop_scheduler(self) -> None:
        """Detener scheduler."""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler detenido")
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Obtener tarea programada."""
        return self.scheduled_tasks.get(task_id)
    
    def list_tasks(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Listar tareas programadas."""
        tasks = [
            {
                "task_id": t.task_id,
                "schedule_type": t.schedule_type.value,
                "schedule_config": t.schedule_config,
                "enabled": t.enabled,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "next_run": t.next_run.isoformat() if t.next_run else None,
                "run_count": t.run_count,
                "max_runs": t.max_runs,
                "metadata": t.metadata
            }
            for t in self.scheduled_tasks.values()
            if not enabled_only or t.enabled
        ]
        return tasks
    
    def enable_task(self, task_id: str) -> bool:
        """Habilitar tarea."""
        task = self.scheduled_tasks.get(task_id)
        if task:
            task.enabled = True
            task._calculate_next_run()
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Deshabilitar tarea."""
        task = self.scheduled_tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Eliminar tarea programada."""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            logger.info(f"Tarea programada eliminada: {task_id}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_tasks": len(self.scheduled_tasks),
            "enabled_tasks": len([t for t in self.scheduled_tasks.values() if t.enabled]),
            "running": self.running,
            "tasks_by_type": {
                st.value: len([t for t in self.scheduled_tasks.values() if t.schedule_type == st])
                for st in ScheduleType
            }
        }

