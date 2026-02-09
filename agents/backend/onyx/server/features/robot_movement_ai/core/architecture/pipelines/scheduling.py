"""
Pipeline Scheduling
==================

Sistema de scheduling y ejecución programada para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum

from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Tipo de schedule."""
    ONCE = "once"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    CRON = "cron"


@dataclass
class Schedule:
    """
    Schedule de pipeline.
    """
    schedule_type: ScheduleType
    pipeline: Pipeline
    start_time: Optional[datetime] = None
    interval: Optional[float] = None
    cron_expression: Optional[str] = None
    enabled: bool = True
    max_executions: Optional[int] = None
    execution_count: int = 0
    last_execution_time: Optional[datetime] = None


def _should_execute_once(
    schedule: Schedule,
    current_time: datetime
) -> bool:
    """
    Verificar si debe ejecutarse schedule ONCE (función pura).
    
    Args:
        schedule: Schedule a verificar
        current_time: Tiempo actual
        
    Returns:
        True si debe ejecutarse
    """
    if schedule.schedule_type != ScheduleType.ONCE:
        return False
    
    if not schedule.start_time:
        return False
    
    return current_time >= schedule.start_time


def _should_execute_interval(
    schedule: Schedule,
    current_time: datetime
) -> bool:
    """
    Verificar si debe ejecutarse schedule INTERVAL (función pura).
    
    Args:
        schedule: Schedule a verificar
        current_time: Tiempo actual
        
    Returns:
        True si debe ejecutarse
    """
    if schedule.schedule_type != ScheduleType.INTERVAL:
        return False
    
    if not schedule.interval or schedule.interval <= 0:
        return False
    
    if schedule.start_time:
        if current_time < schedule.start_time:
            return False
        
        if schedule.last_execution_time:
            elapsed = (current_time - schedule.last_execution_time).total_seconds()
            return elapsed >= schedule.interval
    
    return schedule.execution_count == 0


def _should_execute_daily(
    schedule: Schedule,
    current_time: datetime
) -> bool:
    """
    Verificar si debe ejecutarse schedule DAILY (función pura).
    
    Args:
        schedule: Schedule a verificar
        current_time: Tiempo actual
        
    Returns:
        True si debe ejecutarse
    """
    if schedule.schedule_type != ScheduleType.DAILY:
        return False
    
    if not schedule.start_time:
        return False
    
    if current_time < schedule.start_time:
        return False
    
    if schedule.last_execution_time:
        if (current_time - schedule.last_execution_time).days < 1:
            return False
    
    return (
        current_time.hour == schedule.start_time.hour and
        current_time.minute == schedule.start_time.minute
    )


def _should_execute_schedule(
    schedule: Schedule,
    current_time: datetime
) -> bool:
    """
    Verificar si debe ejecutarse un schedule (función pura).
    
    Args:
        schedule: Schedule a verificar
        current_time: Tiempo actual
        
    Returns:
        True si debe ejecutarse
    """
    if not schedule.enabled:
        return False
    
    if schedule.max_executions and \
       schedule.execution_count >= schedule.max_executions:
        return False
    
    if schedule.schedule_type == ScheduleType.ONCE:
        return _should_execute_once(schedule, current_time)
    
    if schedule.schedule_type == ScheduleType.INTERVAL:
        return _should_execute_interval(schedule, current_time)
    
    if schedule.schedule_type == ScheduleType.DAILY:
        return _should_execute_daily(schedule, current_time)
    
    return False


class PipelineScheduler:
    """
    Planificador de pipelines.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializar planificador."""
        self.schedules: Dict[str, Schedule] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
    
    def schedule(
        self,
        schedule_id: str,
        pipeline: Pipeline,
        schedule_type: ScheduleType,
        **kwargs: Any
    ) -> None:
        """
        Programar pipeline.
        
        Args:
            schedule_id: ID del schedule
            pipeline: Pipeline a ejecutar
            schedule_type: Tipo de schedule
            **kwargs: Argumentos adicionales
            
        Raises:
            ValueError: Si schedule_id está vacío o ya existe
        """
        if not schedule_id:
            raise ValueError("Schedule ID cannot be empty")
        
        if not pipeline:
            raise ValueError("Pipeline cannot be None")
        
        schedule = Schedule(
            schedule_type=schedule_type,
            pipeline=pipeline,
            **kwargs
        )
        
        with self._lock:
            if schedule_id in self.schedules:
                raise ValueError(f"Schedule '{schedule_id}' already exists")
            self.schedules[schedule_id] = schedule
        
        logger.info(
            f"Pipeline scheduled: {schedule_id} ({schedule_type.value})"
        )
    
    def schedule_interval(
        self,
        schedule_id: str,
        pipeline: Pipeline,
        interval: float,
        start_time: Optional[datetime] = None
    ) -> None:
        """
        Programar pipeline con intervalo.
        
        Args:
            schedule_id: ID del schedule
            pipeline: Pipeline
            interval: Intervalo en segundos
            start_time: Hora de inicio (opcional)
        """
        if interval <= 0:
            raise ValueError("Interval must be positive")
        
        self.schedule(
            schedule_id,
            pipeline,
            ScheduleType.INTERVAL,
            interval=interval,
            start_time=start_time
        )
    
    def schedule_daily(
        self,
        schedule_id: str,
        pipeline: Pipeline,
        time_str: str
    ) -> None:
        """
        Programar pipeline diariamente.
        
        Args:
            schedule_id: ID del schedule
            pipeline: Pipeline
            time_str: Hora en formato "HH:MM"
            
        Raises:
            ValueError: Si el formato de tiempo es inválido
        """
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid time range")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid time format '{time_str}': {e}") from e
        
        now = datetime.now(timezone.utc)
        start_time = now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )
        
        if start_time < now:
            start_time += timedelta(days=1)
        
        self.schedule(
            schedule_id,
            pipeline,
            ScheduleType.DAILY,
            start_time=start_time
        )
    
    def start(self) -> None:
        """Iniciar planificador."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Detener planificador.
        
        Args:
            timeout: Timeout para esperar al thread
        """
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=timeout)
        
        logger.info("Scheduler stopped")
    
    def _run(self) -> None:
        """Ejecutar loop del planificador."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                with self._lock:
                    schedules_to_execute = [
                        (schedule_id, schedule)
                        for schedule_id, schedule in self.schedules.items()
                        if _should_execute_schedule(schedule, current_time)
                    ]
                
                for schedule_id, schedule in schedules_to_execute:
                    self._execute_schedule(schedule_id, schedule, current_time)
                
                if self._stop_event.wait(timeout=1.0):
                    break
                    
            except Exception as e:
                logger.error(f"Error in scheduler: {e}", exc_info=True)
                if self._stop_event.wait(timeout=5.0):
                    break
    
    def _execute_schedule(
        self,
        schedule_id: str,
        schedule: Schedule,
        current_time: datetime
    ) -> None:
        """
        Ejecutar schedule.
        
        Args:
            schedule_id: ID del schedule
            schedule: Schedule
            current_time: Tiempo actual
        """
        try:
            logger.info(f"Executing scheduled pipeline: {schedule_id}")
            
            with self._lock:
                schedule.execution_count += 1
                schedule.last_execution_time = current_time
            
            # Nota: En una implementación real, esto requeriría datos de entrada
            # y ejecución async del pipeline
            
        except Exception as e:
            logger.error(
                f"Error executing schedule {schedule_id}: {e}",
                exc_info=True
            )
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """
        Remover schedule.
        
        Args:
            schedule_id: ID del schedule
            
        Returns:
            True si se removió, False si no existía
        """
        if not schedule_id:
            return False
        
        with self._lock:
            if schedule_id in self.schedules:
                del self.schedules[schedule_id]
                logger.info(f"Schedule removed: {schedule_id}")
                return True
            return False
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """
        Habilitar schedule.
        
        Args:
            schedule_id: ID del schedule
            
        Returns:
            True si se habilitó, False si no existe
        """
        with self._lock:
            if schedule_id in self.schedules:
                self.schedules[schedule_id].enabled = True
                return True
            return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """
        Deshabilitar schedule.
        
        Args:
            schedule_id: ID del schedule
            
        Returns:
            True si se deshabilitó, False si no existe
        """
        with self._lock:
            if schedule_id in self.schedules:
                self.schedules[schedule_id].enabled = False
                return True
            return False
    
    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """
        Obtener schedule.
        
        Args:
            schedule_id: ID del schedule
            
        Returns:
            Schedule o None
        """
        with self._lock:
            return self.schedules.get(schedule_id)
    
    def list_schedules(self) -> list[str]:
        """
        Listar IDs de schedules.
        
        Returns:
            Lista de IDs
        """
        with self._lock:
            return list(self.schedules.keys())
