"""
Event Scheduler - Programador de Eventos Avanzado
==================================================

Sistema avanzado de programación de eventos con soporte para cron, intervalos y eventos recurrentes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Tipo de programación."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    RECURRING = "recurring"


class ScheduleStatus(Enum):
    """Estado de programación."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ScheduledEvent:
    """Evento programado."""
    event_id: str
    name: str
    schedule_type: ScheduleType
    handler: Callable
    schedule_config: Dict[str, Any]
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventScheduler:
    """Programador de eventos avanzado."""
    
    def __init__(self):
        self.events: Dict[str, ScheduledEvent] = {}
        self.event_tasks: Dict[str, asyncio.Task] = {}
        self.run_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._scheduler_running = False
    
    def start_scheduler(self):
        """Iniciar scheduler."""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        asyncio.create_task(self._scheduler_loop())
        logger.info("Event scheduler started")
    
    def stop_scheduler(self):
        """Detener scheduler."""
        self._scheduler_running = False
        
        # Cancelar todos los tasks
        for task in self.event_tasks.values():
            task.cancel()
        
        self.event_tasks.clear()
        logger.info("Event scheduler stopped")
    
    async def _scheduler_loop(self):
        """Loop principal del scheduler."""
        while self._scheduler_running:
            try:
                await self._check_and_run_events()
                await asyncio.sleep(1)  # Verificar cada segundo
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def _check_and_run_events(self):
        """Verificar y ejecutar eventos."""
        now = datetime.now()
        
        async with self._lock:
            events_to_run = [
                event for event in self.events.values()
                if event.status == ScheduleStatus.ACTIVE
                and event.next_run
                and now >= event.next_run
            ]
        
        for event in events_to_run:
            asyncio.create_task(self._run_event(event))
    
    async def _run_event(self, event: ScheduledEvent):
        """Ejecutar evento."""
        start_time = datetime.now()
        success = False
        error = None
        
        try:
            if asyncio.iscoroutinefunction(event.handler):
                await event.handler()
            else:
                event.handler()
            
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"Error executing event {event.event_id}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        async with self._lock:
            event.last_run = datetime.now()
            event.run_count += 1
            
            # Calcular próximo run
            event.next_run = self._calculate_next_run(event)
            
            # Verificar si se completó
            if event.max_runs and event.run_count >= event.max_runs:
                event.status = ScheduleStatus.COMPLETED
                event.next_run = None
            
            if error:
                event.status = ScheduleStatus.ERROR
            
            # Guardar historial
            self.run_history.append({
                "event_id": event.event_id,
                "timestamp": event.last_run.isoformat(),
                "success": success,
                "error": error,
                "duration": duration,
            })
            
            # Limitar historial
            if len(self.run_history) > 100000:
                self.run_history.pop(0)
    
    def _calculate_next_run(self, event: ScheduledEvent) -> Optional[datetime]:
        """Calcular próximo run."""
        if event.status != ScheduleStatus.ACTIVE:
            return None
        
        if event.schedule_type == ScheduleType.ONCE:
            return None
        
        elif event.schedule_type == ScheduleType.INTERVAL:
            interval_seconds = event.schedule_config.get("interval_seconds", 60)
            return datetime.now() + timedelta(seconds=interval_seconds)
        
        elif event.schedule_type == ScheduleType.CRON:
            cron_expr = event.schedule_config.get("cron_expression", "")
            return self._parse_cron_next_run(cron_expr)
        
        elif event.schedule_type == ScheduleType.RECURRING:
            interval = event.schedule_config.get("interval", "daily")
            return self._calculate_recurring_next_run(interval)
        
        return None
    
    def _parse_cron_next_run(self, cron_expr: str) -> Optional[datetime]:
        """Parsear expresión cron y calcular próximo run."""
        # Implementación simplificada de cron
        # En producción usaría una librería como croniter
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return None
            
            # Parsear minutos, horas, día, mes, día de semana
            # Por ahora, implementación básica
            now = datetime.now()
            return now + timedelta(minutes=1)  # Simplificado
        except Exception:
            return None
    
    def _calculate_recurring_next_run(self, interval: str) -> datetime:
        """Calcular próximo run para evento recurrente."""
        now = datetime.now()
        
        if interval == "hourly":
            return now + timedelta(hours=1)
        elif interval == "daily":
            return now + timedelta(days=1)
        elif interval == "weekly":
            return now + timedelta(weeks=1)
        elif interval == "monthly":
            # Simplificado: agregar 30 días
            return now + timedelta(days=30)
        else:
            return now + timedelta(hours=1)
    
    def schedule_event(
        self,
        event_id: str,
        name: str,
        schedule_type: ScheduleType,
        handler: Callable,
        schedule_config: Dict[str, Any],
        max_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Programar evento."""
        event = ScheduledEvent(
            event_id=event_id,
            name=name,
            schedule_type=schedule_type,
            handler=handler,
            schedule_config=schedule_config,
            max_runs=max_runs,
            metadata=metadata or {},
        )
        
        # Calcular próximo run
        event.next_run = self._calculate_next_run(event)
        
        async def save_event():
            async with self._lock:
                self.events[event_id] = event
        
        asyncio.create_task(save_event())
        
        logger.info(f"Scheduled event: {event_id} - {name}")
        return event_id
    
    async def pause_event(self, event_id: str) -> bool:
        """Pausar evento."""
        async with self._lock:
            event = self.events.get(event_id)
            if not event:
                return False
            
            event.status = ScheduleStatus.PAUSED
            event.next_run = None
        
        logger.info(f"Paused event: {event_id}")
        return True
    
    async def resume_event(self, event_id: str) -> bool:
        """Reanudar evento."""
        async with self._lock:
            event = self.events.get(event_id)
            if not event:
                return False
            
            event.status = ScheduleStatus.ACTIVE
            event.next_run = self._calculate_next_run(event)
        
        logger.info(f"Resumed event: {event_id}")
        return True
    
    async def cancel_event(self, event_id: str) -> bool:
        """Cancelar evento."""
        async with self._lock:
            event = self.events.get(event_id)
            if not event:
                return False
            
            event.status = ScheduleStatus.CANCELLED
            event.next_run = None
        
        logger.info(f"Cancelled event: {event_id}")
        return True
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de evento."""
        event = self.events.get(event_id)
        if not event:
            return None
        
        return {
            "event_id": event.event_id,
            "name": event.name,
            "schedule_type": event.schedule_type.value,
            "status": event.status.value,
            "next_run": event.next_run.isoformat() if event.next_run else None,
            "last_run": event.last_run.isoformat() if event.last_run else None,
            "run_count": event.run_count,
            "max_runs": event.max_runs,
            "created_at": event.created_at.isoformat(),
        }
    
    def get_event_run_history(self, event_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de ejecuciones."""
        history = self.run_history
        
        if event_id:
            history = [h for h in history if h["event_id"] == event_id]
        
        history.sort(key=lambda h: h["timestamp"], reverse=True)
        return history[:limit]
    
    def get_event_scheduler_summary(self) -> Dict[str, Any]:
        """Obtener resumen del scheduler."""
        by_status: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        
        for event in self.events.values():
            by_status[event.status.value] += 1
            by_type[event.schedule_type.value] += 1
        
        return {
            "total_events": len(self.events),
            "events_by_status": dict(by_status),
            "events_by_type": dict(by_type),
            "active_events": len([e for e in self.events.values() if e.status == ScheduleStatus.ACTIVE]),
            "total_runs": len(self.run_history),
            "scheduler_running": self._scheduler_running,
        }



