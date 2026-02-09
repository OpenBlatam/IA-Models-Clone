"""
Timed Events - Sistema de Eventos Temporales
============================================

Sistema para programar eventos que se ejecutan en momentos específicos.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EventStatus(Enum):
    """Estado de evento"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TimedEvent:
    """Evento temporal"""
    id: str
    name: str
    handler: Callable
    scheduled_time: datetime
    status: EventStatus = EventStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_due(self) -> bool:
        """Verificar si el evento está listo para ejecutar"""
        return datetime.now() >= self.scheduled_time
    
    def time_until_execution(self) -> Optional[float]:
        """Obtener tiempo hasta ejecución en segundos"""
        if self.scheduled_time > datetime.now():
            return (self.scheduled_time - datetime.now()).total_seconds()
        return 0.0


class TimedEventManager:
    """
    Gestor de eventos temporales.
    
    Programa eventos para ejecutarse en momentos específicos.
    """
    
    def __init__(self):
        self.events: Dict[str, TimedEvent] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self.check_interval = 1.0  # segundos
    
    def schedule(
        self,
        name: str,
        handler: Callable,
        scheduled_time: datetime,
        event_id: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Programar evento.
        
        Args:
            name: Nombre del evento
            handler: Función handler (puede ser async o sync)
            scheduled_time: Tiempo de ejecución
            event_id: ID opcional (se genera si no se proporciona)
            **metadata: Metadata adicional
            
        Returns:
            ID del evento
        """
        if event_id is None:
            event_id = f"event_{datetime.now().timestamp()}_{len(self.events)}"
        
        event = TimedEvent(
            id=event_id,
            name=name,
            handler=handler,
            scheduled_time=scheduled_time,
            status=EventStatus.SCHEDULED,
            metadata=metadata
        )
        
        self.events[event_id] = event
        logger.info(f"⏰ Event scheduled: {name} at {scheduled_time}")
        return event_id
    
    def schedule_in(
        self,
        name: str,
        handler: Callable,
        delay_seconds: float,
        event_id: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Programar evento en N segundos.
        
        Args:
            name: Nombre del evento
            handler: Función handler
            delay_seconds: Delay en segundos
            event_id: ID opcional
            **metadata: Metadata adicional
            
        Returns:
            ID del evento
        """
        scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)
        return self.schedule(name, handler, scheduled_time, event_id, **metadata)
    
    def cancel(self, event_id: str) -> bool:
        """
        Cancelar evento.
        
        Args:
            event_id: ID del evento
            
        Returns:
            True si se canceló
        """
        if event_id in self.events:
            event = self.events[event_id]
            if event.status in [EventStatus.PENDING, EventStatus.SCHEDULED]:
                event.status = EventStatus.CANCELLED
                logger.info(f"🚫 Event cancelled: {event.name}")
                return True
        return False
    
    async def start(self) -> None:
        """Iniciar scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("⏰ Timed event scheduler started")
    
    async def stop(self) -> None:
        """Detener scheduler"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("⏰ Timed event scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Loop del scheduler"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Buscar eventos listos
                due_events = [
                    event for event in self.events.values()
                    if event.status == EventStatus.SCHEDULED and event.is_due()
                ]
                
                for event in due_events:
                    await self._execute_event(event)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    async def _execute_event(self, event: TimedEvent) -> None:
        """Ejecutar evento"""
        event.status = EventStatus.EXECUTING
        logger.info(f"⚡ Executing event: {event.name}")
        
        try:
            if asyncio.iscoroutinefunction(event.handler):
                result = await event.handler(event.metadata)
            else:
                result = event.handler(event.metadata)
            
            event.result = result
            event.status = EventStatus.COMPLETED
            event.executed_at = datetime.now()
            logger.info(f"✅ Event completed: {event.name}")
            
        except Exception as e:
            event.status = EventStatus.FAILED
            event.error = str(e)
            event.executed_at = datetime.now()
            logger.error(f"❌ Event failed: {event.name} - {e}")
    
    def get_event(self, event_id: str) -> Optional[TimedEvent]:
        """Obtener evento"""
        return self.events.get(event_id)
    
    def get_pending_events(self) -> List[TimedEvent]:
        """Obtener eventos pendientes"""
        return [
            event for event in self.events.values()
            if event.status in [EventStatus.PENDING, EventStatus.SCHEDULED]
        ]
    
    def get_upcoming_events(self, limit: int = 10) -> List[TimedEvent]:
        """
        Obtener próximos eventos.
        
        Args:
            limit: Número máximo de eventos
            
        Returns:
            Lista de eventos ordenados por tiempo
        """
        pending = self.get_pending_events()
        pending.sort(key=lambda x: x.scheduled_time)
        return pending[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            "total_events": len(self.events),
            "pending": sum(1 for e in self.events.values() if e.status == EventStatus.SCHEDULED),
            "completed": sum(1 for e in self.events.values() if e.status == EventStatus.COMPLETED),
            "failed": sum(1 for e in self.events.values() if e.status == EventStatus.FAILED),
            "cancelled": sum(1 for e in self.events.values() if e.status == EventStatus.CANCELLED),
            "running": self._running
        }




