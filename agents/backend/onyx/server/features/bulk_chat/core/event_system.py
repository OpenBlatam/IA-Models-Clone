"""
Event System - Sistema de Eventos en Tiempo Real
=================================================

Sistema de eventos pub/sub para comunicación en tiempo real.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos."""
    SESSION_CREATED = "session_created"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_STOPPED = "session_stopped"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR_OCCURRED = "error_occurred"
    METRIC_UPDATED = "metric_updated"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


@dataclass
class Event:
    """Evento del sistema."""
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class EventBus:
    """Bus de eventos para pub/sub."""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history: int = 1000
        self._lock = asyncio.Lock()
    
    async def subscribe(
        self,
        event_type: EventType,
        callback: Callable,
    ):
        """
        Suscribirse a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            callback: Función callback (async)
        """
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type.value}")
    
    async def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable,
    ):
        """Desuscribirse de un tipo de evento."""
        async with self._lock:
            if event_type in self.subscribers:
                if callback in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type.value}")
    
    async def publish(self, event: Event):
        """
        Publicar evento.
        
        Args:
            event: Evento a publicar
        """
        async with self._lock:
            # Guardar en historial
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        
        # Notificar suscriptores
        subscribers = self.subscribers.get(event.event_type, [])
        
        if subscribers:
            # Ejecutar callbacks en paralelo
            tasks = [
                self._execute_callback(callback, event)
                for callback in subscribers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published event: {event.event_type.value}")
    
    async def _execute_callback(
        self,
        callback: Callable,
        event: Event,
    ):
        """Ejecutar callback de suscriptor."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error executing callback for {event.event_type.value}: {e}")
    
    async def publish_event(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
    ):
        """Publicar evento de forma conveniente."""
        event = Event(
            event_type=event_type,
            source=source,
            data=data,
        )
        await self.publish(event)
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Obtener historial de eventos."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_subscribers_count(self) -> Dict[str, int]:
        """Obtener conteo de suscriptores por tipo."""
        return {
            event_type.value: len(callbacks)
            for event_type, callbacks in self.subscribers.items()
        }
































