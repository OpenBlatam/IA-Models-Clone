"""
Event Bus - Sistema de eventos
================================

Sistema de eventos pub/sub para comunicación entre componentes.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .error_handling import safe_async_call

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    TASK_ADDED = "task_added"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    COMMAND_RECEIVED = "command_received"
    SCHEDULED_TASK_EXECUTED = "scheduled_task_executed"
    BACKUP_CREATED = "backup_created"
    PLUGIN_LOADED = "plugin_loaded"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Evento"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class EventBus:
    """Bus de eventos"""
    
    def __init__(self, max_history: int = 10000):
        from .validation_utils import validate_positive
        validate_positive(max_history, "max_history")
        
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = max_history
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Suscribirse a un tipo de evento"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.debug(f"📡 Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Desuscribirse de un tipo de evento"""
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                logger.debug(f"📡 Unsubscribed from {event_type.value}")
    
    async def publish(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
        """Publicar evento"""
        event = Event(
            type=event_type,
            data=data,
            source=source
        )
        
        # Agregar al historial
        self.event_history.append(event)
        
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notificar suscriptores
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                async def execute_callback():
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                
                await safe_async_call(
                    execute_callback,
                    operation=f"executing callback for event {event_type.value}",
                    logger_instance=logger,
                    reraise=False
                )
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Obtener eventos del historial"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        # Ordenar por timestamp (más recientes primero)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del event bus"""
        by_type = {}
        for event in self.event_history:
            event_type = event.type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.event_history),
            "subscribers": sum(len(callbacks) for callbacks in self.subscribers.values()),
            "events_by_type": by_type
        }



