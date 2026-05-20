"""
Event System
============
Sistema de eventos para comunicación desacoplada.
"""

from typing import Dict, List, Callable, Any, Optional
from enum import Enum
import asyncio
from datetime import datetime
from ...utils.logger import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Tipos de eventos."""
    COACHING_REQUEST = "coaching_request"
    COACHING_RESPONSE = "coaching_response"
    TRAINING_PLAN_CREATED = "training_plan_created"
    BEHAVIOR_ANALYZED = "behavior_analyzed"
    ERROR = "error"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class Event:
    """Representación de un evento."""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.id = f"{event_type.value}_{self.timestamp.timestamp()}"


class EventBus:
    """Bus de eventos para comunicación desacoplada."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Suscribirse a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribirse de un tipo de evento."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """
        Publicar un evento.
        
        Args:
            event: Evento a publicar
        """
        # Agregar al historial
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notificar suscriptores
        handlers = self._subscribers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}", event_type=event.event_type.value)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """
        Obtener historial de eventos.
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de eventos
            
        Returns:
            Lista de eventos
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del bus."""
        type_counts = {}
        for event in self._event_history:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
        
        return {
            "total_events": len(self._event_history),
            "subscribers": {
                event_type.value: len(handlers)
                for event_type, handlers in self._subscribers.items()
            },
            "events_by_type": type_counts
        }


# Instancia global del bus de eventos
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Obtener instancia global del bus de eventos."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

