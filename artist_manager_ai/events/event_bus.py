"""
Event Bus
=========

Sistema de eventos pub/sub.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos."""
    EVENT_CREATED = "event.created"
    EVENT_UPDATED = "event.updated"
    EVENT_DELETED = "event.deleted"
    ROUTINE_COMPLETED = "routine.completed"
    PROTOCOL_VIOLATION = "protocol.violation"
    NOTIFICATION_SENT = "notification.sent"


@dataclass
class Event:
    """Evento."""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


EventHandler = Callable[[Event], None]


class EventBus:
    """Event bus para pub/sub."""
    
    def __init__(self):
        """Inicializar event bus."""
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self._logger = logger
    
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """
        Suscribirse a un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        self._logger.info(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """
        Desuscribirse de un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        if event_type in self.handlers:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """
        Publicar evento.
        
        Args:
            event: Evento a publicar
        """
        handlers = self.handlers.get(event.type, [])
        
        for handler in handlers:
            try:
                if hasattr(handler, '__code__') and handler.__code__.co_flags & 0x80:
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Error in event handler for {event.type.value}: {str(e)}")
        
        self._logger.debug(f"Published event {event.type.value} to {len(handlers)} handlers")
    
    def get_subscribers_count(self, event_type: EventType) -> int:
        """
        Obtener número de suscriptores.
        
        Args:
            event_type: Tipo de evento
        
        Returns:
            Número de suscriptores
        """
        return len(self.handlers.get(event_type, []))




