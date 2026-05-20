"""
Event Bus - Bus de Eventos
===========================

Sistema de eventos para comunicación desacoplada entre módulos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipo de evento."""
    DOCUMENT_ANALYZED = "document.analyzed"
    DOCUMENT_OPTIMIZED = "document.optimized"
    PLAGIARISM_DETECTED = "plagiarism.detected"
    MODULE_LOADED = "module.loaded"
    MODULE_ERROR = "module.error"
    PLUGIN_EXECUTED = "plugin.executed"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento."""
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Bus de eventos."""
    
    def __init__(self):
        """Inicializar bus."""
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history: int = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Suscribirse a evento."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
            logger.info(f"Handler suscrito a {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribirse de evento."""
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Handler desuscrito de {event_type.value}")
    
    async def publish(self, event: Event):
        """Publicar evento."""
        # Agregar a historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notificar suscriptores
        handlers = self.subscribers.get(event.event_type, [])
        
        if handlers:
            logger.debug(f"Publicando evento {event.event_type.value} a {len(handlers)} handlers")
            
            # Ejecutar handlers en paralelo
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error ejecutando handler para {event.event_type.value}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def publish_sync(self, event_type: EventType, payload: Dict[str, Any], source: Optional[str] = None):
        """Publicar evento de forma síncrona."""
        event = Event(
            event_type=event_type,
            payload=payload,
            source=source
        )
        await self.publish(event)
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Obtener historial de eventos."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def clear_history(self):
        """Limpiar historial."""
        self.event_history.clear()


__all__ = [
    "EventBus",
    "Event",
    "EventType"
]


