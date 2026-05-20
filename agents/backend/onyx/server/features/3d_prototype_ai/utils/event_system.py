"""
Event System - Sistema de eventos pub/sub
==========================================
"""

import logging
import asyncio
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Tipos de eventos"""
    PROTOTYPE_CREATED = "prototype.created"
    PROTOTYPE_UPDATED = "prototype.updated"
    PROTOTYPE_DELETED = "prototype.deleted"
    MATERIAL_SEARCHED = "material.searched"
    VALIDATION_COMPLETED = "validation.completed"
    COST_ANALYZED = "cost.analyzed"
    EXPORT_COMPLETED = "export.completed"
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    SHARE_CREATED = "share.created"
    COMMENT_ADDED = "comment.added"


@dataclass
class Event:
    """Evento del sistema"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None


class EventSystem:
    """Sistema de eventos pub/sub"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Suscribe un handler a un tipo de evento"""
        self.subscribers[event_type].append(handler)
        logger.info(f"Handler suscrito a evento: {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribe un handler"""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Handler desuscrito de evento: {event_type.value}")
    
    async def publish(self, event_type: EventType, data: Dict[str, Any],
                     source: Optional[str] = None):
        """Publica un evento"""
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        # Agregar al historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notificar a suscriptores
        handlers = self.subscribers.get(event_type, [])
        
        if handlers:
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error en handler de evento {event_type.value}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Evento publicado: {event_type.value}")
    
    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de eventos"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        events = events[-limit:]
        
        return [
            {
                "type": e.event_type.value,
                "data": e.data,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source
            }
            for e in events
        ]
    
    def get_subscribers(self, event_type: EventType) -> int:
        """Obtiene número de suscriptores para un evento"""
        return len(self.subscribers.get(event_type, []))




