"""
Events - Sistema de eventos y event bus
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    CONTENT_ADDED = "content_added"
    CONTENT_REMOVED = "content_removed"
    CONTENT_MODIFIED = "content_modified"
    VERSION_CREATED = "version_created"
    BACKUP_CREATED = "backup_created"
    OPERATION_COMPLETE = "operation_complete"
    ERROR_OCCURRED = "error_occurred"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento del sistema"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    id: Optional[str] = None


class EventBus:
    """Bus de eventos"""

    def __init__(self):
        """Inicializar el bus de eventos"""
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000

    def subscribe(self, event_type: EventType, callback: Callable):
        """
        Suscribirse a un tipo de evento.

        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.info(f"Suscripción a evento: {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """
        Desuscribirse de un evento.

        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        """
        Publicar un evento.

        Args:
            event: Evento a publicar
        """
        import uuid
        if not event.id:
            event.id = str(uuid.uuid4())
        
        # Agregar al historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notificar a suscriptores
        subscribers = self.subscribers.get(event.type, [])
        subscribers_all = self.subscribers.get(EventType.CUSTOM, [])
        
        all_subscribers = subscribers + subscribers_all
        
        for callback in all_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error en callback de evento: {e}")

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de eventos.

        Args:
            event_type: Filtrar por tipo
            limit: Límite de resultados

        Returns:
            Lista de eventos
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        events = events[-limit:][::-1]
        
        return [
            {
                "id": e.id,
                "type": e.type.value,
                "data": e.data,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source
            }
            for e in events
        ]

    def clear_history(self):
        """Limpiar historial de eventos"""
        self.event_history.clear()






