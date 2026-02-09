"""
Event Bus System
================

Sistema de bus de eventos para comunicación desacoplada.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Evento."""
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    source: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Bus de eventos.
    
    Gestiona publicación y suscripción de eventos.
    """
    
    def __init__(self):
        """Inicializar bus de eventos."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 10000
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """
        Suscribirse a tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
        """
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable
    ) -> bool:
        """
        Desuscribirse de tipo de evento.
        
        Args:
            event_type: Tipo de evento
            handler: Función manejadora
            
        Returns:
            True si se desuscribió, False si no estaba suscrito
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                return True
        return False
    
    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Publicar evento.
        
        Args:
            event_type: Tipo de evento
            payload: Datos del evento
            source: Fuente del evento
            metadata: Metadata adicional
            
        Returns:
            Evento publicado
        """
        event_id = f"event_{len(self.event_history)}"
        event = Event(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            source=source,
            metadata=metadata or {}
        )
        
        # Guardar en historial
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notificar a suscriptores
        handlers = self.subscribers.get(event_type, [])
        handlers_all = self.subscribers.get("*", [])  # Wildcard
        
        for handler in handlers + handlers_all:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}", exc_info=True)
        
        logger.debug(f"Published event: {event_type} ({event_id})")
        
        return event
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
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
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        """Obtener suscriptores de tipo de evento."""
        return self.subscribers.get(event_type, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del bus de eventos."""
        return {
            "total_events": len(self.event_history),
            "subscribers_by_type": {
                event_type: len(handlers)
                for event_type, handlers in self.subscribers.items()
            },
            "event_types": list(set(e.event_type for e in self.event_history))
        }


# Instancia global
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Obtener instancia global del bus de eventos."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus






