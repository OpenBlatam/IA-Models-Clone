"""
Sistema de Eventos (Observer Pattern)

Permite comunicación desacoplada entre componentes mediante eventos.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos del sistema"""
    # Generación
    MUSIC_GENERATED = "music_generated"
    GENERATION_STARTED = "generation_started"
    GENERATION_FAILED = "generation_failed"
    
    # Audio
    AUDIO_PROCESSED = "audio_processed"
    AUDIO_UPLOADED = "audio_uploaded"
    
    # Usuario
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    
    # Marketplace
    PURCHASE_COMPLETED = "purchase_completed"
    LISTING_CREATED = "listing_created"
    
    # Colaboración
    COLLABORATION_STARTED = "collaboration_started"
    COLLABORATION_ENDED = "collaboration_ended"
    
    # Sistema
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


@dataclass
class Event:
    """Evento del sistema"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class EventBus:
    """Bus de eventos para comunicación entre componentes"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Suscribe un handler a un tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función que maneja el evento
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribe un handler"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Handler unsubscribed from {event_type.value}")
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """
        Publica un evento
        
        Args:
            event: Evento a publicar
        """
        # Guardar en historial
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notificar a suscriptores
        handlers = self._subscribers.get(event.event_type, [])
        
        if handlers:
            # Ejecutar handlers en paralelo
            tasks = []
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    tasks.append(asyncio.to_thread(handler, event))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Event published: {event.event_type.value}")
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Obtiene historial de eventos
        
        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de resultados
        
        Returns:
            Lista de eventos
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def clear_history(self):
        """Limpia el historial de eventos"""
        self._event_history.clear()


# Instancia global
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Obtiene la instancia global del bus de eventos"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def event_handler(event_type: EventType):
    """
    Decorador para registrar handlers de eventos
    
    Args:
        event_type: Tipo de evento
    """
    def decorator(func: Callable):
        bus = get_event_bus()
        bus.subscribe(event_type, func)
        return func
    return decorator

