"""
Event Bus Service
Sistema de eventos para arquitectura event-driven
"""

import logging
from typing import Dict, Any, Callable, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    MUSIC_GENERATED = "music_generated"
    AUDIO_PROCESSED = "audio_processed"
    USER_CREATED = "user_created"
    SONG_UPDATED = "song_updated"
    SEARCH_PERFORMED = "search_performed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Representa un evento"""
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: datetime = None
    event_id: str = None
    source: str = "suno-clone-ai"
    version: str = "1.0"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())


class EventBus:
    """
    Bus de eventos para arquitectura event-driven
    Soporta múltiples subscribers por evento
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._enabled = True
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Suscribe un handler a un tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función async que maneja el evento
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Desuscribe un handler de un tipo de evento"""
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler from {event_type.value}")
    
    async def publish(self, event: Event) -> bool:
        """
        Publica un evento a todos los subscribers
        
        Args:
            event: Evento a publicar
            
        Returns:
            True si fue publicado exitosamente
        """
        if not self._enabled:
            logger.warning("Event bus is disabled")
            return False
        
        try:
            # Guardar en historial
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Obtener subscribers
            handlers = self._subscribers.get(event.event_type, [])
            
            if not handlers:
                logger.debug(f"No subscribers for event type {event.event_type.value}")
                return True
            
            # Ejecutar handlers en paralelo
            tasks = [self._execute_handler(handler, event) for handler in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verificar errores
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.error(f"Errors in event handlers: {errors}")
                return False
            
            logger.info(f"Published event {event.event_type.value} to {len(handlers)} subscribers")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}", exc_info=True)
            return False
    
    async def _execute_handler(self, handler: Callable, event: Event):
        """Ejecuta un handler de evento"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Error in event handler: {e}", exc_info=True)
            raise
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Obtiene el historial de eventos"""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Obtiene el número de subscribers para un tipo de evento"""
        return len(self._subscribers.get(event_type, []))
    
    def enable(self):
        """Habilita el event bus"""
        self._enabled = True
        logger.info("Event bus enabled")
    
    def disable(self):
        """Deshabilita el event bus"""
        self._enabled = False
        logger.info("Event bus disabled")


# Instancia global
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Obtiene la instancia global del event bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus















