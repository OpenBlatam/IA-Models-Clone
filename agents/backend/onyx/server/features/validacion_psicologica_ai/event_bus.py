"""
Sistema de Eventos y Bus de Eventos
====================================
Event-driven architecture
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import structlog
import asyncio
from collections import defaultdict

logger = structlog.get_logger()


class EventType(str, Enum):
    """Tipos de eventos"""
    VALIDATION_CREATED = "validation_created"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    PROFILE_GENERATED = "profile_generated"
    REPORT_GENERATED = "report_generated"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    ALERT_TRIGGERED = "alert_triggered"
    RECOMMENDATION_CREATED = "recommendation_created"
    USER_REGISTERED = "user_registered"
    QUOTA_EXCEEDED = "quota_exceeded"


class Event:
    """Evento"""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.id = uuid4()
        self.event_type = event_type
        self.data = data
        self.source = source or "system"
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }


class EventBus:
    """Bus de eventos"""
    
    def __init__(self):
        """Inicializar bus"""
        self._subscribers: Dict[EventType, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 10000
        logger.info("EventBus initialized")
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """
        Suscribirse a tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        self._subscribers[event_type].append(handler)
        logger.info("Subscribed to event", event_type=event_type.value)
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """
        Desuscribirse de tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.info("Unsubscribed from event", event_type=event_type.value)
    
    async def publish(
        self,
        event: Event
    ) -> None:
        """
        Publicar evento
        
        Args:
            event: Evento a publicar
        """
        # Agregar a historial
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notificar suscriptores
        handlers = self._subscribers.get(event.event_type, [])
        
        if handlers:
            logger.info(
                "Event published",
                event_type=event.event_type.value,
                subscribers=len(handlers)
            )
            
            # Ejecutar handlers en paralelo
            tasks = [handler(event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.debug(
                "Event published (no subscribers)",
                event_type=event.event_type.value
            )
    
    async def publish_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> Event:
        """
        Publicar evento (método helper)
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Fuente del evento
            
        Returns:
            Evento creado
        """
        event = Event(event_type, data, source)
        await self.publish(event)
        return event
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Obtener historial de eventos
        
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
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """
        Obtener número de suscriptores
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Número de suscriptores
        """
        return len(self._subscribers.get(event_type, []))


# Instancia global del bus de eventos
event_bus = EventBus()




