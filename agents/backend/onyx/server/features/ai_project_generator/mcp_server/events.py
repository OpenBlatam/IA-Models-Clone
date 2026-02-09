"""
MCP Event Sourcing - Event sourcing patterns
==============================================
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Tipos de eventos"""
    RESOURCE_CREATED = "resource.created"
    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"
    QUERY_EXECUTED = "query.executed"
    OPERATION_FAILED = "operation.failed"
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"


class Event(BaseModel):
    """Evento del sistema"""
    event_id: str = Field(..., description="ID único del evento")
    event_type: EventType = Field(..., description="Tipo de evento")
    aggregate_id: str = Field(..., description="ID del agregado")
    aggregate_type: str = Field(..., description="Tipo del agregado")
    payload: Dict[str, Any] = Field(..., description="Payload del evento")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, description="Versión del evento")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class EventStore:
    """
    Store de eventos para event sourcing
    
    Almacena eventos de forma inmutable para reconstrucción de estado.
    """
    
    def __init__(self):
        self._events: List[Event] = []
        self._handlers: Dict[EventType, List[Callable]] = {}
    
    def append(self, event: Event):
        """
        Agrega un evento al store
        
        Args:
            event: Evento a agregar
        """
        self._events.append(event)
        logger.info(f"Event stored: {event.event_type.value} for {event.aggregate_id}")
        
        # Disparar handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Obtiene eventos
        
        Args:
            aggregate_id: Filtrar por agregado (opcional)
            event_type: Filtrar por tipo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        events = self._events
        
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Suscribe handler a tipo de evento
        
        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type.value}")
    
    def replay_events(
        self,
        aggregate_id: str,
        handler: Callable,
    ):
        """
        Replay eventos para reconstruir estado
        
        Args:
            aggregate_id: ID del agregado
            handler: Handler para procesar eventos
        """
        events = self.get_events(aggregate_id=aggregate_id)
        
        for event in events:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error replaying event {event.event_id}: {e}")


class EventPublisher:
    """
    Publisher de eventos
    
    Publica eventos al event store y notifica subscribers.
    """
    
    def __init__(self, event_store: EventStore):
        """
        Args:
            event_store: Instancia de EventStore
        """
        self.event_store = event_store
    
    def publish(
        self,
        event_type: EventType,
        aggregate_id: str,
        aggregate_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Publica un evento
        
        Args:
            event_type: Tipo de evento
            aggregate_id: ID del agregado
            aggregate_type: Tipo del agregado
            payload: Payload del evento
            metadata: Metadata adicional
            
        Returns:
            Evento creado
        """
        import uuid
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            payload=payload,
            metadata=metadata or {},
        )
        
        self.event_store.append(event)
        
        return event

