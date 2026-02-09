"""
Event Sourcing System
=====================

Sistema de event sourcing para auditoría y reconstrucción de estado.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipo de evento."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    STATE_CHANGED = "state_changed"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento."""
    event_id: str
    aggregate_id: str
    event_type: EventType
    event_data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventStore:
    """
    Store de eventos.
    
    Almacena eventos para event sourcing.
    """
    
    def __init__(self, storage_path: str = "data/events"):
        """
        Inicializar store de eventos.
        
        Args:
            storage_path: Ruta de almacenamiento
        """
        self.storage_path = storage_path
        self.events: Dict[str, List[Event]] = {}  # aggregate_id -> events
        self.all_events: List[Event] = []
        self.max_events = 100000
    
    def append_event(
        self,
        aggregate_id: str,
        event_type: EventType,
        event_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Agregar evento.
        
        Args:
            aggregate_id: ID del agregado
            event_type: Tipo de evento
            event_data: Datos del evento
            metadata: Metadata adicional
            
        Returns:
            Evento creado
        """
        # Obtener versión actual
        version = len(self.events.get(aggregate_id, [])) + 1
        
        event_id = f"evt_{len(self.all_events)}"
        event = Event(
            event_id=event_id,
            aggregate_id=aggregate_id,
            event_type=event_type,
            event_data=event_data,
            version=version,
            metadata=metadata or {}
        )
        
        if aggregate_id not in self.events:
            self.events[aggregate_id] = []
        self.events[aggregate_id].append(event)
        self.all_events.append(event)
        
        # Limitar tamaño
        if len(self.all_events) > self.max_events:
            self.all_events = self.all_events[-self.max_events:]
        
        logger.debug(f"Appended event: {event_type.value} for {aggregate_id}")
        
        return event
    
    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 1,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """
        Obtener eventos de agregado.
        
        Args:
            aggregate_id: ID del agregado
            from_version: Versión inicial
            to_version: Versión final (opcional)
            
        Returns:
            Lista de eventos
        """
        if aggregate_id not in self.events:
            return []
        
        events = self.events[aggregate_id]
        
        # Filtrar por versión
        filtered = [e for e in events if e.version >= from_version]
        if to_version:
            filtered = [e for e in filtered if e.version <= to_version]
        
        return filtered
    
    def get_all_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 1000
    ) -> List[Event]:
        """
        Obtener todos los eventos.
        
        Args:
            event_type: Filtrar por tipo
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        events = self.all_events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del store."""
        event_type_counts = {}
        for event in self.all_events:
            event_type_counts[event.event_type.value] = event_type_counts.get(
                event.event_type.value, 0
            ) + 1
        
        return {
            "total_events": len(self.all_events),
            "total_aggregates": len(self.events),
            "events_by_type": event_type_counts
        }


# Instancia global
_event_store: Optional[EventStore] = None


def get_event_store(storage_path: str = "data/events") -> EventStore:
    """Obtener instancia global del store de eventos."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore(storage_path=storage_path)
    return _event_store


