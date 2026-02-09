"""
Event Sourcing System
=====================
Sistema de event sourcing para auditoría y reconstrucción de estado
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    """Tipos de eventos"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    PROCESSED = "processed"
    SHARED = "shared"
    EXPORTED = "exported"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento"""
    id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    version: int = 1


@dataclass
class Snapshot:
    """Snapshot de estado"""
    aggregate_id: str
    aggregate_type: str
    state: Dict[str, Any]
    version: int
    timestamp: float


class EventSourcing:
    """
    Sistema de event sourcing
    """
    
    def __init__(self):
        self.events: Dict[str, List[Event]] = defaultdict(list)  # aggregate_id -> events
        self.snapshots: Dict[str, Snapshot] = {}  # aggregate_id -> snapshot
        self.event_store: List[Event] = []  # Todos los eventos en orden
        self.snapshot_interval = 100  # Crear snapshot cada N eventos
    
    def record_event(
        self,
        event_type: EventType,
        aggregate_id: str,
        aggregate_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Event:
        """
        Registrar evento
        
        Args:
            event_type: Tipo de evento
            aggregate_id: ID del agregado
            aggregate_type: Tipo del agregado
            data: Datos del evento
            metadata: Metadata adicional
            user_id: ID del usuario
        """
        event_id = f"event_{int(time.time() * 1000)}"
        
        # Obtener versión actual
        version = len(self.events[aggregate_id]) + 1
        
        event = Event(
            id=event_id,
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            data=data,
            metadata=metadata or {},
            timestamp=time.time(),
            user_id=user_id,
            version=version
        )
        
        self.events[aggregate_id].append(event)
        self.event_store.append(event)
        
        # Crear snapshot si es necesario
        if version % self.snapshot_interval == 0:
            self._create_snapshot(aggregate_id, aggregate_type)
        
        return event
    
    def get_events(
        self,
        aggregate_id: str,
        event_type: Optional[EventType] = None,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """
        Obtener eventos de un agregado
        
        Args:
            aggregate_id: ID del agregado
            event_type: Filtrar por tipo
            from_version: Versión inicial
            to_version: Versión final
        """
        events = self.events.get(aggregate_id, [])
        
        # Filtrar por tipo
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Filtrar por versión
        if from_version:
            events = [e for e in events if e.version >= from_version]
        if to_version:
            events = [e for e in events if e.version <= to_version]
        
        return events
    
    def replay_events(
        self,
        aggregate_id: str,
        from_version: int = 1,
        to_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reconstruir estado desde eventos
        
        Args:
            aggregate_id: ID del agregado
            from_version: Versión inicial
            to_version: Versión final
        """
        # Cargar snapshot más reciente si existe
        state = {}
        start_version = from_version
        
        if aggregate_id in self.snapshots:
            snapshot = self.snapshots[aggregate_id]
            if snapshot.version < from_version:
                state = snapshot.state.copy()
                start_version = snapshot.version + 1
        
        # Aplicar eventos desde snapshot
        events = self.get_events(aggregate_id, from_version=start_version, to_version=to_version)
        
        for event in events:
            state = self._apply_event(state, event)
        
        return state
    
    def _apply_event(self, state: Dict[str, Any], event: Event) -> Dict[str, Any]:
        """Aplicar evento al estado"""
        if event.event_type == EventType.CREATED:
            state = event.data.copy()
        elif event.event_type == EventType.UPDATED:
            state.update(event.data)
        elif event.event_type == EventType.DELETED:
            state['deleted'] = True
            state['deleted_at'] = event.timestamp
        elif event.event_type == EventType.PROCESSED:
            state['processed'] = True
            state['processed_at'] = event.timestamp
            state.update(event.data)
        elif event.event_type == EventType.SHARED:
            if 'shares' not in state:
                state['shares'] = []
            state['shares'].append(event.data)
        elif event.event_type == EventType.EXPORTED:
            if 'exports' not in state:
                state['exports'] = []
            state['exports'].append(event.data)
        else:
            # Evento personalizado
            if 'custom_events' not in state:
                state['custom_events'] = []
            state['custom_events'].append({
                'type': event.event_type.value,
                'data': event.data,
                'timestamp': event.timestamp
            })
        
        return state
    
    def _create_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str
    ):
        """Crear snapshot del estado actual"""
        state = self.replay_events(aggregate_id)
        
        events = self.events.get(aggregate_id, [])
        version = len(events)
        
        snapshot = Snapshot(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            state=state,
            version=version,
            timestamp=time.time()
        )
        
        self.snapshots[aggregate_id] = snapshot
    
    def get_event_history(
        self,
        aggregate_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Obtener historial de eventos"""
        events = self.events.get(aggregate_id, [])
        
        if limit:
            events = events[-limit:]
        
        return [
            {
                'id': e.id,
                'type': e.event_type.value,
                'version': e.version,
                'timestamp': e.timestamp,
                'user_id': e.user_id,
                'data': e.data,
                'metadata': e.metadata
            }
            for e in events
        ]
    
    def search_events(
        self,
        aggregate_type: Optional[str] = None,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        from_timestamp: Optional[float] = None,
        to_timestamp: Optional[float] = None
    ) -> List[Event]:
        """Buscar eventos"""
        results = []
        
        for event in self.event_store:
            if aggregate_type and event.aggregate_type != aggregate_type:
                continue
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if from_timestamp and event.timestamp < from_timestamp:
                continue
            if to_timestamp and event.timestamp > to_timestamp:
                continue
            
            results.append(event)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        event_types = defaultdict(int)
        aggregate_types = defaultdict(int)
        
        for event in self.event_store:
            event_types[event.event_type.value] += 1
            aggregate_types[event.aggregate_type] += 1
        
        return {
            'total_events': len(self.event_store),
            'total_aggregates': len(self.events),
            'total_snapshots': len(self.snapshots),
            'events_by_type': dict(event_types),
            'aggregates_by_type': dict(aggregate_types),
            'snapshot_interval': self.snapshot_interval
        }


# Instancia global
event_sourcing = EventSourcing()

