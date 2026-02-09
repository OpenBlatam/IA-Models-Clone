"""
Event Sourcing - Sistema de eventos para auditoría completa
============================================================
"""

import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos"""
    PAPER_UPLOADED = "paper_uploaded"
    PAPER_PROCESSED = "paper_processed"
    MODEL_TRAINED = "model_trained"
    CODE_IMPROVED = "code_improved"
    CODE_ANALYZED = "code_analyzed"
    TEST_GENERATED = "test_generated"
    GIT_COMMIT = "git_commit"
    GIT_PR_CREATED = "git_pr_created"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Event:
    """Evento individual"""
    id: str
    type: EventType
    aggregate_id: str  # ID del agregado (paper, code, etc.)
    aggregate_type: str  # Tipo de agregado
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario"""
        return {
            "id": self.id,
            "type": self.type.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "version": self.version
        }


class EventStore:
    """Almacén de eventos"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.events: List[Event] = []
        self.aggregates: Dict[str, List[Event]] = {}  # aggregate_id -> events
        self.storage_path = storage_path
        self._load_events()
    
    def _load_events(self):
        """Carga eventos desde almacenamiento persistente"""
        if not self.storage_path:
            return
        
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for event_data in data:
                        event = Event(
                            id=event_data["id"],
                            type=EventType(event_data["type"]),
                            aggregate_id=event_data["aggregate_id"],
                            aggregate_type=event_data["aggregate_type"],
                            payload=event_data["payload"],
                            metadata=event_data["metadata"],
                            timestamp=datetime.fromisoformat(event_data["timestamp"]),
                            user_id=event_data.get("user_id"),
                            version=event_data.get("version", 1)
                        )
                        self._add_event_internal(event)
        except Exception as e:
            logger.error(f"Error cargando eventos: {e}")
    
    def _save_events(self):
        """Guarda eventos en almacenamiento persistente"""
        if not self.storage_path:
            return
        
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([e.to_dict() for e in self.events], f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando eventos: {e}")
    
    def _add_event_internal(self, event: Event):
        """Agrega evento internamente"""
        self.events.append(event)
        if event.aggregate_id not in self.aggregates:
            self.aggregates[event.aggregate_id] = []
        self.aggregates[event.aggregate_id].append(event)
    
    def append_event(self, event: Event):
        """Agrega un nuevo evento"""
        self._add_event_internal(event)
        self._save_events()
    
    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Obtiene eventos con filtros"""
        events = self.events
        
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_aggregate_events(self, aggregate_id: str) -> List[Event]:
        """Obtiene todos los eventos de un agregado"""
        return self.aggregates.get(aggregate_id, [])
    
    def replay_events(
        self,
        aggregate_id: str,
        handler: Callable[[Event], Any]
    ) -> Any:
        """Reconstruye el estado de un agregado aplicando eventos"""
        events = self.get_aggregate_events(aggregate_id)
        state = None
        
        for event in sorted(events, key=lambda e: e.timestamp):
            state = handler(event, state)
        
        return state


class EventSourcing:
    """Sistema de Event Sourcing"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.event_store = EventStore(storage_path)
        self.subscribers: Dict[EventType, List[Callable]] = {}
    
    def publish_event(
        self,
        event_type: EventType,
        aggregate_id: str,
        aggregate_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Event:
        """Publica un nuevo evento"""
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            payload=payload,
            metadata=metadata or {},
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        self.event_store.append_event(event)
        
        # Notificar suscriptores
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error en subscriber para {event_type}: {e}")
        
        logger.info(f"Evento publicado: {event_type.value} para {aggregate_id}")
        return event
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ):
        """Suscribe un handler a un tipo de evento"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def get_event_history(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene el historial de eventos"""
        events = self.event_store.get_events(
            aggregate_id=aggregate_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return [e.to_dict() for e in events]
    
    def rebuild_aggregate(
        self,
        aggregate_id: str,
        handler: Callable[[Event, Any], Any]
    ) -> Any:
        """Reconstruye el estado de un agregado"""
        return self.event_store.replay_events(aggregate_id, handler)
    
    def get_aggregate_snapshot(self, aggregate_id: str) -> Dict[str, Any]:
        """Obtiene un snapshot del estado actual de un agregado"""
        events = self.event_store.get_aggregate_events(aggregate_id)
        return {
            "aggregate_id": aggregate_id,
            "total_events": len(events),
            "first_event": events[0].to_dict() if events else None,
            "last_event": events[-1].to_dict() if events else None,
            "events": [e.to_dict() for e in events]
        }




