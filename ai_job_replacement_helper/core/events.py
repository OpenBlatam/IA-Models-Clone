"""
Events Service - Sistema de eventos y webinars
==============================================

Sistema de eventos, webinars y workshops para desarrollo profesional.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Tipos de eventos"""
    WEBINAR = "webinar"
    WORKSHOP = "workshop"
    NETWORKING = "networking"
    CONFERENCE = "conference"
    MENTORING_SESSION = "mentoring_session"
    Q_A = "q_a"


class EventStatus(str, Enum):
    """Estado del evento"""
    UPCOMING = "upcoming"
    LIVE = "live"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Event:
    """Evento"""
    id: str
    title: str
    description: str
    event_type: EventType
    status: EventStatus
    start_date: datetime
    end_date: datetime
    organizer_id: str
    max_participants: Optional[int] = None
    registered_participants: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    meeting_url: Optional[str] = None
    recording_url: Optional[str] = None
    resources: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class EventsService:
    """Servicio de eventos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.events: Dict[str, Event] = {}
        logger.info("EventsService initialized")
    
    def create_event(
        self,
        organizer_id: str,
        title: str,
        description: str,
        event_type: EventType,
        start_date: datetime,
        end_date: datetime,
        max_participants: Optional[int] = None,
        tags: Optional[List[str]] = None,
        meeting_url: Optional[str] = None
    ) -> Event:
        """Crear nuevo evento"""
        event = Event(
            id=f"event_{organizer_id}_{int(datetime.now().timestamp())}",
            title=title,
            description=description,
            event_type=event_type,
            status=EventStatus.UPCOMING,
            start_date=start_date,
            end_date=end_date,
            organizer_id=organizer_id,
            max_participants=max_participants,
            tags=tags or [],
            meeting_url=meeting_url,
        )
        
        self.events[event.id] = event
        
        logger.info(f"Event created: {event.id}")
        return event
    
    def register_for_event(self, user_id: str, event_id: str) -> bool:
        """Registrarse para un evento"""
        event = self.events.get(event_id)
        if not event:
            return False
        
        if user_id in event.registered_participants:
            return False  # Ya registrado
        
        if event.max_participants and len(event.registered_participants) >= event.max_participants:
            return False  # Lleno
        
        event.registered_participants.append(user_id)
        return True
    
    def get_upcoming_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 20
    ) -> List[Event]:
        """Obtener eventos próximos"""
        now = datetime.now()
        events = [
            e for e in self.events.values()
            if e.status == EventStatus.UPCOMING and e.start_date > now
        ]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        events.sort(key=lambda x: x.start_date)
        return events[:limit]
    
    def get_user_events(self, user_id: str) -> Dict[str, List[Event]]:
        """Obtener eventos del usuario (organizados y registrados)"""
        organized = [
            e for e in self.events.values()
            if e.organizer_id == user_id
        ]
        
        registered = [
            e for e in self.events.values()
            if user_id in e.registered_participants
        ]
        
        return {
            "organized": organized,
            "registered": registered,
        }
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Obtener evento específico"""
        return self.events.get(event_id)
    
    def update_event_status(self, event_id: str, status: EventStatus) -> Event:
        """Actualizar estado del evento"""
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")
        
        event.status = status
        return event




