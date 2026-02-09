"""
Calendar Manager
================

Gestión de calendarios y eventos para artistas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos."""
    CONCERT = "concert"
    INTERVIEW = "interview"
    PHOTOSHOOT = "photoshoot"
    REHEARSAL = "rehearsal"
    MEETING = "meeting"
    TRAVEL = "travel"
    REST = "rest"
    OTHER = "other"


@dataclass
class CalendarEvent:
    """Evento del calendario."""
    id: str
    title: str
    description: str
    event_type: EventType
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = None
    reminders: List[datetime] = None
    protocol_requirements: List[str] = None
    wardrobe_requirements: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []
        if self.reminders is None:
            self.reminders = []
        if self.protocol_requirements is None:
            self.protocol_requirements = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        if self.reminders:
            data['reminders'] = [r.isoformat() for r in self.reminders]
        return data


class CalendarManager:
    """Gestor de calendarios para artistas."""
    
    def __init__(self, artist_id: str):
        """
        Inicializar gestor de calendario.
        
        Args:
            artist_id: ID del artista
        """
        self.artist_id = artist_id
        self.events: Dict[str, CalendarEvent] = {}
        self._logger = logger
    
    def add_event(self, event: CalendarEvent) -> CalendarEvent:
        """
        Agregar evento al calendario.
        
        Args:
            event: Evento a agregar
        
        Returns:
            Evento agregado
        """
        if event.id in self.events:
            raise ValueError(f"Event with id {event.id} already exists")
        
        self.events[event.id] = event
        self._logger.info(f"Added event {event.id} to calendar for artist {self.artist_id}")
        return event
    
    def get_event(self, event_id: str) -> Optional[CalendarEvent]:
        """
        Obtener evento por ID.
        
        Args:
            event_id: ID del evento
        
        Returns:
            Evento o None si no existe
        """
        return self.events.get(event_id)
    
    def update_event(self, event_id: str, **updates) -> CalendarEvent:
        """
        Actualizar evento.
        
        Args:
            event_id: ID del evento
            **updates: Campos a actualizar
        
        Returns:
            Evento actualizado
        """
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found")
        
        event = self.events[event_id]
        for key, value in updates.items():
            if hasattr(event, key):
                setattr(event, key, value)
        
        self._logger.info(f"Updated event {event_id} for artist {self.artist_id}")
        return event
    
    def delete_event(self, event_id: str) -> bool:
        """
        Eliminar evento.
        
        Args:
            event_id: ID del evento
        
        Returns:
            True si se eliminó, False si no existía
        """
        if event_id in self.events:
            del self.events[event_id]
            self._logger.info(f"Deleted event {event_id} for artist {self.artist_id}")
            return True
        return False
    
    def get_events_by_date(self, date: datetime) -> List[CalendarEvent]:
        """
        Obtener eventos de una fecha específica.
        
        Args:
            date: Fecha a consultar
        
        Returns:
            Lista de eventos del día
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        return [
            event for event in self.events.values()
            if start_of_day <= event.start_time < end_of_day
        ]
    
    def get_upcoming_events(self, days: int = 7) -> List[CalendarEvent]:
        """
        Obtener eventos próximos.
        
        Args:
            days: Número de días a futuro
        
        Returns:
            Lista de eventos próximos
        """
        now = datetime.now()
        end_date = now + timedelta(days=days)
        
        upcoming = [
            event for event in self.events.values()
            if now <= event.start_time <= end_date
        ]
        
        return sorted(upcoming, key=lambda e: e.start_time)
    
    def get_events_by_type(self, event_type: EventType) -> List[CalendarEvent]:
        """
        Obtener eventos por tipo.
        
        Args:
            event_type: Tipo de evento
        
        Returns:
            Lista de eventos del tipo especificado
        """
        return [
            event for event in self.events.values()
            if event.event_type == event_type
        ]
    
    def check_conflicts(self, start_time: datetime, end_time: datetime, exclude_event_id: Optional[str] = None) -> List[CalendarEvent]:
        """
        Verificar conflictos de horario.
        
        Args:
            start_time: Hora de inicio
            end_time: Hora de fin
            exclude_event_id: ID de evento a excluir de la verificación
        
        Returns:
            Lista de eventos que conflictúan
        """
        conflicts = []
        for event in self.events.values():
            if exclude_event_id and event.id == exclude_event_id:
                continue
            
            # Verificar si hay solapamiento
            if (start_time < event.end_time and end_time > event.start_time):
                conflicts.append(event)
        
        return conflicts
    
    def get_all_events(self) -> List[CalendarEvent]:
        """
        Obtener todos los eventos.
        
        Returns:
            Lista de todos los eventos ordenados por fecha
        """
        return sorted(self.events.values(), key=lambda e: e.start_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir calendario a diccionario."""
        return {
            "artist_id": self.artist_id,
            "events": [event.to_dict() for event in self.get_all_events()],
            "total_events": len(self.events)
        }




