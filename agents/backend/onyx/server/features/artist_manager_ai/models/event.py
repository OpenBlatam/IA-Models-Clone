"""
Event Models
============

Modelos de datos para eventos.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseModel, TimestampMixin


class EventType(str, Enum):
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
class EventModel(BaseModel, TimestampMixin):
    """Modelo de evento."""
    title: str
    description: str
    event_type: EventType
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    reminders: List[datetime] = field(default_factory=list)
    protocol_requirements: List[str] = field(default_factory=list)
    wardrobe_requirements: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        base_dict = super().to_dict()
        timestamp_dict = TimestampMixin.to_dict(self)
        
        return {
            **base_dict,
            **timestamp_dict,
            "title": self.title,
            "description": self.description,
            "event_type": self.event_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "location": self.location,
            "attendees": self.attendees,
            "reminders": [r.isoformat() for r in self.reminders],
            "protocol_requirements": self.protocol_requirements,
            "wardrobe_requirements": self.wardrobe_requirements,
            "notes": self.notes
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validar modelo.
        
        Returns:
            (es_válido, mensaje_error)
        """
        if not self.title:
            return False, "Title is required"
        
        if self.end_time <= self.start_time:
            return False, "End time must be after start time"
        
        duration = (self.end_time - self.start_time).total_seconds() / 3600
        if duration > 24:
            return False, "Event duration cannot exceed 24 hours"
        
        return True, None




