"""
Routine Models
==============

Modelos de datos para rutinas.
"""

from typing import List, Optional, Dict, Any
from datetime import time
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseModel, TimestampMixin


class RoutineType(str, Enum):
    """Tipos de rutinas."""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


class RoutineStatus(str, Enum):
    """Estado de rutina."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    OVERDUE = "overdue"


@dataclass
class RoutineModel(BaseModel, TimestampMixin):
    """Modelo de rutina."""
    title: str
    description: str
    routine_type: RoutineType
    scheduled_time: time
    duration_minutes: int
    priority: int = 5
    days_of_week: List[int] = field(default_factory=lambda: list(range(7)))
    is_required: bool = True
    reminders: List[time] = field(default_factory=list)
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
            "routine_type": self.routine_type.value,
            "scheduled_time": self.scheduled_time.strftime("%H:%M:%S"),
            "duration_minutes": self.duration_minutes,
            "priority": self.priority,
            "days_of_week": self.days_of_week,
            "is_required": self.is_required,
            "reminders": [r.strftime("%H:%M:%S") for r in self.reminders],
            "notes": self.notes
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validar modelo."""
        if not self.title:
            return False, "Title is required"
        
        if not (1 <= self.priority <= 10):
            return False, "Priority must be between 1 and 10"
        
        if not (0 <= self.duration_minutes <= 1440):
            return False, "Duration must be between 0 and 1440 minutes"
        
        if not all(0 <= day <= 6 for day in self.days_of_week):
            return False, "Days of week must be between 0 and 6"
        
        return True, None




