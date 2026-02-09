"""
Protocol Models
===============

Modelos de datos para protocolos.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseModel, TimestampMixin


class ProtocolCategory(str, Enum):
    """Categorías de protocolos."""
    SOCIAL_MEDIA = "social_media"
    INTERVIEW = "interview"
    PUBLIC_APPEARANCE = "public_appearance"
    NETWORKING = "networking"
    CONTRACT = "contract"
    MEDIA = "media"
    GENERAL = "general"


class ProtocolPriority(str, Enum):
    """Prioridad de protocolo."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProtocolModel(BaseModel, TimestampMixin):
    """Modelo de protocolo."""
    title: str
    description: str
    category: ProtocolCategory
    priority: ProtocolPriority
    rules: List[str] = field(default_factory=list)
    do_s: List[str] = field(default_factory=list)
    dont_s: List[str] = field(default_factory=list)
    context: Optional[str] = None
    applicable_events: List[str] = field(default_factory=list)
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
            "category": self.category.value,
            "priority": self.priority.value,
            "rules": self.rules,
            "do_s": self.do_s,
            "dont_s": self.dont_s,
            "context": self.context,
            "applicable_events": self.applicable_events,
            "notes": self.notes
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validar modelo."""
        if not self.title:
            return False, "Title is required"
        
        if not self.rules:
            return False, "At least one rule is required"
        
        return True, None




