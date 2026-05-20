"""
Event Definitions for Event Sourcing
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class Event(ABC):
    """Base class for all events"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_type: str = field(init=False)
    aggregate_id: str = ""
    aggregate_type: str = ""
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = self.__class__.__name__


@dataclass
class DomainEvent(Event):
    """Domain event - represents something that happened in the domain"""
    pass


# Analysis Domain Events
@dataclass
class AnalysisCreatedEvent(DomainEvent):
    """Event when analysis is created"""
    user_id: str
    image_url: Optional[str] = None


@dataclass
class AnalysisStartedEvent(DomainEvent):
    """Event when analysis processing starts"""
    analysis_id: str


@dataclass
class AnalysisCompletedEvent(DomainEvent):
    """Event when analysis is completed"""
    analysis_id: str
    metrics: Dict[str, float]
    conditions: list[Dict[str, Any]]


@dataclass
class AnalysisFailedEvent(DomainEvent):
    """Event when analysis fails"""
    analysis_id: str
    error: str


# User Domain Events
@dataclass
class UserCreatedEvent(DomainEvent):
    """Event when user is created"""
    user_id: str
    email: str


@dataclass
class UserPreferencesUpdatedEvent(DomainEvent):
    """Event when user preferences are updated"""
    user_id: str
    preferences: Dict[str, Any]















