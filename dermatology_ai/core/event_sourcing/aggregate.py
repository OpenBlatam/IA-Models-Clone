"""
Aggregate Root for Event Sourcing
"""

from abc import ABC
from typing import List
from .event import Event, DomainEvent

class AggregateRoot(ABC):
    """
    Base class for aggregates in event sourcing.
    Tracks uncommitted events and applies events to rebuild state.
    """
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self._uncommitted_events: List[DomainEvent] = []
    
    def raise_event(self, event: DomainEvent):
        """Raise domain event"""
        event.aggregate_id = self.aggregate_id
        event.aggregate_type = self.__class__.__name__
        event.version = self.version + 1
        
        self._uncommitted_events.append(event)
        self.apply_event(event)
        self.version += 1
    
    def apply_event(self, event: DomainEvent):
        """Apply event to aggregate (override in subclasses)"""
        # Subclasses implement specific event application logic
        pass
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Get uncommitted events"""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        """Mark events as committed"""
        self._uncommitted_events.clear()
    
    def load_from_history(self, events: List[Event]):
        """Load aggregate state from event history"""
        for event in events:
            self.apply_event(event)
            self.version = event.version















