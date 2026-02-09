"""
Event Store CQRS
================

Event store for CQRS pattern.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DomainEvent:
    """Domain event."""
    id: str
    aggregate_id: str
    event_type: str
    payload: Dict[str, Any]
    version: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventStoreCQRS:
    """Event store for CQRS."""
    
    def __init__(self):
        self._events: List[DomainEvent] = []
        self._aggregates: Dict[str, List[DomainEvent]] = {}  # aggregate_id -> events
    
    def append_event(self, event: DomainEvent):
        """Append event to store."""
        self._events.append(event)
        
        if event.aggregate_id not in self._aggregates:
            self._aggregates[event.aggregate_id] = []
        
        self._aggregates[event.aggregate_id].append(event)
        logger.debug(f"Appended event {event.id} for aggregate {event.aggregate_id}")
    
    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        from_version: int = 0,
        limit: int = 1000
    ) -> List[DomainEvent]:
        """Get events."""
        events = self._events
        
        if aggregate_id:
            events = self._aggregates.get(aggregate_id, [])
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        events = [e for e in events if e.version >= from_version]
        
        return events[-limit:]
    
    def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of aggregate."""
        events = self._aggregates.get(aggregate_id, [])
        return events[-1].version if events else 0
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event store statistics."""
        return {
            "total_events": len(self._events),
            "total_aggregates": len(self._aggregates),
            "by_type": {
                event_type: sum(1 for e in self._events if e.event_type == event_type)
                for event_type in set(e.event_type for e in self._events)
            }
        }















