"""
Event Store - Stores and retrieves events
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import logging

from .event import Event

logger = logging.getLogger(__name__)


class IEventStore(ABC):
    """Interface for event store"""
    
    @abstractmethod
    async def append(self, events: List[Event]) -> bool:
        """Append events to store"""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 1,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """Get events for aggregate"""
        pass
    
    @abstractmethod
    async def get_events_by_type(
        self,
        event_type: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Event]:
        """Get events by type"""
        pass


class EventStore(IEventStore):
    """
    Event store implementation.
    Stores events and provides query interface.
    """
    
    def __init__(self, database_adapter):
        self.database = database_adapter
        self.events_table = "events"
    
    async def initialize(self):
        """Initialize event store"""
        # Create events table if not exists
        # This would be done via migrations
        pass
    
    async def append(self, events: List[Event]) -> bool:
        """Append events to store"""
        if not events:
            return True
        
        try:
            for event in events:
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "aggregate_id": event.aggregate_id,
                    "aggregate_type": event.aggregate_type,
                    "version": event.version,
                    "occurred_at": event.occurred_at.isoformat(),
                    "data": self._serialize_event(event),
                    "metadata": event.metadata,
                }
                
                await self.database.insert(self.events_table, event_data)
            
            logger.debug(f"Appended {len(events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append events: {e}", exc_info=True)
            return False
    
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 1,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """Get events for aggregate"""
        filter_conditions = {
            "aggregate_id": aggregate_id,
            "version": {"$gte": from_version}
        }
        
        if to_version:
            filter_conditions["version"]["$lte"] = to_version
        
        results = await self.database.query(
            self.events_table,
            filter_conditions=filter_conditions,
            order_by="version"
        )
        
        return [self._deserialize_event(data) for data in results]
    
    async def get_events_by_type(
        self,
        event_type: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Event]:
        """Get events by type"""
        filter_conditions = {"event_type": event_type}
        
        if from_date:
            filter_conditions["occurred_at"] = {"$gte": from_date.isoformat()}
        
        if to_date:
            if "occurred_at" not in filter_conditions:
                filter_conditions["occurred_at"] = {}
            filter_conditions["occurred_at"]["$lte"] = to_date.isoformat()
        
        results = await self.database.query(
            self.events_table,
            filter_conditions=filter_conditions,
            order_by="occurred_at"
        )
        
        return [self._deserialize_event(data) for data in results]
    
    def _serialize_event(self, event: Event) -> dict:
        """Serialize event to dictionary"""
        # Convert event to dict, excluding metadata fields
        import dataclasses
        data = dataclasses.asdict(event)
        # Remove metadata fields
        for key in ["event_id", "occurred_at", "event_type", "aggregate_id", "aggregate_type", "version", "metadata"]:
            data.pop(key, None)
        return data
    
    def _deserialize_event(self, data: dict) -> Event:
        """Deserialize event from dictionary"""
        # This would reconstruct the event based on event_type
        # Simplified for example
        from .event import DomainEvent
        
        event_data = data.get("data", {})
        event_data["event_id"] = data["event_id"]
        event_data["occurred_at"] = datetime.fromisoformat(data["occurred_at"])
        event_data["aggregate_id"] = data["aggregate_id"]
        event_data["version"] = data["version"]
        event_data["metadata"] = data.get("metadata", {})
        
        # Create appropriate event class based on event_type
        # This is simplified - in production, use a registry
        return DomainEvent(**event_data)


# Global event store
_event_store: Optional[EventStore] = None


def get_event_store(database_adapter=None) -> Optional[EventStore]:
    """Get or create event store"""
    global _event_store
    if _event_store is None and database_adapter:
        _event_store = EventStore(database_adapter)
        import asyncio
        asyncio.create_task(_event_store.initialize())
    return _event_store

