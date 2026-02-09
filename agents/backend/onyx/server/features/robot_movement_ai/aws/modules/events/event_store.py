"""
Event Store
===========

Event store for event sourcing.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from aws.modules.events.event_bus import Event
from aws.modules.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)


class EventStore:
    """Event store for persisting events."""
    
    def __init__(self, repository: RepositoryPort):
        self.repository = repository
    
    async def append(self, event: Event) -> bool:
        """Append event to store."""
        try:
            # In production, use proper event store implementation
            # For now, we'll use the repository
            await self.repository.create(event)
            logger.debug(f"Stored event: {event.event_type} ({event.event_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events from store."""
        filters = {}
        if event_type:
            filters["event_type"] = event_type
        if start_time:
            filters["timestamp__gte"] = start_time
        if end_time:
            filters["timestamp__lte"] = end_time
        
        events = await self.repository.get_all(filters)
        if limit:
            events = events[:limit]
        
        return events
    
    async def get_stream(self, stream_id: str) -> List[Event]:
        """Get event stream."""
        filters = {"correlation_id": stream_id}
        return await self.repository.get_all(filters)















