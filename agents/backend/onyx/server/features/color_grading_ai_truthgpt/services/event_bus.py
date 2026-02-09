"""
Event Bus for Color Grading AI
===============================

Event-driven architecture with pub/sub pattern.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    TEMPLATE_APPLIED = "template_applied"
    PRESET_CREATED = "preset_created"
    VERSION_CREATED = "version_created"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    METRIC_RECORDED = "metric_recorded"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class Event:
    """Event data structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "system"


class EventBus:
    """
    Event bus for pub/sub pattern.
    
    Features:
    - Publish/subscribe pattern
    - Event filtering
    - Async event handling
    - Event history
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Handler function (async or sync)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from event type."""
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    async def publish(self, event_type: EventType, data: Dict[str, Any], source: str = "system"):
        """
        Publish event.
        
        Args:
            event_type: Event type
            data: Event data
            source: Event source
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}", exc_info=True)
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            limit: Maximum results
            
        Returns:
            List of events
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        events = events[-limit:]
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source,
                "data": e.data,
            }
            for e in events
        ]
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for event type."""
        return len(self._subscribers.get(event_type, []))




