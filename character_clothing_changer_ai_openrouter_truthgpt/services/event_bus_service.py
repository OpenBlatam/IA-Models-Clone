"""
Event Bus Service
=================
Service for pub/sub event communication
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event definition"""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    event_id: Optional[str] = None


class EventBusService:
    """
    Service for pub/sub event communication.
    
    Features:
    - Publish/subscribe pattern
    - Multiple subscribers per event
    - Async event handling
    - Event history (optional)
    - Statistics
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus service.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._stats = {
            'events_published': 0,
            'events_delivered': 0,
            'events_failed': 0
        }
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]]
    ):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]]
    ) -> bool:
        """
        Unsubscribe from event type.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        
        Returns:
            True if handler was removed
        """
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed from event type: {event_type}")
                return True
        return False
    
    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: Optional[str] = None,
        event_id: Optional[str] = None
    ) -> Event:
        """
        Publish an event.
        
        Args:
            event_type: Event type
            payload: Event payload
            source: Event source
            event_id: Optional event ID
        
        Returns:
            Event object
        """
        event = Event(
            event_type=event_type,
            payload=payload,
            source=source,
            event_id=event_id
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history.pop(0)
        
        self._stats['events_published'] += 1
        
        # Get subscribers
        subscribers = self._subscribers.get(event_type, [])
        
        # Also notify wildcard subscribers
        if '*' in self._subscribers:
            subscribers.extend(self._subscribers['*'])
        
        if not subscribers:
            logger.debug(f"Event '{event_type}' published but no subscribers")
            return event
        
        # Notify all subscribers
        tasks = [self._notify_subscriber(subscriber, event) for subscriber in subscribers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        self._stats['events_delivered'] += successful
        self._stats['events_failed'] += failed
        
        logger.info(
            f"Event '{event_type}' published to {len(subscribers)} subscribers "
            f"({successful} successful, {failed} failed)"
        )
        
        return event
    
    async def _notify_subscriber(
        self,
        handler: Callable[[Event], Awaitable[None]],
        event: Event
    ):
        """Notify a single subscriber"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in event handler for '{event.event_type}': {e}")
            raise
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events to return
        
        Returns:
            List of events
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_subscribers(self, event_type: Optional[str] = None) -> Dict[str, int]:
        """
        Get subscriber counts.
        
        Args:
            event_type: Optional event type filter
        
        Returns:
            Dictionary mapping event types to subscriber counts
        """
        if event_type:
            return {event_type: len(self._subscribers.get(event_type, []))}
        
        return {
            event_type: len(handlers)
            for event_type, handlers in self._subscribers.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            'events_published': self._stats['events_published'],
            'events_delivered': self._stats['events_delivered'],
            'events_failed': self._stats['events_failed'],
            'total_subscribers': sum(len(handlers) for handlers in self._subscribers.values()),
            'subscribers_by_type': self.get_subscribers(),
            'history_size': len(self._event_history)
        }


# Global event bus service instance
_event_bus_service: Optional[EventBusService] = None


def get_event_bus_service() -> EventBusService:
    """Get or create event bus service instance"""
    global _event_bus_service
    if _event_bus_service is None:
        _event_bus_service = EventBusService()
    return _event_bus_service

