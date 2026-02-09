"""
Event Bus
=========

Simple event bus for pub/sub pattern.
"""

import asyncio
import logging
from typing import Callable, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data"""
    event_type: str
    data: Any
    timestamp: datetime = None
    source: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """
    Simple event bus for pub/sub pattern.
    
    Features:
    - Publish events
    - Subscribe to events
    - Async event handling
    - Event filtering
    """
    
    def __init__(self):
        """Initialize event bus"""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable[[Event], Any]):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Handler function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], Any]):
        """
        Unsubscribe from event type.
        
        Args:
            event_type: Event type
            handler: Handler function to remove
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type}")
    
    async def publish(self, event: Event):
        """
        Publish event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Get subscribers
        handlers = self.subscribers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"No subscribers for event type: {event.event_type}")
            return
        
        # Call all handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}", exc_info=True)
    
    def get_recent_events(self, event_type: str = None, limit: int = 10) -> List[Event]:
        """
        Get recent events.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events
            
        Returns:
            List of recent events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]


# Global event bus instance
_event_bus = None


def get_event_bus() -> EventBus:
    """Get or create event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

