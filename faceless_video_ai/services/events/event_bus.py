"""
Event Bus
Pub/Sub event system
"""

from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import logging
import asyncio

logger = logging.getLogger(__name__)


class Event:
    """Base event class"""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = None
        from datetime import datetime
        self.timestamp = datetime.utcnow()


class EventBus:
    """Event bus for pub/sub pattern"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable):
        """
        Subscribe to event type
        
        Args:
            event_type: Event type to subscribe to
            handler: Handler function (async or sync)
        """
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type"""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Unsubscribed from {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """
        Publish event
        
        Args:
            event_type: Event type
            data: Event data
        """
        event = Event(event_type, data)
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        handlers_all = self.subscribers.get("*", [])  # Wildcard subscribers
        
        all_handlers = handlers + handlers_all
        
        for handler in all_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {str(e)}")
        
        logger.debug(f"Published event: {event_type}")
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")


_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get event bus instance (singleton)"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

