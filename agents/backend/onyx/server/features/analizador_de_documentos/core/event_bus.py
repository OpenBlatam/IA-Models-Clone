"""
Event Bus for Document Analyzer
=================================

Advanced event-driven architecture with pub/sub pattern.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Event definition"""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """Advanced event bus with pub/sub"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 1000
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler from event type: {event_type}")
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """Publish an event"""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Get subscribers for this event type
        handlers = self.subscribers.get(event.event_type, [])
        
        # Also check wildcard subscribers
        handlers.extend(self.subscribers.get("*", []))
        
        # Execute handlers
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published event: {event.event_type}")
    
    async def publish_sync(self, event_type: str, payload: Dict[str, Any], source: Optional[str] = None):
        """Publish event synchronously"""
        event = Event(
            event_type=event_type,
            payload=payload,
            source=source
        )
        await self.publish(event)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        """Get subscribers for an event type"""
        return self.subscribers.get(event_type, []).copy()

# Global instance
event_bus = EventBus()
















