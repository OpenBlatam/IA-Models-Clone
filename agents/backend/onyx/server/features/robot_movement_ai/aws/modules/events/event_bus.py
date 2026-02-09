"""
Event Bus
=========

Event bus for event-driven communication.
"""

import logging
import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure."""
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    correlation_id: Optional[str] = None


EventHandler = Callable[[Event], Any]


class EventBus:
    """Event bus for publishing and subscribing to events."""
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._middleware: List[Callable[[Event], Event]] = []
        self._event_history: List[Event] = []
        self._max_history: int = 1000
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe to event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Unsubscribe from event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def add_middleware(self, middleware: Callable[[Event], Event]):
        """Add event middleware."""
        self._middleware.append(middleware)
    
    async def publish(self, event: Event):
        """Publish event."""
        # Apply middleware
        for middleware in self._middleware:
            event = middleware(event)
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Get handlers
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get("*", []))  # Wildcard handlers
        
        # Execute handlers
        if handlers:
            tasks = []
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    tasks.append(asyncio.to_thread(handler, event))
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published event: {event.event_type} ({event.event_id})")
    
    def get_event_history(self, event_type: Optional[str] = None) -> List[Event]:
        """Get event history."""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()















