"""
Event Bus
=========

Event-driven architecture support.
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Event:
    """Event data."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    source: str = "system"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


class EventBus:
    """
    Event bus for pub/sub pattern.
    
    Features:
    - Event publishing
    - Event subscription
    - Filtering
    - Async handlers
    - Event history
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable] = []
        self._history: List[Event] = []
        self.max_history = max_history
        self._lock = asyncio.Lock()
    
    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: Callable
    ):
        """
        Subscribe to events.
        
        Args:
            event_type: Event type to subscribe to (None for all events)
            handler: Async handler function
        """
        if event_type is None:
            self._wildcard_subscribers.append(handler)
            logger.debug(f"Subscribed to all events: {handler.__name__}")
        else:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Subscribed to {event_type.value}: {handler.__name__}")
    
    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: Callable
    ):
        """
        Unsubscribe from events.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type is None:
            if handler in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(handler)
        else:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """
        Publish an event.
        
        Args:
            event: Event to publish
        """
        # Add to history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
        
        # Get subscribers
        handlers = []
        
        # Type-specific subscribers
        if event.event_type in self._subscribers:
            handlers.extend(self._subscribers[event.event_type])
        
        # Wildcard subscribers
        handlers.extend(self._wildcard_subscribers)
        
        # Execute handlers
        if handlers:
            tasks = [self._execute_handler(handler, event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published event {event.event_type.value} to {len(handlers)} handlers")
    
    async def _execute_handler(self, handler: Callable, event: Event):
        """Execute event handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Error executing event handler {handler.__name__}: {e}", exc_info=True)
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        history = self._history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get number of subscribers.
        
        Args:
            event_type: Optional event type
            
        Returns:
            Number of subscribers
        """
        if event_type is None:
            return len(self._wildcard_subscribers) + sum(len(h) for h in self._subscribers.values())
        return len(self._subscribers.get(event_type, []))

