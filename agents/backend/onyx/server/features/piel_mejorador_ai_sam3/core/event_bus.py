"""
Event Bus for Piel Mejorador AI SAM3
====================================

Pub/Sub event system for decoupled communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None


class EventBus:
    """
    Event bus for pub/sub communication.
    
    Features:
    - Publish/subscribe pattern
    - Event filtering
    - Async handlers
    - Event history
    - Wildcard subscriptions
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._lock = asyncio.Lock()
        
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
        }
    
    def subscribe(self, event_type: str, handler: Callable):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type (supports wildcards: "task.*")
            handler: Handler function (event: Event) -> None
        """
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """
        Publish an event.
        
        Args:
            event: Event to publish
        """
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            self._stats["events_published"] += 1
        
        # Find matching subscribers
        matching_handlers = []
        
        for event_type, handlers in self._subscribers.items():
            if self._matches(event.event_type, event_type):
                matching_handlers.extend(handlers)
        
        # Execute handlers
        if matching_handlers:
            tasks = [
                self._execute_handler(handler, event)
                for handler in matching_handlers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _matches(self, event_type: str, pattern: str) -> bool:
        """
        Check if event type matches pattern.
        
        Supports wildcards:
        - "task.*" matches "task.created", "task.completed", etc.
        - "*.completed" matches any ".completed" event
        """
        if pattern == "*" or pattern == event_type:
            return True
        
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".")
        
        if pattern.startswith("*."):
            suffix = pattern[2:]
            return event_type.endswith("." + suffix)
        
        return False
    
    async def _execute_handler(self, handler: Callable, event: Event):
        """Execute event handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
            self._stats["events_delivered"] += 1
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
            self._stats["events_failed"] += 1
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "subscribers": {
                event_type: len(handlers)
                for event_type, handlers in self._subscribers.items()
            },
            "history_size": len(self._event_history),
        }




