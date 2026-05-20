"""
Advanced Event Bus System
=========================

Advanced event bus with pub/sub, filtering, and event history.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Pattern
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class Event:
    """Event definition."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority.value,
            "metadata": self.metadata
        }


@dataclass
class EventHandler:
    """Event handler definition."""
    handler_id: str
    handler: Callable
    event_types: List[str]
    priority: int = 0
    filter: Optional[Callable] = None
    enabled: bool = True


class AdvancedEventBus:
    """Advanced event bus with pub/sub and filtering."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize advanced event bus.
        
        Args:
            max_history: Maximum event history size
        """
        self.max_history = max_history
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.history: List[Event] = []
        self.subscriptions: Dict[str, List[EventHandler]] = {}
        self.lock = asyncio.Lock()
    
    def subscribe(
        self,
        handler_id: str,
        handler: Callable,
        event_types: List[str],
        priority: int = 0,
        filter_func: Optional[Callable] = None
    ):
        """
        Subscribe to events.
        
        Args:
            handler_id: Handler ID
            handler: Handler function
            event_types: List of event types to subscribe to
            priority: Handler priority
            filter_func: Optional filter function
        """
        event_handler = EventHandler(
            handler_id=handler_id,
            handler=handler,
            event_types=event_types,
            priority=priority,
            filter=filter_func
        )
        
        for event_type in event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(event_handler)
            # Sort by priority
            self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
        
        self.subscriptions[handler_id] = event_handler
        logger.info(f"Subscribed handler {handler_id} to events: {event_types}")
    
    def unsubscribe(self, handler_id: str):
        """
        Unsubscribe handler.
        
        Args:
            handler_id: Handler ID
        """
        if handler_id not in self.subscriptions:
            return
        
        handler = self.subscriptions[handler_id]
        
        # Remove from handlers
        for event_type in handler.event_types:
            if event_type in self.handlers:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type]
                    if h.handler_id != handler_id
                ]
        
        del self.subscriptions[handler_id]
        logger.info(f"Unsubscribed handler {handler_id}")
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Publish event.
        
        Args:
            event_type: Event type
            data: Event data
            source: Optional source
            priority: Event priority
            metadata: Optional metadata
        """
        event = Event(
            type=event_type,
            data=data,
            source=source,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to history
        async with self.lock:
            self.history.append(event)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
        
        # Get handlers for this event type
        handlers = self.handlers.get(event_type, [])
        
        # Also check for wildcard handlers
        wildcard_handlers = []
        for event_type_pattern, type_handlers in self.handlers.items():
            if '*' in event_type_pattern:
                pattern = event_type_pattern.replace('*', '.*')
                if re.match(pattern, event_type):
                    wildcard_handlers.extend(type_handlers)
        
        all_handlers = handlers + wildcard_handlers
        
        # Execute handlers
        for handler in all_handlers:
            if not handler.enabled:
                continue
            
            # Apply filter if any
            if handler.filter:
                try:
                    if not handler.filter(event):
                        continue
                except Exception as e:
                    logger.warning(f"Filter function failed for {handler.handler_id}: {e}")
                    continue
            
            # Execute handler
            try:
                if asyncio.iscoroutinefunction(handler.handler):
                    await handler.handler(event)
                else:
                    handler.handler(event)
            except Exception as e:
                logger.error(f"Event handler {handler.handler_id} failed: {e}", exc_info=True)
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Optional event type filter
            since: Optional timestamp filter
            limit: Optional limit
            
        Returns:
            List of events
        """
        events = self.history.copy()
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Statistics dictionary
        """
        by_type = {}
        for event in self.history:
            event_type = event.type
            by_type[event_type] = by_type.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.history),
            "by_type": by_type,
            "handlers": len(self.subscriptions),
            "event_types": len(self.handlers)
        }



