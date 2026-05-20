"""
Advanced Event System
=====================

Advanced event system with filtering, routing, and transformation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event handler priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event data structure."""
    name: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class EventHandler:
    """Event handler definition."""
    name: str
    handler: Callable[[Event], Awaitable[None]]
    event_filter: Optional[Callable[[Event], bool]] = None
    priority: EventPriority = EventPriority.NORMAL
    enabled: bool = True
    
    def __lt__(self, other):
        """Compare by priority."""
        return self.priority.value < other.priority.value


class EventRouter:
    """Event router with filtering and routing."""
    
    def __init__(self):
        """Initialize event router."""
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.middleware: List[Callable[[Event], Awaitable[Event]]] = []
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(
        self,
        event_name: str,
        handler: Callable[[Event], Awaitable[None]],
        name: Optional[str] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: EventPriority = EventPriority.NORMAL
    ):
        """
        Subscribe to an event.
        
        Args:
            event_name: Event name
            handler: Async handler function
            name: Optional handler name
            filter_func: Optional filter function
            priority: Handler priority
        """
        handler_name = name or handler.__name__
        event_handler = EventHandler(
            name=handler_name,
            handler=handler,
            event_filter=filter_func,
            priority=priority
        )
        
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        
        self.handlers[event_name].append(event_handler)
        self.handlers[event_name].sort()
        
        logger.info(f"Subscribed handler '{handler_name}' to event '{event_name}'")
    
    def add_middleware(self, middleware: Callable[[Event], Awaitable[Event]]):
        """
        Add event middleware.
        
        Args:
            middleware: Middleware function
        """
        self.middleware.append(middleware)
        logger.info(f"Added event middleware: {middleware.__name__}")
    
    async def emit(self, event: Event):
        """
        Emit an event.
        
        Args:
            event: Event to emit
        """
        # Apply middleware
        for middleware in self.middleware:
            event = await middleware(event)
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Get handlers for this event
        handlers = self.handlers.get(event.name, [])
        
        # Execute handlers
        tasks = []
        for handler_def in handlers:
            if not handler_def.enabled:
                continue
            
            # Apply filter
            if handler_def.event_filter:
                if not handler_def.event_filter(event):
                    continue
            
            # Execute handler
            tasks.append(self._execute_handler(handler_def, event))
        
        # Wait for all handlers
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_handler(self, handler_def: EventHandler, event: Event):
        """Execute event handler."""
        try:
            await handler_def.handler(event)
        except Exception as e:
            logger.error(f"Error in event handler '{handler_def.name}': {e}")
    
    def unsubscribe(self, event_name: str, handler_name: str):
        """
        Unsubscribe handler from event.
        
        Args:
            event_name: Event name
            handler_name: Handler name
        """
        if event_name in self.handlers:
            self.handlers[event_name] = [
                h for h in self.handlers[event_name]
                if h.name != handler_name
            ]
            logger.info(f"Unsubscribed handler '{handler_name}' from event '{event_name}'")
    
    def get_event_history(
        self,
        event_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_name: Optional event name filter
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        events = self.event_history
        if event_name:
            events = [e for e in events if e.name == event_name]
        return events[-limit:]
    
    def get_subscribers(self, event_name: str) -> List[str]:
        """
        Get list of subscriber names for an event.
        
        Args:
            event_name: Event name
            
        Returns:
            List of handler names
        """
        if event_name in self.handlers:
            return [h.name for h in self.handlers[event_name]]
        return []




