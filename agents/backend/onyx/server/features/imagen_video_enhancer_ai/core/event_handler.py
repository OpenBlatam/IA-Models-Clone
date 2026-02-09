"""
Event Handler
=============

Base event handler system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EventPriority(Enum):
    """Event priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event definition."""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventHandler(ABC):
    """Base event handler."""
    
    def __init__(self, name: str, event_types: List[str], priority: EventPriority = EventPriority.NORMAL):
        """
        Initialize event handler.
        
        Args:
            name: Handler name
            event_types: List of event types to handle
            priority: Handler priority
        """
        self.name = name
        self.event_types = event_types
        self.priority = priority
        self.stats = {
            "total_events": 0,
            "handled_events": 0,
            "failed_events": 0
        }
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """
        Handle event.
        
        Args:
            event: Event to handle
            
        Returns:
            True if handled successfully
        """
        pass
    
    def can_handle(self, event: Event) -> bool:
        """
        Check if handler can handle event.
        
        Args:
            event: Event to check
            
        Returns:
            True if can handle
        """
        return event.type in self.event_types
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "name": self.name,
            "event_types": self.event_types,
            "priority": self.priority.value,
            "total_events": self.stats["total_events"],
            "handled_events": self.stats["handled_events"],
            "failed_events": self.stats["failed_events"],
            "success_rate": (
                self.stats["handled_events"] / self.stats["total_events"]
                if self.stats["total_events"] > 0 else 0.0
            )
        }


class EventDispatcher:
    """Event dispatcher with priority handling."""
    
    def __init__(self):
        """Initialize event dispatcher."""
        self.handlers: List[EventHandler] = []
        self.event_history: List[Event] = []
        self.max_history = 10000
        self._lock = asyncio.Lock()
    
    def register(self, handler: EventHandler):
        """
        Register event handler.
        
        Args:
            handler: Handler instance
        """
        self.handlers.append(handler)
        # Sort by priority (higher first)
        self.handlers.sort(key=lambda h: h.priority.value, reverse=True)
        logger.debug(f"Registered event handler: {handler.name}")
    
    def unregister(self, handler_name: str):
        """
        Unregister event handler.
        
        Args:
            handler_name: Handler name
        """
        self.handlers = [h for h in self.handlers if h.name != handler_name]
        logger.debug(f"Unregistered event handler: {handler_name}")
    
    async def dispatch(self, event: Event) -> Dict[str, bool]:
        """
        Dispatch event to handlers.
        
        Args:
            event: Event to dispatch
            
        Returns:
            Dictionary of handler results
        """
        async with self._lock:
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        results = {}
        
        # Find handlers that can handle this event
        handlers = [h for h in self.handlers if h.can_handle(event)]
        
        for handler in handlers:
            try:
                handler.stats["total_events"] += 1
                success = await handler.handle(event)
                results[handler.name] = success
                
                if success:
                    handler.stats["handled_events"] += 1
                else:
                    handler.stats["failed_events"] += 1
                    
            except Exception as e:
                logger.error(f"Error in handler {handler.name}: {e}")
                handler.stats["failed_events"] += 1
                results[handler.name] = False
        
        return results
    
    def get_handlers_for_type(self, event_type: str) -> List[EventHandler]:
        """Get handlers for event type."""
        return [h for h in self.handlers if event_type in h.event_types]
    
    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "total_handlers": len(self.handlers),
            "total_events": len(self.event_history),
            "handlers": [h.get_stats() for h in self.handlers]
        }




