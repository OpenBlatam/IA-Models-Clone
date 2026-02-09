"""
Event Manager for Flux2 Clothing Changer
========================================

Event management and pub/sub system.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event information."""
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventManager:
    """Event management system."""
    
    def __init__(self):
        """Initialize event manager."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history: int = 10000
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type
            handler: Event handler function
        """
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """
        Unsubscribe from event type.
        
        Args:
            event_type: Event type
            handler: Event handler function
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed from event type: {event_type}")
    
    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Publish event.
        
        Args:
            event_type: Event type
            data: Event data
            source: Optional source identifier
            metadata: Optional metadata
            
        Returns:
            Created event
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            metadata=metadata or {},
        )
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Also notify wildcard subscribers
        wildcard_handlers = self.subscribers.get("*", [])
        for handler in wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in wildcard event handler: {e}")
        
        logger.debug(f"Published event: {event_type}")
        return event
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        if event_type:
            filtered = [e for e in self.event_history if e.event_type == event_type]
        else:
            filtered = self.event_history
        
        return filtered[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event manager statistics."""
        event_counts = defaultdict(int)
        for event in self.event_history:
            event_counts[event.event_type] += 1
        
        return {
            "total_events": len(self.event_history),
            "event_types": len(event_counts),
            "subscribers": {
                event_type: len(handlers)
                for event_type, handlers in self.subscribers.items()
            },
            "event_counts": dict(event_counts),
        }


