"""
Event System - Publish/Subscribe pattern for decoupled communication
"""

from typing import Dict, Any, Callable, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure"""
    name: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None


class EventBus:
    """
    Event bus for publish/subscribe pattern
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
    
    def subscribe(self, event_name: str, callback: Callable) -> None:
        """
        Subscribe to an event
        
        Args:
            event_name: Name of the event
            callback: Callback function
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        self._subscribers[event_name].append(callback)
        logger.debug(f"Subscribed to event: {event_name}")
    
    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """Unsubscribe from an event"""
        if event_name in self._subscribers:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)
                logger.debug(f"Unsubscribed from event: {event_name}")
    
    def publish(self, event_name: str, data: Any, source: Optional[str] = None) -> None:
        """
        Publish an event
        
        Args:
            event_name: Name of the event
            data: Event data
            source: Source of the event
        """
        event = Event(
            name=event_name,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_name}: {str(e)}")
        
        logger.debug(f"Published event: {event_name}")
    
    def get_history(self, event_name: Optional[str] = None) -> List[Event]:
        """Get event history"""
        if event_name:
            return [e for e in self._event_history if e.name == event_name]
        return self._event_history.copy()
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()


# Global event bus
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get global event bus"""
    return _event_bus


def subscribe(event_name: str, callback: Callable) -> None:
    """Subscribe to event"""
    _event_bus.subscribe(event_name, callback)


def publish(event_name: str, data: Any, source: Optional[str] = None) -> None:
    """Publish event"""
    _event_bus.publish(event_name, data, source)









