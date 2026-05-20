"""
Event Bus Module

Publish/Subscribe pattern implementation for decoupled communication.
"""

from typing import Dict, Any, Callable, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .event import Event


class EventBus:
    """
    Event bus for publish/subscribe pattern.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus.
        
        Args:
            max_history: Maximum number of events to keep in history.
        """
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = max_history
    
    def subscribe(self, event_name: str, callback: Callable) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event.
            callback: Callback function to call when event is published.
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        self._subscribers[event_name].append(callback)
        logger.debug(f"Subscribed to event: {event_name}")
    
    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of the event.
            callback: Callback function to remove.
        """
        if event_name in self._subscribers:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)
                logger.debug(f"Unsubscribed from event: {event_name}")
    
    def publish(self, event_name: str, data: Any, source: Optional[str] = None) -> None:
        """
        Publish an event.
        
        Args:
            event_name: Name of the event.
            data: Event data.
            source: Optional source of the event.
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
        """
        Get event history.
        
        Args:
            event_name: Optional event name to filter by.
        
        Returns:
            List of events.
        """
        if event_name:
            return [e for e in self._event_history if e.name == event_name]
        return self._event_history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
        logger.debug("Event history cleared")



