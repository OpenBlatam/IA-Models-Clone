"""
Events Submodule
Aggregates event system components.
"""

from typing import Callable, Any, Optional
from .event import Event
from .event_bus import EventBus

# Global event bus
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """
    Get global event bus.
    
    Returns:
        Global EventBus instance.
    """
    return _event_bus


def subscribe(event_name: str, callback: Callable) -> None:
    """
    Subscribe to event using global event bus.
    
    Args:
        event_name: Name of the event.
        callback: Callback function.
    """
    _event_bus.subscribe(event_name, callback)


def publish(event_name: str, data: Any, source: Optional[str] = None) -> None:
    """
    Publish event using global event bus.
    
    Args:
        event_name: Name of the event.
        data: Event data.
        source: Optional event source.
    """
    _event_bus.publish(event_name, data, source)


__all__ = [
    "Event",
    "EventBus",
    "get_event_bus",
    "subscribe",
    "publish",
]

