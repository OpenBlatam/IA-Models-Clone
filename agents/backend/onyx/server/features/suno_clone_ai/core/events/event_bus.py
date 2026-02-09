"""
Event Bus

Event publishing and subscription system.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data class."""
    event_type: str
    data: Any
    timestamp: datetime = None
    event_id: str = None
    source: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())


class EventBus:
    """Event bus for pub/sub."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type
            handler: Event handler function
        """
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed to event: {event_type}")
    
    def publish(
        self,
        event_type: str,
        data: Any,
        source: Optional[str] = None
    ) -> str:
        """
        Publish event.
        
        Args:
            event_type: Event type
            data: Event data
            source: Event source
            
        Returns:
            Event ID
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source
        )
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        logger.debug(f"Published event: {event_type} ({event.event_id})")
        
        return event.event_id
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """
        Unsubscribe from event type.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Unsubscribed from event: {event_type}")


def create_event_bus() -> EventBus:
    """Create event bus."""
    return EventBus()


def publish_event(
    bus: EventBus,
    event_type: str,
    data: Any,
    **kwargs
) -> str:
    """Publish event."""
    return bus.publish(event_type, data, **kwargs)


def subscribe_event(
    bus: EventBus,
    event_type: str,
    handler: Callable
) -> None:
    """Subscribe to event."""
    bus.subscribe(event_type, handler)


def create_event(
    event_type: str,
    data: Any,
    **kwargs
) -> Event:
    """Create event."""
    return Event(event_type=event_type, data=data, **kwargs)


class EventType:
    """Common event types."""
    MODEL_TRAINED = "model.trained"
    MODEL_SAVED = "model.saved"
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"
    ERROR_OCCURRED = "error.occurred"
    HEALTH_CHECK = "health.check"



