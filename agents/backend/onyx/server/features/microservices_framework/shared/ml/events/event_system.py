"""
Event System
Observer pattern for event-driven architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""
    TRAINING_STARTED = "training_started"
    TRAINING_ENDED = "training_ended"
    EPOCH_STARTED = "epoch_started"
    EPOCH_ENDED = "epoch_ended"
    BATCH_STARTED = "batch_started"
    BATCH_ENDED = "batch_ended"
    MODEL_LOADED = "model_loaded"
    MODEL_SAVED = "model_saved"
    CHECKPOINT_SAVED = "checkpoint_saved"
    METRICS_UPDATED = "metrics_updated"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Event data structure."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        import time
        if self.timestamp is None:
            self.timestamp = time.time()


class EventListener(ABC):
    """Base event listener interface."""
    
    @abstractmethod
    def on_event(self, event: Event):
        """Handle an event."""
        pass


class EventEmitter:
    """Event emitter for publishing events."""
    
    def __init__(self):
        self._listeners: Dict[EventType, List[EventListener]] = {}
        self._global_listeners: List[EventListener] = []
    
    def subscribe(self, event_type: EventType, listener: EventListener):
        """Subscribe a listener to an event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(listener)
    
    def subscribe_all(self, listener: EventListener):
        """Subscribe to all events."""
        self._global_listeners.append(listener)
    
    def unsubscribe(self, event_type: EventType, listener: EventListener):
        """Unsubscribe a listener from an event type."""
        if event_type in self._listeners:
            self._listeners[event_type].remove(listener)
    
    def emit(self, event: Event):
        """Emit an event to all subscribers."""
        # Notify specific listeners
        if event.event_type in self._listeners:
            for listener in self._listeners[event.event_type]:
                try:
                    listener.on_event(event)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
        
        # Notify global listeners
        for listener in self._global_listeners:
            try:
                listener.on_event(event)
            except Exception as e:
                logger.error(f"Error in global listener: {e}")


class LoggingEventListener(EventListener):
    """Event listener that logs events."""
    
    def on_event(self, event: Event):
        logger.info(f"Event: {event.event_type.value}, Data: {event.data}")


class MetricsEventListener(EventListener):
    """Event listener that tracks metrics."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
    
    def on_event(self, event: Event):
        if event.event_type == EventType.METRICS_UPDATED:
            self.metrics_history.append(event.data)


class EventBus:
    """Central event bus for the application."""
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._emitter = EventEmitter()
        return cls._instance
    
    @property
    def emitter(self) -> EventEmitter:
        """Get the event emitter."""
        return self._emitter
    
    def publish(self, event_type: EventType, data: Dict[str, Any]):
        """Publish an event."""
        event = Event(event_type=event_type, data=data)
        self._emitter.emit(event)
    
    def subscribe(self, event_type: EventType, listener: EventListener):
        """Subscribe to events."""
        self._emitter.subscribe(event_type, listener)
    
    def subscribe_all(self, listener: EventListener):
        """Subscribe to all events."""
        self._emitter.subscribe_all(listener)



