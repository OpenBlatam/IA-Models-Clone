"""
Observer Pattern - Event System
================================

Provides observer pattern for event-driven architecture.
"""

import logging
from typing import Dict, Any, Callable, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Observer(ABC):
    """Abstract observer interface."""
    
    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """
        Handle event update.
        
        Args:
            event: Event name
            data: Event data
        """
        pass


class EventPublisher:
    """
    Event publisher for observer pattern.
    
    Allows components to subscribe to events and receive notifications.
    """
    
    def __init__(self):
        """Initialize event publisher."""
        self._observers: Dict[str, List[Observer]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event: str, observer: Observer) -> None:
        """
        Subscribe observer to event.
        
        Args:
            event: Event name
            observer: Observer instance
        """
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(observer)
        logger.debug(f"Observer subscribed to event: {event}")
    
    def subscribe_callback(self, event: str, callback: Callable) -> None:
        """
        Subscribe callback function to event.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
        logger.debug(f"Callback subscribed to event: {event}")
    
    def publish(self, event: str, data: Dict[str, Any]) -> None:
        """
        Publish event to all subscribers.
        
        Args:
            event: Event name
            data: Event data
        """
        # Notify observers
        if event in self._observers:
            for observer in self._observers[event]:
                try:
                    observer.update(event, data)
                except Exception as e:
                    logger.error(f"Error in observer {observer}: {e}")
        
        # Call callbacks
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(event, data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
    
    def unsubscribe(self, event: str, observer: Observer) -> None:
        """
        Unsubscribe observer from event.
        
        Args:
            event: Event name
            observer: Observer instance
        """
        if event in self._observers:
            self._observers[event].remove(observer)
    
    def clear(self, event: Optional[str] = None) -> None:
        """
        Clear subscribers for event or all events.
        
        Args:
            event: Event name (None for all)
        """
        if event is None:
            self._observers.clear()
            self._callbacks.clear()
        else:
            self._observers.pop(event, None)
            self._callbacks.pop(event, None)


class TrainingObserver(Observer):
    """Observer for training events."""
    
    def __init__(self, tracker: Optional[Any] = None):
        """
        Initialize training observer.
        
        Args:
            tracker: Experiment tracker
        """
        self.tracker = tracker
    
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """Handle training events."""
        if event == 'epoch_end':
            if self.tracker:
                self.tracker.log_metrics(data.get('metrics', {}), data.get('epoch', 0))
        elif event == 'training_end':
            logger.info(f"Training completed: {data}")


class LoggingObserver(Observer):
    """Observer for logging events."""
    
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """Log events."""
        logger.info(f"Event [{event}]: {data}")



