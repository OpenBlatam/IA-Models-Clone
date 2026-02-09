"""
Event Bus Module - Event-driven architecture.

Provides:
- Event publishing
- Event subscriptions
- Event filtering
- Async event handling
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import queue

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types."""
    BENCHMARK_STARTED = "benchmark.started"
    BENCHMARK_COMPLETED = "benchmark.completed"
    BENCHMARK_FAILED = "benchmark.failed"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    MODEL_REGISTERED = "model.registered"
    COST_THRESHOLD = "cost.threshold"
    ALERT_TRIGGERED = "alert.triggered"
    SYSTEM_ERROR = "system.error"


@dataclass
class Event:
    """Event definition."""
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
        }


class EventBus:
    """Event bus for pub/sub pattern."""
    
    def __init__(self, async_mode: bool = True):
        """
        Initialize event bus.
        
        Args:
            async_mode: Use async event handling
        """
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.global_subscribers: List[Callable] = []
        self.async_mode = async_mode
        self.event_queue = queue.Queue() if async_mode else None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(
        self,
        event_type: Optional[EventType] = None,
        handler: Optional[Callable] = None,
    ) -> Callable:
        """
        Subscribe to events.
        
        Args:
            event_type: Event type (None for all events)
            handler: Handler function (optional, can use as decorator)
            
        Returns:
            Handler function
        """
        def decorator(func: Callable) -> Callable:
            if event_type:
                self.subscribers[event_type].append(func)
                logger.info(f"Subscribed {func.__name__} to {event_type.value}")
            else:
                self.global_subscribers.append(func)
                logger.info(f"Subscribed {func.__name__} to all events")
            return func
        
        if handler:
            # Direct subscription
            if event_type:
                self.subscribers[event_type].append(handler)
            else:
                self.global_subscribers.append(handler)
            return handler
        
        return decorator
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Unsubscribe from events.
        
        Args:
            event_type: Event type
            handler: Handler function
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed {handler.__name__} from {event_type.value}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event.
        
        Args:
            event: Event to publish
        """
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        if self.async_mode and self.running:
            self.event_queue.put(event)
        else:
            self._handle_event(event)
    
    def _handle_event(self, event: Event) -> None:
        """Handle event synchronously."""
        # Call specific subscribers
        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
        
        # Call global subscribers
        for handler in self.global_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler {handler.__name__}: {e}")
    
    def _worker_loop(self) -> None:
        """Worker loop for async event handling."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event worker: {e}")
    
    def start(self) -> None:
        """Start async event processing."""
        if self.async_mode and not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Event bus started")
    
    def stop(self) -> None:
        """Stop async event processing."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Event bus stopped")
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


# Global event bus instance
event_bus = EventBus()












