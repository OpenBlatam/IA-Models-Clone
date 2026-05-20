"""
Event Bus System
===============

Advanced event bus for domain events with:
- Async event handling
- Event filtering and routing
- Event persistence
- Dead letter queue
- Event replay
- Performance monitoring
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class EventMetadata:
    """Event metadata"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainEvent:
    """Base domain event"""
    event_type: str
    data: Dict[str, Any]
    metadata: EventMetadata = field(default_factory=EventMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "metadata": {
                "event_id": self.metadata.event_id,
                "timestamp": self.metadata.timestamp.isoformat(),
                "source": self.metadata.source,
                "correlation_id": self.metadata.correlation_id,
                "causation_id": self.metadata.causation_id,
                "priority": self.metadata.priority.value,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "tags": list(self.metadata.tags),
                "properties": self.metadata.properties
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainEvent:
        """Create from dictionary"""
        metadata_data = data.get("metadata", {})
        metadata = EventMetadata(
            event_id=metadata_data.get("event_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(metadata_data.get("timestamp", datetime.utcnow().isoformat())),
            source=metadata_data.get("source", ""),
            correlation_id=metadata_data.get("correlation_id"),
            causation_id=metadata_data.get("causation_id"),
            priority=EventPriority(metadata_data.get("priority", EventPriority.NORMAL.value)),
            retry_count=metadata_data.get("retry_count", 0),
            max_retries=metadata_data.get("max_retries", 3),
            tags=set(metadata_data.get("tags", [])),
            properties=metadata_data.get("properties", {})
        )
        
        return cls(
            event_type=data["event_type"],
            data=data["data"],
            metadata=metadata
        )


class EventHandler(ABC):
    """Abstract event handler"""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        pass


class EventFilter(ABC):
    """Abstract event filter"""
    
    @abstractmethod
    def should_handle(self, event: DomainEvent) -> bool:
        """Check if event should be handled"""
        pass


class EventStore(ABC):
    """Abstract event store"""
    
    @abstractmethod
    async def save(self, event: DomainEvent) -> None:
        """Save event to store"""
        pass
    
    @abstractmethod
    async def get_events(
        self, 
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DomainEvent]:
        """Get events from store"""
        pass
    
    @abstractmethod
    async def get_event_by_id(self, event_id: str) -> Optional[DomainEvent]:
        """Get event by ID"""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store implementation"""
    
    def __init__(self, max_events: int = 10000):
        self._events: List[DomainEvent] = []
        self._events_by_id: Dict[str, DomainEvent] = {}
        self._events_by_type: Dict[str, List[DomainEvent]] = defaultdict(list)
        self._max_events = max_events
        self._lock = asyncio.Lock()
    
    async def save(self, event: DomainEvent) -> None:
        """Save event to store"""
        async with self._lock:
            self._events.append(event)
            self._events_by_id[event.metadata.event_id] = event
            self._events_by_type[event.event_type].append(event)
            
            # Maintain max events limit
            if len(self._events) > self._max_events:
                old_event = self._events.pop(0)
                del self._events_by_id[old_event.metadata.event_id]
                self._events_by_type[old_event.event_type].remove(old_event)
    
    async def get_events(
        self, 
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DomainEvent]:
        """Get events from store"""
        async with self._lock:
            events = self._events
            
            # Filter by event type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Filter by date range
            if start_date:
                events = [e for e in events if e.metadata.timestamp >= start_date]
            
            if end_date:
                events = [e for e in events if e.metadata.timestamp <= end_date]
            
            # Apply pagination
            return events[offset:offset + limit]
    
    async def get_event_by_id(self, event_id: str) -> Optional[DomainEvent]:
        """Get event by ID"""
        async with self._lock:
            return self._events_by_id.get(event_id)


class EventBus:
    """
    Advanced event bus for domain events
    
    Supports async event handling, filtering, persistence, and monitoring.
    """
    
    def __init__(self, event_store: Optional[EventStore] = None):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._filters: List[EventFilter] = []
        self._event_store = event_store or InMemoryEventStore()
        self._dead_letter_queue: deque = deque(maxlen=1000)
        self._metrics: Dict[str, Any] = {
            "events_published": 0,
            "events_handled": 0,
            "events_failed": 0,
            "handlers_registered": 0
        }
        self._lock = asyncio.Lock()
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe handler to event type"""
        self._handlers[event_type].append(handler)
        self._metrics["handlers_registered"] += 1
        logger.info(f"Handler {handler.__class__.__name__} subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe handler from event type"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            self._metrics["handlers_registered"] -= 1
            logger.info(f"Handler {handler.__class__.__name__} unsubscribed from {event_type}")
    
    def add_filter(self, filter_obj: EventFilter) -> None:
        """Add event filter"""
        self._filters.append(filter_obj)
        logger.info(f"Added event filter {filter_obj.__class__.__name__}")
    
    def remove_filter(self, filter_obj: EventFilter) -> None:
        """Remove event filter"""
        if filter_obj in self._filters:
            self._filters.remove(filter_obj)
            logger.info(f"Removed event filter {filter_obj.__class__.__name__}")
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        try:
            # Save to event store
            await self._event_store.save(event)
            
            # Apply filters
            if not self._should_handle_event(event):
                return
            
            # Get handlers for event type
            handlers = self._handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type}")
                return
            
            # Handle event with all handlers
            tasks = []
            for handler in handlers:
                if handler.can_handle(event.event_type):
                    task = asyncio.create_task(self._handle_event(handler, event))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self._metrics["events_published"] += 1
            logger.debug(f"Published event {event.event_type} with ID {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error publishing event {event.event_type}: {e}")
            self._metrics["events_failed"] += 1
            await self._add_to_dead_letter_queue(event, str(e))
    
    async def _handle_event(self, handler: EventHandler, event: DomainEvent) -> None:
        """Handle event with specific handler"""
        try:
            await handler.handle(event)
            self._metrics["events_handled"] += 1
            logger.debug(f"Handler {handler.__class__.__name__} handled event {event.event_type}")
            
        except Exception as e:
            logger.error(f"Handler {handler.__class__.__name__} failed to handle event {event.event_type}: {e}")
            self._metrics["events_failed"] += 1
            
            # Retry logic
            if event.metadata.retry_count < event.metadata.max_retries:
                event.metadata.retry_count += 1
                await asyncio.sleep(2 ** event.metadata.retry_count)  # Exponential backoff
                await self._handle_event(handler, event)
            else:
                await self._add_to_dead_letter_queue(event, str(e))
    
    def _should_handle_event(self, event: DomainEvent) -> bool:
        """Check if event should be handled based on filters"""
        for filter_obj in self._filters:
            if not filter_obj.should_handle(event):
                return False
        return True
    
    async def _add_to_dead_letter_queue(self, event: DomainEvent, error: str) -> None:
        """Add failed event to dead letter queue"""
        dead_letter_event = {
            "original_event": event.to_dict(),
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }
        self._dead_letter_queue.append(dead_letter_event)
        logger.error(f"Added event {event.metadata.event_id} to dead letter queue: {error}")
    
    async def replay_events(
        self, 
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> None:
        """Replay events from event store"""
        try:
            events = await self._event_store.get_events(
                event_type=event_type,
                start_date=start_date,
                end_date=end_date
            )
            
            for event in events:
                await self.publish(event)
            
            logger.info(f"Replayed {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error replaying events: {e}")
    
    async def get_dead_letter_events(self) -> List[Dict[str, Any]]:
        """Get events from dead letter queue"""
        return list(self._dead_letter_queue)
    
    async def retry_dead_letter_event(self, event_id: str) -> bool:
        """Retry event from dead letter queue"""
        for i, dead_letter_event in enumerate(self._dead_letter_queue):
            if dead_letter_event["original_event"]["metadata"]["event_id"] == event_id:
                # Remove from dead letter queue
                self._dead_letter_queue.remove(dead_letter_event)
                
                # Replay event
                event = DomainEvent.from_dict(dead_letter_event["original_event"])
                await self.publish(event)
                
                logger.info(f"Retried dead letter event {event_id}")
                return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            **self._metrics,
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "handlers_by_event_type": {
                event_type: len(handlers) 
                for event_type, handlers in self._handlers.items()
            },
            "filters_count": len(self._filters)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        return {
            "metrics": self.get_metrics(),
            "event_types": list(self._handlers.keys()),
            "handlers": [
                {
                    "event_type": event_type,
                    "handler_count": len(handlers),
                    "handler_names": [h.__class__.__name__ for h in handlers]
                }
                for event_type, handlers in self._handlers.items()
            ]
        }


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Decorator for event handlers
def event_handler(event_type: str):
    """Decorator for event handlers"""
    def decorator(cls):
        class EventHandlerWrapper(EventHandler):
            def __init__(self, *args, **kwargs):
                self._instance = cls(*args, **kwargs)
            
            async def handle(self, event: DomainEvent) -> None:
                await self._instance.handle(event)
            
            def can_handle(self, event_type: str) -> bool:
                return event_type == event_type
        
        # Register with event bus
        event_bus = get_event_bus()
        handler = EventHandlerWrapper()
        event_bus.subscribe(event_type, handler)
        
        return cls
    return decorator




