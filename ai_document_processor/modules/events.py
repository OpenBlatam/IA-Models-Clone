"""
Event System - Ultra-Modular Event-Driven Architecture
====================================================

Ultra-modular event system for decoupled communication between components.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EventType(str, Enum):
    """Event type enumeration."""
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_FAILED = "document_failed"
    DOCUMENT_DELETED = "document_deleted"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    AI_CLASSIFICATION_COMPLETED = "ai_classification_completed"
    AI_TRANSFORMATION_COMPLETED = "ai_transformation_completed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    SERVICE_ERROR = "service_error"
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    METRIC_RECORDED = "metric_recorded"
    HEALTH_CHECK_FAILED = "health_check_failed"
    CUSTOM = "custom"


class EventPriority(str, Enum):
    """Event priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Event data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.CUSTOM
    priority: EventPriority = EventPriority.NORMAL
    source: str = "system"
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'source': self.source,
            'target': self.target,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'ttl': self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=EventType(data.get('type', 'custom')),
            priority=EventPriority(data.get('priority', 'normal')),
            source=data.get('source', 'system'),
            target=data.get('target'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            ttl=data.get('ttl')
        )
    
    def is_expired(self) -> bool:
        """Check if event is expired."""
        if self.ttl is None:
            return False
        
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl


class EventHandler(ABC):
    """Base event handler interface."""
    
    def __init__(self, name: str, event_types: List[EventType] = None):
        self.name = name
        self.event_types = event_types or []
        self.priority = EventPriority.NORMAL
        self.enabled = True
        self.error_count = 0
        self.last_error = None
        self.processed_count = 0
        self.last_processed = None
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Handle an event."""
        pass
    
    def can_handle(self, event: Event) -> bool:
        """Check if handler can handle the event."""
        if not self.enabled:
            return False
        
        if self.event_types and event.type not in self.event_types:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            'name': self.name,
            'event_types': [et.value for et in self.event_types],
            'priority': self.priority.value,
            'enabled': self.enabled,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'processed_count': self.processed_count,
            'last_processed': self.last_processed.isoformat() if self.last_processed else None
        }


class AsyncEventHandler(EventHandler):
    """Async event handler implementation."""
    
    def __init__(self, name: str, handler_func: Callable[[Event], Any], event_types: List[EventType] = None):
        super().__init__(name, event_types)
        self.handler_func = handler_func
    
    async def handle(self, event: Event) -> bool:
        """Handle an event asynchronously."""
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                result = await self.handler_func(event)
            else:
                result = self.handler_func(event)
            
            self.processed_count += 1
            self.last_processed = datetime.utcnow()
            
            return result is not False
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Event handler {self.name} failed: {e}")
            return False


class EventBus:
    """Ultra-modular event bus for event-driven architecture."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self._handlers: Dict[str, EventHandler] = {}
        self._handlers_by_type: Dict[EventType, List[str]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._processing_task: Optional[asyncio.Task] = None
        self._stats = {
            'events_processed': 0,
            'events_failed': 0,
            'handlers_registered': 0,
            'queue_size': 0,
            'start_time': datetime.utcnow()
        }
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the event bus."""
        if self._processing_task and not self._processing_task.done():
            return
        
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        
        logger.info("Event bus stopped")
    
    async def register_handler(self, handler: EventHandler) -> str:
        """Register an event handler."""
        async with self._lock:
            handler_id = str(uuid.uuid4())
            self._handlers[handler_id] = handler
            
            # Index by event type
            for event_type in handler.event_types:
                if event_type not in self._handlers_by_type:
                    self._handlers_by_type[event_type] = []
                self._handlers_by_type[event_type].append(handler_id)
            
            self._stats['handlers_registered'] = len(self._handlers)
            
            logger.info(f"Registered event handler: {handler.name} ({handler_id})")
            return handler_id
    
    async def unregister_handler(self, handler_id: str) -> bool:
        """Unregister an event handler."""
        async with self._lock:
            if handler_id not in self._handlers:
                return False
            
            handler = self._handlers[handler_id]
            
            # Remove from type index
            for event_type in handler.event_types:
                if event_type in self._handlers_by_type:
                    if handler_id in self._handlers_by_type[event_type]:
                        self._handlers_by_type[event_type].remove(handler_id)
            
            del self._handlers[handler_id]
            self._stats['handlers_registered'] = len(self._handlers)
            
            logger.info(f"Unregistered event handler: {handler.name} ({handler_id})")
            return True
    
    async def publish(self, event: Event) -> bool:
        """Publish an event."""
        try:
            # Check if event is expired
            if event.is_expired():
                logger.warning(f"Event {event.id} is expired, discarding")
                return False
            
            # Add to queue
            await self._event_queue.put(event)
            self._stats['queue_size'] = self._event_queue.qsize()
            
            logger.debug(f"Published event: {event.type.value} ({event.id})")
            return True
            
        except asyncio.QueueFull:
            logger.error("Event queue is full, dropping event")
            return False
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def publish_sync(self, event: Event) -> bool:
        """Publish an event and wait for processing."""
        try:
            # Find handlers for this event type
            handler_ids = self._handlers_by_type.get(event.type, [])
            
            if not handler_ids:
                logger.debug(f"No handlers for event type: {event.type.value}")
                return True
            
            # Process with all handlers
            success = True
            for handler_id in handler_ids:
                handler = self._handlers.get(handler_id)
                if handler and handler.can_handle(event):
                    try:
                        result = await handler.handle(event)
                        if not result:
                            success = False
                    except Exception as e:
                        logger.error(f"Handler {handler.name} failed: {e}")
                        success = False
            
            if success:
                self._stats['events_processed'] += 1
            else:
                self._stats['events_failed'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process event synchronously: {e}")
            self._stats['events_failed'] += 1
            return False
    
    async def _process_events(self):
        """Process events from the queue."""
        while True:
            try:
                # Get event from queue
                event = await self._event_queue.get()
                
                # Process event
                await self.publish_sync(event)
                
                # Mark task as done
                self._event_queue.task_done()
                self._stats['queue_size'] = self._event_queue.qsize()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                self._stats['events_failed'] += 1
    
    async def get_handlers_for_type(self, event_type: EventType) -> List[EventHandler]:
        """Get handlers for a specific event type."""
        handler_ids = self._handlers_by_type.get(event_type, [])
        return [self._handlers[hid] for hid in handler_ids if hid in self._handlers]
    
    async def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        handler_stats = {}
        for handler_id, handler in self._handlers.items():
            handler_stats[handler_id] = handler.get_stats()
        
        return handler_stats
    
    async def get_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            'queue_size': self._event_queue.qsize(),
            'processing_active': self._processing_task is not None and not self._processing_task.done(),
            'uptime_seconds': (datetime.utcnow() - self._stats['start_time']).total_seconds()
        }
    
    async def clear_queue(self):
        """Clear the event queue."""
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self._stats['queue_size'] = 0
        logger.info("Event queue cleared")


class EventStore:
    """Event store for persistence and replay."""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events: List[Event] = []
        self._events_by_type: Dict[EventType, List[Event]] = {}
        self._events_by_source: Dict[str, List[Event]] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, event: Event) -> bool:
        """Store an event."""
        async with self._lock:
            try:
                # Add to main store
                self._events.append(event)
                
                # Index by type
                if event.type not in self._events_by_type:
                    self._events_by_type[event.type] = []
                self._events_by_type[event.type].append(event)
                
                # Index by source
                if event.source not in self._events_by_source:
                    self._events_by_source[event.source] = []
                self._events_by_source[event.source].append(event)
                
                # Maintain max size
                if len(self._events) > self.max_events:
                    old_event = self._events.pop(0)
                    self._events_by_type[old_event.type].remove(old_event)
                    self._events_by_source[old_event.source].remove(old_event)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to store event: {e}")
                return False
    
    async def get_events_by_type(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """Get events by type."""
        events = self._events_by_type.get(event_type, [])
        return events[-limit:] if limit else events
    
    async def get_events_by_source(self, source: str, limit: int = 100) -> List[Event]:
        """Get events by source."""
        events = self._events_by_source.get(source, [])
        return events[-limit:] if limit else events
    
    async def get_events_since(self, since: datetime, limit: int = 100) -> List[Event]:
        """Get events since a timestamp."""
        events = [e for e in self._events if e.timestamp >= since]
        return events[-limit:] if limit else events
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        for event in self._events:
            if event.id == event_id:
                return event
        return None
    
    async def get_store_stats(self) -> Dict[str, Any]:
        """Get event store statistics."""
        return {
            'total_events': len(self._events),
            'events_by_type': {et.value: len(events) for et, events in self._events_by_type.items()},
            'events_by_source': {source: len(events) for source, events in self._events_by_source.items()},
            'max_events': self.max_events
        }


# Global event system instances
_event_bus: Optional[EventBus] = None
_event_store: Optional[EventStore] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def get_event_store() -> EventStore:
    """Get global event store instance."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store


# Convenience functions for common events
async def publish_document_created(document_id: str, document_data: Dict[str, Any]):
    """Publish document created event."""
    event = Event(
        type=EventType.DOCUMENT_CREATED,
        source="document_service",
        data={'document_id': document_id, 'document_data': document_data}
    )
    await get_event_bus().publish(event)


async def publish_document_processed(document_id: str, result_data: Dict[str, Any]):
    """Publish document processed event."""
    event = Event(
        type=EventType.DOCUMENT_PROCESSED,
        source="document_processor",
        data={'document_id': document_id, 'result_data': result_data}
    )
    await get_event_bus().publish(event)


async def publish_processing_failed(document_id: str, error_message: str):
    """Publish processing failed event."""
    event = Event(
        type=EventType.PROCESSING_FAILED,
        source="document_processor",
        priority=EventPriority.HIGH,
        data={'document_id': document_id, 'error_message': error_message}
    )
    await get_event_bus().publish(event)


async def publish_service_started(service_name: str, service_data: Dict[str, Any]):
    """Publish service started event."""
    event = Event(
        type=EventType.SERVICE_STARTED,
        source="service_manager",
        data={'service_name': service_name, 'service_data': service_data}
    )
    await get_event_bus().publish(event)


async def publish_health_check_failed(service_name: str, health_score: float):
    """Publish health check failed event."""
    event = Event(
        type=EventType.HEALTH_CHECK_FAILED,
        source="health_monitor",
        priority=EventPriority.HIGH,
        data={'service_name': service_name, 'health_score': health_score}
    )
    await get_event_bus().publish(event)

















