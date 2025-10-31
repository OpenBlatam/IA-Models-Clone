"""
Refactored Events System

Sistema de eventos y comunicación refactorizado para el AI History Comparison System.
Maneja eventos asíncronos, pub/sub, message queues, y comunicación distribuida.
"""

import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EventType(Enum):
    """Event type enumeration"""
    SYSTEM = "system"
    USER = "user"
    DATA = "data"
    AI = "ai"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Event status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SubscriptionType(Enum):
    """Subscription type enumeration"""
    EXACT = "exact"
    PATTERN = "pattern"
    WILDCARD = "wildcard"
    REGEX = "regex"


@dataclass
class EventMetadata:
    """Event metadata"""
    event_id: str
    event_type: EventType
    priority: EventPriority
    source: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    ttl: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Event:
    """Event with payload and metadata"""
    name: str
    payload: Any
    metadata: EventMetadata
    status: EventStatus = EventStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Subscription:
    """Event subscription"""
    subscriber_id: str
    event_pattern: str
    subscription_type: SubscriptionType
    handler: Callable
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class EventHandler:
    """Event handler with metadata"""
    handler_id: str
    handler_func: Callable
    event_types: Set[EventType]
    priority: EventPriority
    is_async: bool
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


class EventSerializer:
    """Event serialization and deserialization"""
    
    @staticmethod
    def serialize(event: Event) -> bytes:
        """Serialize event to bytes"""
        event_dict = {
            "name": event.name,
            "payload": event.payload,
            "metadata": asdict(event.metadata),
            "status": event.status.value,
            "processed_at": event.processed_at.isoformat() if event.processed_at else None,
            "error_message": event.error_message
        }
        
        # Convert datetime objects to ISO strings
        event_dict["metadata"]["timestamp"] = event.metadata.timestamp.isoformat()
        event_dict["metadata"]["created_at"] = event.metadata.created_at.isoformat()
        
        return pickle.dumps(event_dict)
    
    @staticmethod
    def deserialize(data: bytes) -> Event:
        """Deserialize bytes to event"""
        event_dict = pickle.loads(data)
        
        # Convert ISO strings back to datetime objects
        event_dict["metadata"]["timestamp"] = datetime.fromisoformat(event_dict["metadata"]["timestamp"])
        event_dict["metadata"]["created_at"] = datetime.fromisoformat(event_dict["metadata"]["created_at"])
        
        # Reconstruct metadata
        metadata = EventMetadata(**event_dict["metadata"])
        
        # Reconstruct event
        event = Event(
            name=event_dict["name"],
            payload=event_dict["payload"],
            metadata=metadata,
            status=EventStatus(event_dict["status"]),
            processed_at=datetime.fromisoformat(event_dict["processed_at"]) if event_dict["processed_at"] else None,
            error_message=event_dict["error_message"]
        )
        
        return event


class EventQueue:
    """Event queue with priority and persistence"""
    
    def __init__(self, max_size: int = 10000):
        self._queues: Dict[EventPriority, deque] = {
            priority: deque(maxlen=max_size)
            for priority in EventPriority
        }
        self._lock = asyncio.Lock()
        self._size = 0
        self._max_size = max_size
    
    async def enqueue(self, event: Event) -> bool:
        """Enqueue event"""
        async with self._lock:
            if self._size >= self._max_size:
                return False
            
            self._queues[event.metadata.priority].append(event)
            self._size += 1
            return True
    
    async def dequeue(self) -> Optional[Event]:
        """Dequeue highest priority event"""
        async with self._lock:
            # Process in priority order (highest first)
            for priority in sorted(EventPriority, key=lambda p: p.value, reverse=True):
                if self._queues[priority]:
                    event = self._queues[priority].popleft()
                    self._size -= 1
                    return event
            
            return None
    
    async def peek(self) -> Optional[Event]:
        """Peek at highest priority event without removing"""
        async with self._lock:
            for priority in sorted(EventPriority, key=lambda p: p.value, reverse=True):
                if self._queues[priority]:
                    return self._queues[priority][0]
            
            return None
    
    async def size(self) -> int:
        """Get queue size"""
        async with self._lock:
            return self._size
    
    async def clear(self) -> None:
        """Clear all queues"""
        async with self._lock:
            for queue in self._queues.values():
                queue.clear()
            self._size = 0


class EventRouter:
    """Event router with pattern matching and filtering"""
    
    def __init__(self):
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._wildcard_subscriptions: List[Subscription] = []
        self._pattern_subscriptions: List[Subscription] = []
        self._lock = asyncio.Lock()
    
    async def subscribe(self, subscription: Subscription) -> None:
        """Subscribe to events"""
        async with self._lock:
            if subscription.subscription_type == SubscriptionType.EXACT:
                self._subscriptions[subscription.event_pattern].append(subscription)
            elif subscription.subscription_type == SubscriptionType.WILDCARD:
                self._wildcard_subscriptions.append(subscription)
            elif subscription.subscription_type == SubscriptionType.PATTERN:
                self._pattern_subscriptions.append(subscription)
            
            logger.info(f"Subscribed {subscription.subscriber_id} to {subscription.event_pattern}")
    
    async def unsubscribe(self, subscriber_id: str, event_pattern: str = None) -> None:
        """Unsubscribe from events"""
        async with self._lock:
            if event_pattern:
                # Unsubscribe from specific pattern
                if event_pattern in self._subscriptions:
                    self._subscriptions[event_pattern] = [
                        sub for sub in self._subscriptions[event_pattern]
                        if sub.subscriber_id != subscriber_id
                    ]
            else:
                # Unsubscribe from all patterns
                for pattern in list(self._subscriptions.keys()):
                    self._subscriptions[pattern] = [
                        sub for sub in self._subscriptions[pattern]
                        if sub.subscriber_id != subscriber_id
                    ]
                
                self._wildcard_subscriptions = [
                    sub for sub in self._wildcard_subscriptions
                    if sub.subscriber_id != subscriber_id
                ]
                
                self._pattern_subscriptions = [
                    sub for sub in self._pattern_subscriptions
                    if sub.subscriber_id != subscriber_id
                ]
            
            logger.info(f"Unsubscribed {subscriber_id} from {event_pattern or 'all'}")
    
    async def route_event(self, event: Event) -> List[Subscription]:
        """Route event to matching subscriptions"""
        async with self._lock:
            matching_subscriptions = []
            
            # Exact match subscriptions
            if event.name in self._subscriptions:
                matching_subscriptions.extend(self._subscriptions[event.name])
            
            # Wildcard subscriptions
            for subscription in self._wildcard_subscriptions:
                if self._matches_wildcard(event.name, subscription.event_pattern):
                    matching_subscriptions.append(subscription)
            
            # Pattern subscriptions
            for subscription in self._pattern_subscriptions:
                if self._matches_pattern(event.name, subscription.event_pattern):
                    matching_subscriptions.append(subscription)
            
            # Filter by filter function if provided
            filtered_subscriptions = []
            for subscription in matching_subscriptions:
                if subscription.filter_func:
                    try:
                        if subscription.filter_func(event):
                            filtered_subscriptions.append(subscription)
                    except Exception as e:
                        logger.error(f"Error in filter function: {e}")
                else:
                    filtered_subscriptions.append(subscription)
            
            return filtered_subscriptions
    
    def _matches_wildcard(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches wildcard pattern"""
        if "*" not in pattern:
            return event_name == pattern
        
        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(event_name, pattern)
    
    def _matches_pattern(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches regex pattern"""
        import re
        try:
            return bool(re.match(pattern, event_name))
        except re.error:
            return False


class EventProcessor:
    """Event processor with retry logic and error handling"""
    
    def __init__(self, max_concurrent: int = 10):
        self._handlers: Dict[str, EventHandler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._processing_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    async def register_handler(self, handler: EventHandler) -> None:
        """Register event handler"""
        async with self._lock:
            self._handlers[handler.handler_id] = handler
            logger.info(f"Registered event handler: {handler.handler_id}")
    
    async def unregister_handler(self, handler_id: str) -> None:
        """Unregister event handler"""
        async with self._lock:
            if handler_id in self._handlers:
                del self._handlers[handler_id]
                logger.info(f"Unregistered event handler: {handler_id}")
    
    async def process_event(self, event: Event, subscription: Subscription) -> bool:
        """Process event with handler"""
        async with self._semaphore:
            try:
                # Update event status
                event.status = EventStatus.PROCESSING
                
                # Get handler
                handler = self._handlers.get(subscription.subscriber_id)
                if not handler:
                    logger.error(f"Handler not found: {subscription.subscriber_id}")
                    return False
                
                # Execute handler
                if handler.is_async:
                    if handler.timeout:
                        await asyncio.wait_for(
                            handler.handler_func(event),
                            timeout=handler.timeout
                        )
                    else:
                        await handler.handler_func(event)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler.handler_func, event)
                
                # Update event status
                event.status = EventStatus.COMPLETED
                event.processed_at = datetime.utcnow()
                
                # Update stats
                self._processing_stats[handler.handler_id]["processed"] += 1
                
                return True
                
            except asyncio.TimeoutError:
                event.status = EventStatus.FAILED
                event.error_message = "Handler timeout"
                self._processing_stats[subscription.subscriber_id]["timeouts"] += 1
                return False
                
            except Exception as e:
                event.status = EventStatus.FAILED
                event.error_message = str(e)
                self._processing_stats[subscription.subscriber_id]["errors"] += 1
                logger.error(f"Error processing event {event.name}: {e}")
                return False
    
    async def retry_event(self, event: Event) -> bool:
        """Retry failed event"""
        if event.metadata.retry_count >= event.metadata.max_retries:
            event.status = EventStatus.FAILED
            return False
        
        event.metadata.retry_count += 1
        event.status = EventStatus.RETRYING
        event.error_message = None
        
        # Find matching subscriptions and retry
        # This would be implemented based on the event router
        return True
    
    async def get_processing_stats(self) -> Dict[str, Dict[str, int]]:
        """Get processing statistics"""
        async with self._lock:
            return dict(self._processing_stats)


class RefactoredEventManager:
    """Refactored event manager with advanced features"""
    
    def __init__(self):
        self._queue = EventQueue()
        self._router = EventRouter()
        self._processor = EventProcessor()
        self._serializer = EventSerializer()
        self._event_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval: float = 300.0  # 5 minutes
        self._event_ttl: timedelta = timedelta(hours=24)
        self._callbacks: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize event manager"""
        # Start processing task
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Refactored event manager initialized")
    
    async def _processing_loop(self) -> None:
        """Event processing loop"""
        while True:
            try:
                event = await self._queue.dequeue()
                if event:
                    await self._process_event(event)
                else:
                    await asyncio.sleep(0.1)  # Small delay when no events
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired events"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _process_event(self, event: Event) -> None:
        """Process single event"""
        try:
            # Route event to subscriptions
            subscriptions = await self._router.route_event(event)
            
            if not subscriptions:
                logger.warning(f"No subscriptions found for event: {event.name}")
                return
            
            # Process event with all matching subscriptions
            tasks = []
            for subscription in subscriptions:
                if subscription.is_active:
                    task = asyncio.create_task(
                        self._processor.process_event(event, subscription)
                    )
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add to history
            self._event_history.append(event)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        except Exception as e:
            logger.error(f"Error processing event {event.name}: {e}")
    
    async def _cleanup_expired_events(self) -> None:
        """Cleanup expired events from history"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - self._event_ttl
        
        # Remove expired events from history
        while self._event_history and self._event_history[0].metadata.created_at < cutoff_time:
            self._event_history.popleft()
    
    async def publish_event(self, name: str, payload: Any, event_type: EventType = EventType.CUSTOM,
                          priority: EventPriority = EventPriority.NORMAL, source: str = "system",
                          correlation_id: str = None, parent_event_id: str = None,
                          tags: Set[str] = None, labels: Dict[str, str] = None,
                          ttl: timedelta = None) -> str:
        """Publish event"""
        event_id = str(uuid.uuid4())
        
        metadata = EventMetadata(
            event_id=event_id,
            event_type=event_type,
            priority=priority,
            source=source,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            tags=tags or set(),
            labels=labels or {},
            ttl=ttl
        )
        
        event = Event(
            name=name,
            payload=payload,
            metadata=metadata
        )
        
        # Enqueue event
        success = await self._queue.enqueue(event)
        if not success:
            raise RuntimeError("Event queue is full")
        
        logger.info(f"Published event: {name} (ID: {event_id})")
        return event_id
    
    async def subscribe(self, subscriber_id: str, event_pattern: str,
                       handler: Callable, subscription_type: SubscriptionType = SubscriptionType.EXACT,
                       priority: EventPriority = EventPriority.NORMAL,
                       filter_func: Callable = None) -> None:
        """Subscribe to events"""
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_pattern=event_pattern,
            subscription_type=subscription_type,
            handler=handler,
            priority=priority,
            filter_func=filter_func
        )
        
        await self._router.subscribe(subscription)
    
    async def unsubscribe(self, subscriber_id: str, event_pattern: str = None) -> None:
        """Unsubscribe from events"""
        await self._router.unsubscribe(subscriber_id, event_pattern)
    
    async def register_handler(self, handler_id: str, handler_func: Callable,
                              event_types: Set[EventType], priority: EventPriority = EventPriority.NORMAL,
                              timeout: float = None, max_retries: int = 3) -> None:
        """Register event handler"""
        is_async = asyncio.iscoroutinefunction(handler_func)
        
        handler = EventHandler(
            handler_id=handler_id,
            handler_func=handler_func,
            event_types=event_types,
            priority=priority,
            is_async=is_async,
            timeout=timeout,
            max_retries=max_retries
        )
        
        await self._processor.register_handler(handler)
    
    async def unregister_handler(self, handler_id: str) -> None:
        """Unregister event handler"""
        await self._processor.unregister_handler(handler_id)
    
    async def get_event_history(self, event_name: str = None, start_time: datetime = None,
                               end_time: datetime = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        events = list(self._event_history)
        
        if event_name:
            events = [e for e in events if e.name == event_name]
        
        if start_time:
            events = [e for e in events if e.metadata.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.metadata.timestamp <= end_time]
        
        return events[-limit:] if limit else events
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "queue_size": await self._queue.size(),
            "max_size": self._queue._max_size,
            "history_size": len(self._event_history),
            "processing_stats": await self._processor.get_processing_stats()
        }
    
    def add_callback(self, callback: Callable) -> None:
        """Add event callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove event callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get event manager health status"""
        return {
            "queue_size": await self._queue.size(),
            "history_size": len(self._event_history),
            "cleanup_interval": self._cleanup_interval,
            "event_ttl": self._event_ttl.total_seconds(),
            "processing_stats": await self._processor.get_processing_stats()
        }
    
    async def shutdown(self) -> None:
        """Shutdown event manager"""
        if self._processing_task:
            self._processing_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Refactored event manager shutdown")


# Global event manager
event_manager = RefactoredEventManager()


# Convenience functions
async def publish_event(name: str, payload: Any, **kwargs):
    """Publish event"""
    return await event_manager.publish_event(name, payload, **kwargs)


async def subscribe_to_event(subscriber_id: str, event_pattern: str, handler: Callable, **kwargs):
    """Subscribe to event"""
    await event_manager.subscribe(subscriber_id, event_pattern, handler, **kwargs)


async def register_event_handler(handler_id: str, handler_func: Callable, **kwargs):
    """Register event handler"""
    await event_manager.register_handler(handler_id, handler_func, **kwargs)


# Event decorators
def event_handler(event_types: Set[EventType], priority: EventPriority = EventPriority.NORMAL):
    """Event handler decorator"""
    def decorator(func):
        handler_id = f"{func.__module__}.{func.__name__}"
        asyncio.create_task(
            register_event_handler(handler_id, func, event_types=event_types, priority=priority)
        )
        return func
    return decorator


def event_publisher(event_name: str, event_type: EventType = EventType.CUSTOM):
    """Event publisher decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            await publish_event(event_name, result, event_type=event_type)
            return result
        return wrapper
    return decorator





















