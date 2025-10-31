#!/usr/bin/env python3
"""
Event-Driven Architecture System

Advanced event system with:
- Event publishing and subscription
- Message queues and event streaming
- Event sourcing and CQRS
- Event replay and recovery
- Event correlation and tracking
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Type
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import weakref

logger = structlog.get_logger("event_system")

# =============================================================================
# EVENT SYSTEM MODELS
# =============================================================================

class EventType(Enum):
    """Event types enumeration."""
    VIDEO_PROCESSED = "video_processed"
    VIDEO_FAILED = "video_failed"
    BATCH_COMPLETED = "batch_completed"
    BATCH_FAILED = "batch_failed"
    VIRAL_GENERATED = "viral_generated"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PAYMENT_PROCESSED = "payment_processed"
    SYSTEM_HEALTH = "system_health"
    CUSTOM = "custom"

class EventPriority(Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"

@dataclass
class Event:
    """Base event structure."""
    event_id: str
    event_type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    version: int = 1
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data["metadata"],
            priority=EventPriority(data.get("priority", "normal")),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            version=data.get("version", 1)
        )

@dataclass
class EventHandler:
    """Event handler definition."""
    handler_id: str
    event_type: EventType
    handler_func: Callable[[Event], Awaitable[None]]
    priority: int = 0
    retry_count: int = 3
    timeout: int = 30
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handler_id": self.handler_id,
            "event_type": self.event_type.value,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "enabled": self.enabled
        }

@dataclass
class EventSubscription:
    """Event subscription definition."""
    subscription_id: str
    subscriber: str
    event_types: List[EventType]
    filter_conditions: Dict[str, Any]
    handler: EventHandler
    created_at: datetime
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subscription_id": self.subscription_id,
            "subscriber": self.subscriber,
            "event_types": [et.value for et in self.event_types],
            "filter_conditions": self.filter_conditions,
            "handler": self.handler.to_dict(),
            "created_at": self.created_at.isoformat(),
            "active": self.active
        }

# =============================================================================
# EVENT BUS
# =============================================================================

class EventBus:
    """Advanced event bus with pub/sub capabilities."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_history: deque = deque(maxlen=10000)
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'handlers_executed': 0,
            'queue_size': 0,
            'processing_time': 0.0
        }
        
        # Event correlation tracking
        self.correlation_tracker: Dict[str, List[str]] = defaultdict(list)
        
        # Dead letter queue
        self.dead_letter_queue: deque = deque(maxlen=1000)
    
    async def start(self) -> None:
        """Start the event bus."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        try:
            # Add to queue
            await self.event_queue.put(event)
            
            # Add to history
            self.event_history.append(event)
            
            # Track correlation
            if event.correlation_id:
                self.correlation_tracker[event.correlation_id].append(event.event_id)
            
            # Update statistics
            self.stats['events_published'] += 1
            self.stats['queue_size'] = self.event_queue.qsize()
            
            logger.debug(
                "Event published",
                event_id=event.event_id,
                event_type=event.event_type.value,
                source=event.source,
                correlation_id=event.correlation_id
            )
            
        except asyncio.QueueFull:
            logger.error("Event queue is full, dropping event", event_id=event.event_id)
            raise RuntimeError("Event queue is full")
    
    async def subscribe(self, subscription: EventSubscription) -> str:
        """Subscribe to events."""
        self.subscriptions[subscription.subscription_id] = subscription
        
        # Add handler to event type mapping
        for event_type in subscription.event_types:
            self.handlers[event_type].append(subscription.handler)
            # Sort by priority (higher priority first)
            self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
        
        logger.info(
            "Event subscription created",
            subscription_id=subscription.subscription_id,
            subscriber=subscription.subscriber,
            event_types=[et.value for et in subscription.event_types]
        )
        
        return subscription.subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        if subscription_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove handler from event type mapping
        for event_type in subscription.event_types:
            if subscription.handler in self.handlers[event_type]:
                self.handlers[event_type].remove(subscription.handler)
        
        # Remove subscription
        del self.subscriptions[subscription_id]
        
        logger.info("Event subscription removed", subscription_id=subscription_id)
        return True
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.is_running:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Process event
                await self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
                # Update statistics
                self.stats['events_processed'] += 1
                self.stats['queue_size'] = self.event_queue.qsize()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing event", error=str(e))
                self.stats['events_failed'] += 1
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        start_time = time.time()
        
        try:
            # Get handlers for this event type
            handlers = self.handlers.get(event.event_type, [])
            
            if not handlers:
                logger.debug("No handlers for event type", event_type=event.event_type.value)
                return
            
            # Execute handlers
            for handler in handlers:
                if not handler.enabled:
                    continue
                
                try:
                    await self._execute_handler(handler, event)
                    self.stats['handlers_executed'] += 1
                    
                except Exception as e:
                    logger.error(
                        "Handler execution failed",
                        handler_id=handler.handler_id,
                        event_id=event.event_id,
                        error=str(e)
                    )
                    
                    # Add to dead letter queue if retries exhausted
                    if handler.retry_count <= 0:
                        self.dead_letter_queue.append({
                            'event': event.to_dict(),
                            'handler': handler.to_dict(),
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat()
                        })
        
        finally:
            # Update processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
    
    async def _execute_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute a single event handler."""
        retry_count = handler.retry_count
        
        while retry_count >= 0:
            try:
                # Execute handler with timeout
                await asyncio.wait_for(
                    handler.handler_func(event),
                    timeout=handler.timeout
                )
                return  # Success
                
            except asyncio.TimeoutError:
                logger.warning(
                    "Handler timeout",
                    handler_id=handler.handler_id,
                    event_id=event.event_id,
                    timeout=handler.timeout
                )
                retry_count -= 1
                
            except Exception as e:
                logger.error(
                    "Handler error",
                    handler_id=handler.handler_id,
                    event_id=event.event_id,
                    error=str(e),
                    retry_count=retry_count
                )
                retry_count -= 1
            
            if retry_count >= 0:
                # Wait before retry
                await asyncio.sleep(1)
        
        # All retries exhausted
        raise Exception(f"Handler {handler.handler_id} failed after {handler.retry_count} retries")
    
    def get_event_correlation(self, correlation_id: str) -> List[str]:
        """Get all events for a correlation ID."""
        return self.correlation_tracker.get(correlation_id, [])
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Get dead letter queue."""
        return list(self.dead_letter_queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self.stats,
            'active_subscriptions': len(self.subscriptions),
            'total_handlers': sum(len(handlers) for handlers in self.handlers.values()),
            'dead_letter_count': len(self.dead_letter_queue),
            'correlation_tracking': len(self.correlation_tracker)
        }

# =============================================================================
# EVENT STORE
# =============================================================================

class EventStore:
    """Event store for event sourcing."""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self.events: Dict[str, List[Event]] = defaultdict(list)
        self.event_index: Dict[str, str] = {}  # event_id -> stream_id
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.snapshot_interval = 100  # Create snapshot every N events
    
    async def append_event(self, stream_id: str, event: Event) -> None:
        """Append event to stream."""
        self.events[stream_id].append(event)
        self.event_index[event.event_id] = stream_id
        
        # Create snapshot if needed
        if len(self.events[stream_id]) % self.snapshot_interval == 0:
            await self._create_snapshot(stream_id)
        
        logger.debug(
            "Event appended to stream",
            stream_id=stream_id,
            event_id=event.event_id,
            event_type=event.event_type.value
        )
    
    async def get_events(self, stream_id: str, from_version: int = 0) -> List[Event]:
        """Get events from stream."""
        events = self.events.get(stream_id, [])
        return events[from_version:]
    
    async def get_event(self, event_id: str) -> Optional[Event]:
        """Get specific event by ID."""
        stream_id = self.event_index.get(event_id)
        if not stream_id:
            return None
        
        events = self.events.get(stream_id, [])
        for event in events:
            if event.event_id == event_id:
                return event
        
        return None
    
    async def get_stream_snapshot(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get stream snapshot."""
        return self.snapshots.get(stream_id)
    
    async def _create_snapshot(self, stream_id: str) -> None:
        """Create snapshot for stream."""
        events = self.events.get(stream_id, [])
        if not events:
            return
        
        # Create snapshot from last event
        last_event = events[-1]
        snapshot = {
            'stream_id': stream_id,
            'version': len(events),
            'last_event_id': last_event.event_id,
            'last_event_type': last_event.event_type.value,
            'timestamp': last_event.timestamp.isoformat(),
            'data': last_event.data
        }
        
        self.snapshots[stream_id] = snapshot
        
        logger.debug("Snapshot created", stream_id=stream_id, version=snapshot['version'])
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            'total_streams': len(self.events),
            'total_events': sum(len(events) for events in self.events.values()),
            'total_snapshots': len(self.snapshots),
            'streams': {
                stream_id: {
                    'event_count': len(events),
                    'has_snapshot': stream_id in self.snapshots
                }
                for stream_id, events in self.events.items()
            }
        }

# =============================================================================
# EVENT REPLAYER
# =============================================================================

class EventReplayer:
    """Event replayer for event sourcing and recovery."""
    
    def __init__(self, event_store: EventStore, event_bus: EventBus):
        self.event_store = event_store
        self.event_bus = event_bus
        self.replay_stats = {
            'events_replayed': 0,
            'replay_duration': 0.0,
            'last_replay': None
        }
    
    async def replay_events(self, stream_id: str, from_version: int = 0, 
                          to_version: Optional[int] = None) -> int:
        """Replay events from stream."""
        start_time = time.time()
        
        try:
            events = await self.event_store.get_events(stream_id, from_version)
            
            if to_version:
                events = events[:to_version - from_version]
            
            replayed_count = 0
            
            for event in events:
                # Create new event with replay flag
                replay_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=event.event_type,
                    source=f"replay:{event.source}",
                    timestamp=datetime.utcnow(),
                    data=event.data,
                    metadata={**event.metadata, 'replay': True, 'original_event_id': event.event_id},
                    priority=event.priority,
                    correlation_id=event.correlation_id,
                    causation_id=event.causation_id,
                    version=event.version
                )
                
                # Publish replayed event
                await self.event_bus.publish(replay_event)
                replayed_count += 1
            
            # Update statistics
            self.replay_stats['events_replayed'] += replayed_count
            self.replay_stats['replay_duration'] = time.time() - start_time
            self.replay_stats['last_replay'] = datetime.utcnow().isoformat()
            
            logger.info(
                "Events replayed",
                stream_id=stream_id,
                replayed_count=replayed_count,
                duration=self.replay_stats['replay_duration']
            )
            
            return replayed_count
            
        except Exception as e:
            logger.error("Event replay failed", stream_id=stream_id, error=str(e))
            raise
    
    async def replay_all_events(self, from_timestamp: Optional[datetime] = None) -> int:
        """Replay all events from all streams."""
        total_replayed = 0
        
        for stream_id in self.event_store.events.keys():
            try:
                replayed_count = await self.replay_events(stream_id)
                total_replayed += replayed_count
            except Exception as e:
                logger.error("Stream replay failed", stream_id=stream_id, error=str(e))
        
        return total_replayed
    
    def get_replay_stats(self) -> Dict[str, Any]:
        """Get replay statistics."""
        return self.replay_stats.copy()

# =============================================================================
# EVENT CORRELATION
# =============================================================================

class EventCorrelation:
    """Event correlation and tracking system."""
    
    def __init__(self):
        self.correlations: Dict[str, Dict[str, Any]] = {}
        self.event_flows: Dict[str, List[str]] = defaultdict(list)
    
    def track_correlation(self, correlation_id: str, event: Event) -> None:
        """Track event correlation."""
        if correlation_id not in self.correlations:
            self.correlations[correlation_id] = {
                'correlation_id': correlation_id,
                'events': [],
                'start_time': event.timestamp,
                'end_time': None,
                'status': 'active'
            }
        
        correlation = self.correlations[correlation_id]
        correlation['events'].append({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'source': event.source,
            'timestamp': event.timestamp.isoformat(),
            'causation_id': event.causation_id
        })
        
        # Update end time
        if event.timestamp > correlation['start_time']:
            correlation['end_time'] = event.timestamp
        
        # Track event flow
        if event.causation_id:
            self.event_flows[event.causation_id].append(event.event_id)
    
    def get_correlation(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get correlation details."""
        return self.correlations.get(correlation_id)
    
    def get_event_flow(self, event_id: str) -> List[str]:
        """Get event flow for an event."""
        return self.event_flows.get(event_id, [])
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        active_correlations = len([c for c in self.correlations.values() if c['status'] == 'active'])
        completed_correlations = len([c for c in self.correlations.values() if c['status'] == 'completed'])
        
        return {
            'total_correlations': len(self.correlations),
            'active_correlations': active_correlations,
            'completed_correlations': completed_correlations,
            'total_event_flows': len(self.event_flows)
        }

# =============================================================================
# GLOBAL EVENT SYSTEM INSTANCES
# =============================================================================

# Global event system instances
event_bus = EventBus()
event_store = EventStore()
event_replayer = EventReplayer(event_store, event_bus)
event_correlation = EventCorrelation()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EventType',
    'EventPriority',
    'EventStatus',
    'Event',
    'EventHandler',
    'EventSubscription',
    'EventBus',
    'EventStore',
    'EventReplayer',
    'EventCorrelation',
    'event_bus',
    'event_store',
    'event_replayer',
    'event_correlation'
]





























