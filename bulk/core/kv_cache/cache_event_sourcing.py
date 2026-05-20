"""
Event sourcing system for KV cache.

This module provides event-driven architecture capabilities,
allowing full audit trails and event replay.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import uuid


class EventType(Enum):
    """Types of cache events."""
    CACHE_GET = "cache_get"
    CACHE_SET = "cache_set"
    CACHE_DELETE = "cache_delete"
    CACHE_CLEAR = "cache_clear"
    CACHE_EVICT = "cache_evict"
    CACHE_UPDATE = "cache_update"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CONFIG_CHANGE = "config_change"
    STRATEGY_CHANGE = "strategy_change"


@dataclass
class CacheEvent:
    """A cache event."""
    event_id: str
    event_type: EventType
    timestamp: float
    key: Optional[str] = None
    value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'key': self.key,
            'value': self.value,
            'metadata': self.metadata,
            'version': self.version
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEvent':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=data['timestamp'],
            key=data.get('key'),
            value=data.get('value'),
            metadata=data.get('metadata', {}),
            version=data.get('version', 1)
        )


@dataclass
class EventStream:
    """A stream of events."""
    stream_id: str
    events: List[CacheEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_event_at: Optional[float] = None


class CacheEventStore:
    """Store for cache events."""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._streams: Dict[str, EventStream] = {}
        self._lock = threading.Lock()
        
    def append(self, event: CacheEvent, stream_id: Optional[str] = None) -> None:
        """Append an event to the store."""
        with self._lock:
            self._events.append(event)
            
            if stream_id:
                if stream_id not in self._streams:
                    self._streams[stream_id] = EventStream(stream_id=stream_id)
                stream = self._streams[stream_id]
                stream.events.append(event)
                stream.last_event_at = event.timestamp
                
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[CacheEvent]:
        """Get events matching criteria."""
        with self._lock:
            events = list(self._events)
            
            # Filter by type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
                
            # Filter by time range
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
                
            # Apply limit
            if limit:
                events = events[-limit:]
                
            return events
            
    def get_stream(self, stream_id: str) -> Optional[EventStream]:
        """Get an event stream."""
        return self._streams.get(stream_id)
        
    def get_stream_events(
        self,
        stream_id: str,
        start_version: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[CacheEvent]:
        """Get events from a stream."""
        stream = self.get_stream(stream_id)
        if not stream:
            return []
            
        events = stream.events
        
        if start_version:
            events = [e for e in events if e.version >= start_version]
            
        if limit:
            events = events[-limit:]
            
        return events
        
    def replay_events(
        self,
        events: List[CacheEvent],
        handler: Callable[[CacheEvent], None]
    ) -> None:
        """Replay events through a handler."""
        for event in events:
            handler(event)
            
    def get_event_count(self) -> int:
        """Get total number of events."""
        return len(self._events)
        
    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()
            self._streams.clear()


class CacheEventSourcing:
    """Event sourcing system for cache."""
    
    def __init__(self, cache: Any, enable_event_sourcing: bool = True):
        self.cache = cache
        self.enable_event_sourcing = enable_event_sourcing
        self.event_store = CacheEventStore()
        self._subscribers: List[Callable[[CacheEvent], None]] = []
        self._lock = threading.Lock()
        
    def _emit_event(
        self,
        event_type: EventType,
        key: Optional[str] = None,
        value: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit an event."""
        if not self.enable_event_sourcing:
            return
            
        event = CacheEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            key=key,
            value=value,
            metadata=metadata or {}
        )
        
        self.event_store.append(event)
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
                
    def subscribe(self, handler: Callable[[CacheEvent], None]) -> None:
        """Subscribe to events."""
        with self._lock:
            self._subscribers.append(handler)
            
    def unsubscribe(self, handler: Callable[[CacheEvent], None]) -> None:
        """Unsubscribe from events."""
        with self._lock:
            if handler in self._subscribers:
                self._subscribers.remove(handler)
                
    def get_event_history(
        self,
        key: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[CacheEvent]:
        """Get event history."""
        events = self.event_store.get_events(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time
        )
        
        if key:
            events = [e for e in events if e.key == key]
            
        return events
        
    def replay_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        handler: Optional[Callable[[CacheEvent], None]] = None
    ) -> None:
        """Replay event history."""
        events = self.event_store.get_events(
            start_time=start_time,
            end_time=end_time
        )
        
        if handler:
            self.event_store.replay_events(events, handler)
        else:
            # Default replay: rebuild cache state
            for event in events:
                if event.event_type == EventType.CACHE_SET:
                    if event.key and event.value is not None:
                        self.cache.put(event.key, event.value)
                elif event.event_type == EventType.CACHE_DELETE:
                    if event.key:
                        self.cache.delete(event.key)
                elif event.event_type == EventType.CACHE_CLEAR:
                    self.cache.clear()
                    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about events."""
        events = list(self.event_store._events)
        
        if not events:
            return {}
            
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        return {
            'total_events': len(events),
            'event_counts': event_counts,
            'first_event_time': events[0].timestamp if events else None,
            'last_event_time': events[-1].timestamp if events else None,
            'time_span': events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0
        }
        
    def export_events(
        self,
        filepath: str,
        format: str = 'json'
    ) -> None:
        """Export events to a file."""
        events = [event.to_dict() for event in self.event_store._events]
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(events, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def import_events(self, filepath: str, format: str = 'json') -> None:
        """Import events from a file."""
        if format == 'json':
            with open(filepath, 'r') as f:
                events_data = json.load(f)
                
            for event_data in events_data:
                event = CacheEvent.from_dict(event_data)
                self.event_store.append(event)
        else:
            raise ValueError(f"Unsupported import format: {format}")


class CacheEventProjector:
    """Project events into read models."""
    
    def __init__(self, event_store: CacheEventStore):
        self.event_store = event_store
        self._projections: Dict[str, Any] = {}
        
    def create_projection(
        self,
        name: str,
        projection_func: Callable[[List[CacheEvent]], Any]
    ) -> None:
        """Create a projection from events."""
        events = list(self.event_store._events)
        self._projections[name] = projection_func(events)
        
    def get_projection(self, name: str) -> Optional[Any]:
        """Get a projection."""
        return self._projections.get(name)
        
    def update_projection(
        self,
        name: str,
        new_events: List[CacheEvent],
        projection_func: Callable[[List[CacheEvent]], Any]
    ) -> None:
        """Update a projection with new events."""
        if name in self._projections:
            # Incremental update
            current_projection = self._projections[name]
            # Apply projection function to new events
            # This is simplified - real implementation would merge properly
            pass
        else:
            # Create new projection
            self.create_projection(name, projection_func)
















