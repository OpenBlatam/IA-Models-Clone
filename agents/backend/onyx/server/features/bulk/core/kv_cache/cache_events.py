"""
Cache event system.

Provides event-driven architecture for cache operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class CacheEventType(Enum):
    """Cache event types."""
    HIT = "hit"
    MISS = "miss"
    PUT = "put"
    GET = "get"
    EVICT = "evict"
    CLEAR = "clear"
    ERROR = "error"
    WARMUP = "warmup"
    OPTIMIZATION = "optimization"


class CacheEvent:
    """
    Cache event.
    
    Represents an event in the cache system.
    """
    
    def __init__(
        self,
        event_type: CacheEventType,
        position: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize cache event.
        
        Args:
            event_type: Type of event
            position: Optional cache position
            data: Optional event data
            timestamp: Optional timestamp (defaults to current time)
        """
        import time
        self.event_type = event_type
        self.position = position
        self.data = data or {}
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.event_type.value,
            "position": self.position,
            "data": self.data,
            "timestamp": self.timestamp
        }


class CacheEventEmitter:
    """
    Cache event emitter.
    
    Provides event-driven architecture for cache.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize event emitter.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.listeners: Dict[CacheEventType, List[Callable]] = defaultdict(list)
        self.event_history: List[CacheEvent] = []
        self.max_history = 10000
    
    def on(
        self,
        event_type: CacheEventType,
        callback: Callable[[CacheEvent], None]
    ) -> None:
        """
        Register event listener.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function
        """
        self.listeners[event_type].append(callback)
    
    def off(
        self,
        event_type: CacheEventType,
        callback: Callable[[CacheEvent], None]
    ) -> None:
        """
        Unregister event listener.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
    
    def emit(
        self,
        event_type: CacheEventType,
        position: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit event.
        
        Args:
            event_type: Type of event
            position: Optional cache position
            data: Optional event data
        """
        event = CacheEvent(event_type, position, data)
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify listeners
        for callback in self.listeners[event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event listener failed: {e}")
        
        # Also notify wildcard listeners (all events)
        for callback in self.listeners.get(CacheEventType.HIT, []):  # Placeholder
            # This would be for "all" events if we add that
            pass
    
    def get_event_history(
        self,
        event_type: Optional[CacheEventType] = None,
        limit: int = 100
    ) -> List[CacheEvent]:
        """
        Get event history.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        if event_type is None:
            return self.event_history[-limit:]
        
        filtered = [
            event for event in self.event_history
            if event.event_type == event_type
        ]
        
        return filtered[-limit:]
    
    def get_event_stats(self) -> Dict[str, Any]:
        """
        Get event statistics.
        
        Returns:
            Dictionary with event stats
        """
        event_counts: Dict[str, int] = defaultdict(int)
        
        for event in self.event_history:
            event_counts[event.event_type.value] += 1
        
        return {
            "total_events": len(self.event_history),
            "event_counts": dict(event_counts),
            "listeners": {
                event_type.value: len(callbacks)
                for event_type, callbacks in self.listeners.items()
            }
        }


class CacheEventListener:
    """
    Base class for cache event listeners.
    
    Provides helper methods for event handling.
    """
    
    def __init__(self, emitter: CacheEventEmitter):
        """
        Initialize listener.
        
        Args:
            emitter: Event emitter
        """
        self.emitter = emitter
        self._register_listeners()
    
    def _register_listeners(self) -> None:
        """Register event listeners (override in subclasses)."""
        pass
    
    def on_hit(self, event: CacheEvent) -> None:
        """Handle cache hit event."""
        pass
    
    def on_miss(self, event: CacheEvent) -> None:
        """Handle cache miss event."""
        pass
    
    def on_evict(self, event: CacheEvent) -> None:
        """Handle eviction event."""
        pass
    
    def on_error(self, event: CacheEvent) -> None:
        """Handle error event."""
        pass

