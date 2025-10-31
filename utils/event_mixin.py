from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import EventType, EventStatus
import msgspec
import numpy as np
import time
    import pandas as pd
    from datadog import api as dd_api
    import sentry_sdk
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Event Mixin - Onyx Integration
Event handling functionality for models.
"""
try:
except ImportError:
    pd = None
try:
except ImportError:
    dd_api = None
try:
except ImportError:
    sentry_sdk = None

T = TypeVar("T")

class Event(msgspec.Struct, frozen=True, slots=True):
    id: str
    type: str
    status: str
    timestamp: str = msgspec.field(default_factory=lambda: datetime.utcnow().isoformat())
    data: dict = msgspec.field(default_factory=dict)
    metadata: dict = msgspec.field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return (self.id, self.type, self.status, self.timestamp, self.data, self.metadata)

    @staticmethod
    def batch_encode(items: List["Event"]) -> bytes:
        start = time.time()
        out = msgspec.json.encode(items)
        Event._log_metric("batch_encode", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_decode(data: bytes) -> List["Event"]:
        start = time.time()
        out = msgspec.json.decode(data, type=List[Event])
        Event._log_metric("batch_decode", len(out), time.time() - start)
        return out

    @staticmethod
    def batch_deduplicate(items: List["Event"], key: Callable[[Any], Any] = lambda x: x.id) -> List["Event"]:
        start = time.time()
        seen = set()
        out = []
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                out.append(item)
        Event._log_metric("batch_deduplicate", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_validate_unique(items: List["Event"], key: Callable[[Any], Any] = lambda x: x.id) -> None:
        seen = set()
        for item in items:
            k = key(item)
            if k in seen:
                Event._log_error("duplicate_key", k)
                raise ValueError(f"Duplicate key found: {k}")
            seen.add(k)

    @staticmethod
    def batch_filter(items: List["Event"], predicate: Callable[["Event"], bool]) -> List["Event"]:
        return [item for item in items if predicate(item)]

    @staticmethod
    def batch_map(items: List["Event"], func: Callable[["Event"], T]) -> List[T]:
        return [func(item) for item in items]

    @staticmethod
    def batch_groupby(items: List["Event"], key: Callable[["Event"], Any]) -> Dict[Any, List["Event"]]:
        groups = {}
        for item in items:
            k = key(item)
            groups.setdefault(k, []).append(item)
        return groups

    @staticmethod
    def batch_sort(items: List["Event"], key: Callable[["Event"], Any], reverse: bool = False) -> List["Event"]:
        return sorted(items, key=key, reverse=reverse)

    @staticmethod
    def batch_to_dicts(items: List["Event"]) -> List[dict]:
        return [item.__dict__ for item in items]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List["Event"]:
        return [Event(**d) for d in dicts]

    @staticmethod
    def batch_to_numpy(items: List["Event"]):
        
    """batch_to_numpy function."""
arr = np.array([item.as_tuple() for item in items], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(items: List["Event"]):
        
    """batch_to_pandas function."""
if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(Event.batch_to_dicts(items))

    @staticmethod
    def batch_to_parquet(items: List["Event"], path: str):
        
    """batch_to_parquet function."""
if pd is None:
            raise ImportError("pandas is not installed")
        Event.batch_to_pandas(items).to_parquet(path)

    @staticmethod
    def batch_from_parquet(path: str) -> List["Event"]:
        if pd is None:
            raise ImportError("pandas is not installed")
        df = pd.read_parquet(path)
        return Event.batch_from_dicts(df.to_dict(orient="records"))

    @staticmethod
    def validate_batch(items: List["Event"]) -> None:
        for item in items:
            if not item.id or not item.type or not item.status:
                Event._log_error("invalid_event", item.id)
                raise ValueError(f"Invalid Event: {item}")

    @staticmethod
    def _log_metric(operation: str, count: int, duration: float):
        
    """_log_metric function."""
if dd_api:
            dd_api.Metric.send(
                metric=f"event.{operation}.duration",
                points=duration,
                tags=[f"count:{count}"]
            )

    @staticmethod
    def _log_error(error_type: str, value: Any):
        
    """_log_error function."""
if sentry_sdk:
            sentry_sdk.capture_message(f"Event error: {error_type} - {value}")

class EventMixin:
    """Mixin for event handling functionality."""
    
    _event_handlers: Dict[EventType, Set[Callable]] = {}
    _event_history: List[Event] = []
    _max_history_size: int = 1000
    
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = set()
        self._event_handlers[event_type].add(handler)
    
    def unregister_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].discard(handler)
    
    def emit_event(self, event_type: EventType, data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Event:
        """Emit an event."""
        event = Event(
            type=event_type,
            status=EventStatus.PENDING,
            data=data or {},
            metadata=metadata or {}
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        # Notify handlers
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    event.status = EventStatus.FAILED
                    event.metadata['error'] = str(e)
                    continue
        
        event.status = EventStatus.COMPLETED
        return event
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: Optional[int] = None) -> List[Event]:
        """Get event history."""
        history = self._event_history
        if event_type:
            history = [e for e in history if e.type == event_type]
        if limit:
            history = history[-limit:]
        return history
    
    def clear_event_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def get_event_handlers(self, event_type: Optional[EventType] = None) -> Dict[EventType, Set[Callable]]:
        """Get event handlers."""
        if event_type:
            return {event_type: self._event_handlers.get(event_type, set())}
        return self._event_handlers.copy() 