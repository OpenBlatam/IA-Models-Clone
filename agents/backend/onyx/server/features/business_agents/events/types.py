"""
Event Types and Definitions
===========================

Type definitions for event-driven architecture.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import asyncio

class EventType(Enum):
    """Event type enumeration."""
    # Agent events
    AGENT_CREATED = "agent_created"
    AGENT_UPDATED = "agent_updated"
    AGENT_DELETED = "agent_deleted"
    AGENT_EXECUTION_STARTED = "agent_execution_started"
    AGENT_EXECUTION_COMPLETED = "agent_execution_completed"
    AGENT_EXECUTION_FAILED = "agent_execution_failed"
    
    # Workflow events
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKFLOW_EXECUTION_STARTED = "workflow_execution_started"
    WORKFLOW_EXECUTION_COMPLETED = "workflow_execution_completed"
    WORKFLOW_EXECUTION_FAILED = "workflow_execution_failed"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    
    # Document events
    DOCUMENT_GENERATION_STARTED = "document_generation_started"
    DOCUMENT_GENERATION_COMPLETED = "document_generation_completed"
    DOCUMENT_GENERATION_FAILED = "document_generation_failed"
    DOCUMENT_DOWNLOADED = "document_downloaded"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_ALERT = "system_alert"
    METRICS_UPDATED = "metrics_updated"
    
    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    
    # Integration events
    EXTERNAL_API_CALL = "external_api_call"
    EXTERNAL_API_RESPONSE = "external_api_response"
    EXTERNAL_API_ERROR = "external_api_error"
    
    # Custom events
    CUSTOM_EVENT = "custom_event"

@dataclass
class Event:
    """Event definition."""
    id: str
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "business_agents_system"
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    @classmethod
    def create(
        cls,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "business_agents_system",
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Event':
        """Create a new event."""
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            data=data,
            source=source,
            correlation_id=correlation_id,
            causation_id=causation_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Handle an event. Return True if successful, False otherwise."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the given event."""
        pass
    
    def get_priority(self) -> int:
        """Get handler priority (lower number = higher priority)."""
        return 100
    
    def is_enabled(self) -> bool:
        """Check if handler is enabled."""
        return self.enabled
    
    def enable(self):
        """Enable the handler."""
        self.enabled = True
    
    def disable(self):
        """Disable the handler."""
        self.enabled = False

@dataclass
class EventSubscription:
    """Event subscription definition."""
    id: str
    event_types: List[EventType]
    handler: EventHandler
    filter_conditions: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    
    def matches(self, event: Event) -> bool:
        """Check if subscription matches the event."""
        if not self.enabled:
            return False
        
        if event.type not in self.event_types:
            return False
        
        if not self.handler.can_handle(event):
            return False
        
        # Apply filter conditions
        if self.filter_conditions:
            for key, expected_value in self.filter_conditions.items():
                if key in event.data:
                    if event.data[key] != expected_value:
                        return False
                elif key in event.metadata:
                    if event.metadata[key] != expected_value:
                        return False
                else:
                    return False
        
        return True

class EventFilter:
    """Event filtering utility."""
    
    def __init__(self):
        self.conditions: List[Callable[[Event], bool]] = []
    
    def add_condition(self, condition: Callable[[Event], bool]):
        """Add a filter condition."""
        self.conditions.append(condition)
    
    def matches(self, event: Event) -> bool:
        """Check if event matches all conditions."""
        return all(condition(event) for condition in self.conditions)
    
    @classmethod
    def by_event_type(cls, event_types: List[EventType]) -> 'EventFilter':
        """Create filter by event types."""
        filter_obj = cls()
        filter_obj.add_condition(lambda e: e.type in event_types)
        return filter_obj
    
    @classmethod
    def by_source(cls, sources: List[str]) -> 'EventFilter':
        """Create filter by sources."""
        filter_obj = cls()
        filter_obj.add_condition(lambda e: e.source in sources)
        return filter_obj
    
    @classmethod
    def by_user(cls, user_ids: List[str]) -> 'EventFilter':
        """Create filter by user IDs."""
        filter_obj = cls()
        filter_obj.add_condition(lambda e: e.user_id in user_ids)
        return filter_obj
    
    @classmethod
    def by_data_field(cls, field: str, value: Any) -> 'EventFilter':
        """Create filter by data field."""
        filter_obj = cls()
        filter_obj.add_condition(lambda e: e.data.get(field) == value)
        return filter_obj

class EventStore:
    """Event store for persisting events."""
    
    def __init__(self):
        self.events: List[Event] = []
        self._lock = asyncio.Lock()
    
    async def append(self, event: Event):
        """Append an event to the store."""
        async with self._lock:
            self.events.append(event)
    
    async def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events with filters."""
        async with self._lock:
            filtered_events = self.events.copy()
            
            if event_types:
                filtered_events = [e for e in filtered_events if e.type in event_types]
            
            if start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
            if end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
            
            if limit:
                filtered_events = filtered_events[:limit]
            
            return filtered_events
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        async with self._lock:
            for event in self.events:
                if event.id == event_id:
                    return event
            return None
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID."""
        async with self._lock:
            return [e for e in self.events if e.correlation_id == correlation_id]

class EventMetrics:
    """Event processing metrics."""
    
    def __init__(self):
        self.events_processed = 0
        self.events_failed = 0
        self.handlers_executed = 0
        self.handlers_failed = 0
        self.processing_times: List[float] = []
        self.event_type_counts: Dict[EventType, int] = {}
        self.handler_performance: Dict[str, List[float]] = {}
    
    def record_event_processed(self, event: Event, processing_time: float):
        """Record a processed event."""
        self.events_processed += 1
        self.processing_times.append(processing_time)
        self.event_type_counts[event.type] = self.event_type_counts.get(event.type, 0) + 1
    
    def record_event_failed(self, event: Event):
        """Record a failed event."""
        self.events_failed += 1
    
    def record_handler_executed(self, handler_name: str, execution_time: float):
        """Record a handler execution."""
        self.handlers_executed += 1
        if handler_name not in self.handler_performance:
            self.handler_performance[handler_name] = []
        self.handler_performance[handler_name].append(execution_time)
    
    def record_handler_failed(self, handler_name: str):
        """Record a failed handler."""
        self.handlers_failed += 1
    
    def get_average_processing_time(self) -> float:
        """Get average event processing time."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
    
    def get_success_rate(self) -> float:
        """Get event processing success rate."""
        total = self.events_processed + self.events_failed
        return (self.events_processed / total * 100) if total > 0 else 0.0
    
    def get_handler_average_time(self, handler_name: str) -> float:
        """Get average handler execution time."""
        times = self.handler_performance.get(handler_name, [])
        return sum(times) / len(times) if times else 0.0
