#!/usr/bin/env python3
"""
Distributed Tracing System

Advanced distributed tracing with:
- Request tracing across services
- Span creation and management
- Trace correlation and analysis
- Performance monitoring
- Error tracking and debugging
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import threading
import contextvars

logger = structlog.get_logger("distributed_tracing")

# =============================================================================
# TRACING MODELS
# =============================================================================

class SpanStatus(Enum):
    """Span status enumeration."""
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class SpanKind(Enum):
    """Span kind enumeration."""
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"

@dataclass
class SpanContext:
    """Span context for trace propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = None
    
    def __post_init__(self):
        if self.baggage is None:
            self.baggage = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpanContext:
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {})
        )

@dataclass
class Span:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime]
    status: SpanStatus
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    service_name: str
    operation_name: str
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []
        if self.links is None:
            self.links = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
            "service_name": self.service_name,
            "operation_name": self.operation_name,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Span:
        """Create from dictionary."""
        return cls(
            span_id=data["span_id"],
            trace_id=data["trace_id"],
            parent_span_id=data.get("parent_span_id"),
            name=data["name"],
            kind=SpanKind(data["kind"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=SpanStatus(data["status"]),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
            links=data.get("links", []),
            service_name=data["service_name"],
            operation_name=data["operation_name"]
        )

@dataclass
class Trace:
    """Complete trace with all spans."""
    trace_id: str
    spans: List[Span]
    start_time: datetime
    end_time: Optional[datetime]
    service_count: int
    span_count: int
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.spans],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "service_count": self.service_count,
            "span_count": self.span_count,
            "error_count": self.error_count,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }

# =============================================================================
# DISTRIBUTED TRACER
# =============================================================================

class DistributedTracer:
    """Advanced distributed tracer."""
    
    def __init__(self, service_name: str, max_traces: int = 10000):
        self.service_name = service_name
        self.max_traces = max_traces
        self.traces: Dict[str, Trace] = {}
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, Span] = {}
        
        # Context variables for async context
        self.current_span_context = contextvars.ContextVar('span_context', default=None)
        self.current_span = contextvars.ContextVar('current_span', default=None)
        
        # Statistics
        self.stats = {
            'traces_created': 0,
            'spans_created': 0,
            'spans_completed': 0,
            'spans_failed': 0,
            'total_trace_duration': 0.0,
            'average_trace_duration': 0.0
        }
        
        # Trace sampling
        self.sampling_rate = 1.0  # 100% sampling by default
        self.sampled_traces: Set[str] = set()
    
    def start_trace(self, trace_id: Optional[str] = None, 
                   parent_context: Optional[SpanContext] = None) -> SpanContext:
        """Start a new trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        # Check sampling
        if not self._should_sample(trace_id):
            return None
        
        # Create root span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_context.span_id if parent_context else None,
            baggage=parent_context.baggage.copy() if parent_context else {}
        )
        
        # Set context
        self.current_span_context.set(span_context)
        
        # Create trace
        trace = Trace(
            trace_id=trace_id,
            spans=[],
            start_time=datetime.utcnow(),
            end_time=None,
            service_count=0,
            span_count=0,
            error_count=0
        )
        
        self.traces[trace_id] = trace
        self.sampled_traces.add(trace_id)
        self.stats['traces_created'] += 1
        
        logger.debug("Trace started", trace_id=trace_id)
        
        return span_context
    
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                  parent_context: Optional[SpanContext] = None,
                  attributes: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span."""
        # Get current context
        current_context = parent_context or self.current_span_context.get()
        
        if not current_context:
            # Create new trace if no context
            current_context = self.start_trace()
        
        # Create span
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=current_context.trace_id,
            parent_span_id=current_context.span_id,
            name=name,
            kind=kind,
            start_time=datetime.utcnow(),
            end_time=None,
            status=SpanStatus.OK,
            attributes=attributes or {},
            events=[],
            links=[],
            service_name=self.service_name,
            operation_name=name
        )
        
        # Store span
        self.spans[span.span_id] = span
        self.active_spans[span.span_id] = span
        
        # Add to trace
        if span.trace_id in self.traces:
            self.traces[span.trace_id].spans.append(span)
            self.traces[span.trace_id].span_count += 1
        
        # Set current span
        self.current_span.set(span)
        
        # Update context
        new_context = SpanContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            baggage=current_context.baggage.copy()
        )
        self.current_span_context.set(new_context)
        
        self.stats['spans_created'] += 1
        
        logger.debug(
            "Span started",
            span_id=span.span_id,
            trace_id=span.trace_id,
            name=name,
            kind=kind.value
        )
        
        return span
    
    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK,
                attributes: Optional[Dict[str, Any]] = None) -> None:
        """End a span."""
        span.end_time = datetime.utcnow()
        span.status = status
        
        if attributes:
            span.attributes.update(attributes)
        
        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        # Update statistics
        if status == SpanStatus.OK:
            self.stats['spans_completed'] += 1
        else:
            self.stats['spans_failed'] += 1
        
        # Update trace
        if span.trace_id in self.traces:
            trace = self.traces[span.trace_id]
            if span.status == SpanStatus.ERROR:
                trace.error_count += 1
            
            # Check if trace is complete
            if self._is_trace_complete(trace):
                trace.end_time = span.end_time
                self._update_trace_stats(trace)
        
        logger.debug(
            "Span ended",
            span_id=span.span_id,
            trace_id=span.trace_id,
            status=status.value,
            duration=(span.end_time - span.start_time).total_seconds()
        )
    
    def add_span_event(self, span: Span, name: str, 
                      attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        event = {
            'name': name,
            'timestamp': datetime.utcnow().isoformat(),
            'attributes': attributes or {}
        }
        
        span.events.append(event)
        
        logger.debug(
            "Event added to span",
            span_id=span.span_id,
            event_name=name
        )
    
    def add_span_attribute(self, span: Span, key: str, value: Any) -> None:
        """Add attribute to span."""
        span.attributes[key] = value
    
    def create_span_context(self, trace_id: str, span_id: str,
                           parent_span_id: Optional[str] = None,
                           baggage: Optional[Dict[str, str]] = None) -> SpanContext:
        """Create span context for propagation."""
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage or {}
        )
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers."""
        trace_id = headers.get('X-Trace-ID')
        span_id = headers.get('X-Span-ID')
        parent_span_id = headers.get('X-Parent-Span-ID')
        
        if not trace_id or not span_id:
            return None
        
        # Extract baggage
        baggage = {}
        for key, value in headers.items():
            if key.startswith('X-Baggage-'):
                baggage_key = key[10:]  # Remove 'X-Baggage-' prefix
                baggage[baggage_key] = value
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )
    
    def inject_context(self, context: SpanContext) -> Dict[str, str]:
        """Inject span context into headers."""
        headers = {
            'X-Trace-ID': context.trace_id,
            'X-Span-ID': context.span_id
        }
        
        if context.parent_span_id:
            headers['X-Parent-Span-ID'] = context.parent_span_id
        
        # Inject baggage
        for key, value in context.baggage.items():
            headers[f'X-Baggage-{key}'] = value
        
        return headers
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self.traces.get(trace_id)
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        return self.spans.get(span_id)
    
    def get_traces_by_service(self, service_name: str) -> List[Trace]:
        """Get traces by service name."""
        return [
            trace for trace in self.traces.values()
            if any(span.service_name == service_name for span in trace.spans)
        ]
    
    def get_traces_by_time_range(self, start_time: datetime, 
                                end_time: datetime) -> List[Trace]:
        """Get traces by time range."""
        return [
            trace for trace in self.traces.values()
            if start_time <= trace.start_time <= end_time
        ]
    
    def get_traces_with_errors(self) -> List[Trace]:
        """Get traces with errors."""
        return [
            trace for trace in self.traces.values()
            if trace.error_count > 0
        ]
    
    def _should_sample(self, trace_id: str) -> bool:
        """Check if trace should be sampled."""
        if self.sampling_rate >= 1.0:
            return True
        
        # Simple hash-based sampling
        hash_value = hash(trace_id) % 100
        return hash_value < (self.sampling_rate * 100)
    
    def _is_trace_complete(self, trace: Trace) -> bool:
        """Check if trace is complete."""
        # A trace is complete when all spans are ended
        return all(span.end_time is not None for span in trace.spans)
    
    def _update_trace_stats(self, trace: Trace) -> None:
        """Update trace statistics."""
        if trace.end_time:
            duration = (trace.end_time - trace.start_time).total_seconds()
            self.stats['total_trace_duration'] += duration
            
            # Update average
            completed_traces = len([t for t in self.traces.values() if t.end_time])
            if completed_traces > 0:
                self.stats['average_trace_duration'] = (
                    self.stats['total_trace_duration'] / completed_traces
                )
    
    def get_tracer_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        return {
            **self.stats,
            'active_traces': len(self.traces),
            'active_spans': len(self.active_spans),
            'sampling_rate': self.sampling_rate,
            'sampled_traces': len(self.sampled_traces)
        }
    
    def set_sampling_rate(self, rate: float) -> None:
        """Set sampling rate (0.0 to 1.0)."""
        self.sampling_rate = max(0.0, min(1.0, rate))
        logger.info("Sampling rate updated", rate=self.sampling_rate)

# =============================================================================
# TRACE CONTEXT MANAGER
# =============================================================================

class TraceContext:
    """Context manager for automatic span management."""
    
    def __init__(self, tracer: DistributedTracer, name: str, 
                 kind: SpanKind = SpanKind.INTERNAL,
                 attributes: Optional[Dict[str, Any]] = None):
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.attributes = attributes
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Span:
        """Enter context and start span."""
        self.span = self.tracer.start_span(
            name=self.name,
            kind=self.kind,
            attributes=self.attributes
        )
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and end span."""
        if self.span:
            status = SpanStatus.ERROR if exc_type else SpanStatus.OK
            self.tracer.end_span(self.span, status=status)
    
    async def __aenter__(self) -> Span:
        """Async enter context and start span."""
        self.span = self.tracer.start_span(
            name=self.name,
            kind=self.kind,
            attributes=self.attributes
        )
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async exit context and end span."""
        if self.span:
            status = SpanStatus.ERROR if exc_type else SpanStatus.OK
            self.tracer.end_span(self.span, status=status)

# =============================================================================
# TRACE DECORATOR
# =============================================================================

def trace_span(name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL,
               attributes: Optional[Dict[str, Any]] = None):
    """Decorator for automatic span creation."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with TraceContext(distributed_tracer, span_name, kind, attributes):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with TraceContext(distributed_tracer, span_name, kind, attributes):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =============================================================================
# GLOBAL DISTRIBUTED TRACER INSTANCE
# =============================================================================

# Global distributed tracer
distributed_tracer = DistributedTracer("video-opusclip-api")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SpanStatus',
    'SpanKind',
    'SpanContext',
    'Span',
    'Trace',
    'DistributedTracer',
    'TraceContext',
    'trace_span',
    'distributed_tracer'
]





























