"""
Distributed Tracing
===================

System for distributed tracing and request tracking.
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Context variable for current trace
_current_trace: ContextVar[Optional['Trace']] = ContextVar('current_trace', default=None)


@dataclass
class Span:
    """Tracing span."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=dict)
    parent_id: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "parent_id": self.parent_id
        }


@dataclass
class Trace:
    """Distributed trace."""
    trace_id: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "spans": [span.to_dict() for span in self.spans],
            "tags": self.tags
        }
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class Tracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str):
        """
        Initialize tracer.
        
        Args:
            service_name: Service name
        """
        self.service_name = service_name
        self.traces: List[Trace] = []
        self.max_traces = 1000  # Keep last 1000 traces
    
    def start_trace(
        self,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Start a new trace.
        
        Args:
            trace_id: Optional trace ID (generated if not provided)
            tags: Optional trace tags
            
        Returns:
            Trace object
        """
        trace_id = trace_id or str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            service_name=self.service_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        _current_trace.set(trace)
        return trace
    
    def end_trace(self, trace: Optional[Trace] = None):
        """
        End a trace.
        
        Args:
            trace: Trace to end (uses current if not provided)
        """
        if trace is None:
            trace = _current_trace.get()
        
        if trace:
            trace.end_time = time.time()
            self._add_trace(trace)
            _current_trace.set(None)
    
    def _add_trace(self, trace: Trace):
        """Add trace to storage."""
        self.traces.append(trace)
        
        # Limit traces
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]
    
    @contextmanager
    def trace(self, trace_id: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing.
        
        Args:
            trace_id: Optional trace ID
            tags: Optional trace tags
            
        Usage:
            with tracer.trace() as trace:
                # code to trace
        """
        trace = self.start_trace(trace_id, tags)
        try:
            yield trace
        finally:
            self.end_trace(trace)
    
    def start_span(
        self,
        name: str,
        trace: Optional[Trace] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a span.
        
        Args:
            name: Span name
            trace: Trace to add span to (uses current if not provided)
            tags: Optional span tags
            
        Returns:
            Span object
        """
        if trace is None:
            trace = _current_trace.get()
        
        if not trace:
            raise ValueError("No active trace")
        
        span = Span(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        trace.spans.append(span)
        return span
    
    def end_span(self, span: Span):
        """
        End a span.
        
        Args:
            span: Span to end
        """
        span.end_time = time.time()
    
    @contextmanager
    def span(self, name: str, trace: Optional[Trace] = None, tags: Optional[Dict[str, Any]] = None):
        """
        Context manager for span.
        
        Args:
            name: Span name
            trace: Trace to add span to
            tags: Optional span tags
            
        Usage:
            with tracer.span("operation") as span:
                # code to trace
        """
        span = self.start_span(name, trace, tags)
        try:
            yield span
        finally:
            self.end_span(span)
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """
        Get trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace or None
        """
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None
    
    def get_recent_traces(self, limit: int = 10) -> List[Trace]:
        """
        Get recent traces.
        
        Args:
            limit: Number of traces to return
            
        Returns:
            List of recent traces
        """
        return self.traces[-limit:]


def get_current_trace() -> Optional[Trace]:
    """Get current trace from context."""
    return _current_trace.get()


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID from context."""
    trace = _current_trace.get()
    return trace.trace_id if trace else None




