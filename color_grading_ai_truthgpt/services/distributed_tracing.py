"""
Distributed Tracing for Color Grading AI
=========================================

Advanced distributed tracing with spans, traces, and context propagation.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable for current trace
_current_trace: ContextVar[Optional['Trace']] = ContextVar('current_trace', default=None)


class SpanKind(Enum):
    """Span kinds."""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Span status."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class Span:
    """Tracing span."""
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    parent_span_id: Optional[str] = None
    error: Optional[Exception] = None


@dataclass
class Trace:
    """Distributed trace."""
    trace_id: str
    service_name: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.UNSET


class DistributedTracer:
    """
    Distributed tracing system.
    
    Features:
    - Span creation and management
    - Trace context propagation
    - Automatic instrumentation
    - Export to backends
    - Sampling
    """
    
    def __init__(self, service_name: str, sample_rate: float = 1.0):
        """
        Initialize distributed tracer.
        
        Args:
            service_name: Service name
            sample_rate: Sampling rate (0.0 - 1.0)
        """
        self.service_name = service_name
        self.sample_rate = sample_rate
        self._traces: Dict[str, Trace] = {}
        self._spans: Dict[str, Span] = {}
        self._max_traces = 10000
        self._exporters: List[Callable] = []
    
    def start_trace(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Start new trace.
        
        Args:
            operation_name: Operation name
            trace_id: Optional trace ID
            parent_trace_id: Optional parent trace ID
            attributes: Optional attributes
            
        Returns:
            Trace object
        """
        trace_id = trace_id or self._generate_id()
        
        trace = Trace(
            trace_id=trace_id,
            service_name=self.service_name,
            operation_name=operation_name,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        
        self._traces[trace_id] = trace
        
        # Set in context
        _current_trace.set(trace)
        
        logger.debug(f"Started trace: {trace_id} - {operation_name}")
        
        return trace
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start new span.
        
        Args:
            name: Span name
            kind: Span kind
            parent_span_id: Optional parent span ID
            attributes: Optional attributes
            
        Returns:
            Span object
        """
        trace = _current_trace.get()
        if not trace:
            # Create new trace if none exists
            trace = self.start_trace(name)
        
        span_id = self._generate_id()
        
        span = Span(
            span_id=span_id,
            trace_id=trace.trace_id,
            name=name,
            kind=kind,
            start_time=datetime.now(),
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        
        self._spans[span_id] = span
        trace.spans.append(span)
        
        logger.debug(f"Started span: {span_id} - {name}")
        
        return span
    
    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[Exception] = None
    ):
        """
        End span.
        
        Args:
            span_id: Span ID
            status: Span status
            error: Optional error
        """
        if span_id not in self._spans:
            logger.warning(f"Span not found: {span_id}")
            return
        
        span = self._spans[span_id]
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error = error
        
        if error:
            span.attributes["error"] = str(error)
            span.events.append({
                "name": "exception",
                "timestamp": span.end_time.isoformat(),
                "attributes": {"error": str(error)}
            })
        
        logger.debug(f"Ended span: {span_id} - {span.duration_ms:.2f}ms")
    
    def end_trace(self, trace_id: str, status: SpanStatus = SpanStatus.OK):
        """
        End trace.
        
        Args:
            trace_id: Trace ID
            status: Trace status
        """
        if trace_id not in self._traces:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        trace = self._traces[trace_id]
        trace.end_time = datetime.now()
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        trace.status = status
        
        # Export trace
        self._export_trace(trace)
        
        logger.info(f"Ended trace: {trace_id} - {trace.duration_ms:.2f}ms")
    
    def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        if span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        span.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def set_attribute(self, span_id: str, key: str, value: Any):
        """Set span attribute."""
        if span_id not in self._spans:
            return
        
        self._spans[span_id].attributes[key] = value
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self._traces.get(trace_id)
    
    def get_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Trace]:
        """Get traces with filters."""
        traces = list(self._traces.values())
        
        if service_name:
            traces = [t for t in traces if t.service_name == service_name]
        if operation_name:
            traces = [t for t in traces if t.operation_name == operation_name]
        
        # Sort by start time
        traces.sort(key=lambda t: t.start_time, reverse=True)
        
        return traces[:limit]
    
    def register_exporter(self, exporter: Callable):
        """
        Register trace exporter.
        
        Args:
            exporter: Exporter function (trace: Trace) -> None
        """
        self._exporters.append(exporter)
        logger.info("Registered trace exporter")
    
    def _export_trace(self, trace: Trace):
        """Export trace to all exporters."""
        for exporter in self._exporters:
            try:
                exporter(trace)
            except Exception as e:
                logger.error(f"Error exporting trace: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return str(uuid.uuid4())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        if not self._traces:
            return {
                "traces_count": 0,
                "spans_count": 0,
                "avg_duration_ms": 0.0,
            }
        
        durations = [t.duration_ms for t in self._traces.values() if t.duration_ms]
        total_spans = sum(len(t.spans) for t in self._traces.values())
        
        return {
            "traces_count": len(self._traces),
            "spans_count": total_spans,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "min_duration_ms": min(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
        }


class TraceContext:
    """Context manager for tracing."""
    
    def __init__(self, tracer: DistributedTracer, name: str, kind: SpanKind = SpanKind.INTERNAL):
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.span: Optional[Span] = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.name, self.kind)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            status = SpanStatus.ERROR if exc_type else SpanStatus.OK
            self.tracer.end_span(self.span.span_id, status, exc_val)

