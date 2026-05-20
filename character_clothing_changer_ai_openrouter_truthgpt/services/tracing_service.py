"""
Distributed Tracing Service
===========================
Service for distributed tracing across services
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Span status"""
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class SpanKind(Enum):
    """Span kind"""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


@dataclass
class Span:
    """Tracing span"""
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNKNOWN
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


@dataclass
class Trace:
    """Distributed trace"""
    trace_id: str
    spans: List[Span]
    start_time: datetime
    end_time: Optional[datetime] = None
    service_name: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get trace duration in milliseconds"""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class TracingService:
    """
    Service for distributed tracing.
    
    Features:
    - Span creation and management
    - Trace context propagation
    - Span attributes and events
    - Error tracking
    - Statistics
    """
    
    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize tracing service.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name or "unknown"
        self._traces: Dict[str, Trace] = {}
        self._active_spans: Dict[str, Span] = {}
        self._stats = {
            'traces': 0,
            'spans': 0,
            'errors': 0
        }
    
    def start_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Start a new trace.
        
        Args:
            name: Trace name
            trace_id: Optional trace ID
            attributes: Optional trace attributes
        
        Returns:
            Trace object
        """
        if trace_id is None:
            trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        
        trace = Trace(
            trace_id=trace_id,
            spans=[],
            start_time=datetime.now(),
            service_name=self.service_name
        )
        
        self._traces[trace_id] = trace
        self._stats['traces'] += 1
        
        logger.debug(f"Trace started: {trace_id} ({name})")
        return trace
    
    def start_span(
        self,
        name: str,
        trace_id: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            trace_id: Trace ID
            kind: Span kind
            parent_span_id: Optional parent span ID
            attributes: Optional span attributes
        
        Returns:
            Span object
        """
        span_id = f"span_{uuid.uuid4().hex[:16]}"
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            kind=kind,
            start_time=datetime.now(),
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        
        # Add to trace
        if trace_id in self._traces:
            self._traces[trace_id].spans.append(span)
        
        self._active_spans[span_id] = span
        self._stats['spans'] += 1
        
        logger.debug(f"Span started: {span_id} ({name}) in trace {trace_id}")
        return span
    
    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        """
        End a span.
        
        Args:
            span_id: Span ID
            status: Span status
            error: Optional error message
            attributes: Optional additional attributes
        
        Returns:
            Span object or None if not found
        """
        span = self._active_spans.get(span_id)
        if not span:
            return None
        
        span.end_time = datetime.now()
        span.status = status
        span.error = error
        
        if attributes:
            span.attributes.update(attributes)
        
        if status == SpanStatus.ERROR:
            self._stats['errors'] += 1
        
        del self._active_spans[span_id]
        
        logger.debug(f"Span ended: {span_id} (status: {status.value}, duration: {span.duration_ms:.2f}ms)")
        return span
    
    def add_span_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Add event to span"""
        span = self._active_spans.get(span_id)
        if span:
            event = {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'attributes': attributes or {}
            }
            span.events.append(event)
    
    def add_span_attribute(
        self,
        span_id: str,
        key: str,
        value: Any
    ):
        """Add attribute to span"""
        span = self._active_spans.get(span_id)
        if span:
            span.attributes[key] = value
    
    def end_trace(self, trace_id: str) -> Optional[Trace]:
        """End a trace"""
        trace = self._traces.get(trace_id)
        if trace:
            trace.end_time = datetime.now()
            logger.debug(f"Trace ended: {trace_id} (duration: {trace.duration_ms:.2f}ms, spans: {len(trace.spans)})")
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        return self._traces.get(trace_id)
    
    def get_active_spans(self, trace_id: Optional[str] = None) -> List[Span]:
        """Get active spans, optionally filtered by trace"""
        if trace_id:
            return [s for s in self._active_spans.values() if s.trace_id == trace_id]
        return list(self._active_spans.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        total_spans = sum(len(trace.spans) for trace in self._traces.values())
        error_spans = sum(
            1 for trace in self._traces.values()
            for span in trace.spans
            if span.status == SpanStatus.ERROR
        )
        
        return {
            'service_name': self.service_name,
            'total_traces': len(self._traces),
            'total_spans': total_spans,
            'active_spans': len(self._active_spans),
            'error_spans': error_spans,
            'error_rate': error_spans / total_spans if total_spans > 0 else 0.0
        }


# Global tracing service instance
_tracing_service: Optional[TracingService] = None


def get_tracing_service(service_name: Optional[str] = None) -> TracingService:
    """Get or create tracing service instance"""
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService(service_name=service_name)
    return _tracing_service

