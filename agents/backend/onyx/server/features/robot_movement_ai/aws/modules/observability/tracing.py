"""
Distributed Tracing
===================

Advanced distributed tracing with OpenTelemetry.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Trace context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


class DistributedTracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str, endpoint: Optional[str] = None):
        self.service_name = service_name
        self.endpoint = endpoint
        self._spans: Dict[str, Dict[str, Any]] = {}
    
    def start_trace(self, operation_name: str, context: Optional[TraceContext] = None) -> TraceContext:
        """Start a new trace."""
        if context:
            trace_id = context.trace_id
            parent_span_id = context.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        span_id = str(uuid.uuid4())
        
        new_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=context.baggage.copy() if context else {}
        )
        
        self._spans[span_id] = {
            "operation": operation_name,
            "start_time": time.time(),
            "context": new_context,
            "tags": {},
            "logs": []
        }
        
        logger.debug(f"Started trace: {trace_id}, span: {span_id}, operation: {operation_name}")
        return new_context
    
    @contextmanager
    def span(self, operation_name: str, context: Optional[TraceContext] = None):
        """Context manager for span."""
        span_context = self.start_trace(operation_name, context)
        try:
            yield span_context
        finally:
            self.end_span(span_context.span_id)
    
    def end_span(self, span_id: str, status: str = "OK", error: Optional[Exception] = None):
        """End a span."""
        if span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        duration = time.time() - span["start_time"]
        
        span["end_time"] = time.time()
        span["duration"] = duration
        span["status"] = status
        if error:
            span["error"] = str(error)
        
        logger.info(
            f"Span ended: {span_id}, operation: {span['operation']}, "
            f"duration: {duration:.3f}s, status: {status}"
        )
    
    def add_tag(self, span_id: str, key: str, value: Any):
        """Add tag to span."""
        if span_id in self._spans:
            self._spans[span_id]["tags"][key] = value
    
    def add_log(self, span_id: str, message: str, level: str = "INFO", **kwargs):
        """Add log to span."""
        if span_id in self._spans:
            self._spans[span_id]["logs"].append({
                "message": message,
                "level": level,
                "timestamp": time.time(),
                **kwargs
            })
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get all spans for a trace."""
        spans = [
            span for span in self._spans.values()
            if span["context"].trace_id == trace_id
        ]
        return {
            "trace_id": trace_id,
            "spans": spans,
            "total_spans": len(spans),
            "total_duration": sum(s.get("duration", 0) for s in spans)
        }
    
    def inject_headers(self, context: TraceContext) -> Dict[str, str]:
        """Inject trace context into headers."""
        return {
            "X-Trace-ID": context.trace_id,
            "X-Span-ID": context.span_id,
            "X-Parent-Span-ID": context.parent_span_id or "",
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")
        parent_span_id = headers.get("X-Parent-Span-ID")
        
        if trace_id and span_id:
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id or None
            )
        return None















