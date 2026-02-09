"""
Request Tracing
===============

Advanced request tracing for distributed systems and debugging.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable for trace ID
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)

@dataclass
class TraceSpan:
    """Trace span definition."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

class RequestTracer:
    """Advanced request tracer."""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self.max_traces = 10000
        self.max_spans_per_trace = 1000
    
    def start_trace(self, operation_name: str = "root", trace_id: Optional[str] = None) -> str:
        """Start a new trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span
        
        # Set context variable
        trace_id_var.set(trace_id)
        
        logger.debug(f"Trace started: {trace_id}, span: {span_id}")
        
        return span_id
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> str:
        """Start a new span."""
        if trace_id is None:
            trace_id = trace_id_var.get() or self.start_trace("root")
        
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span
        
        logger.debug(f"Span started: {span_id} (parent: {parent_span_id})")
        
        return span_id
    
    def end_span(self, span_id: str, error: Optional[str] = None):
        """End a span."""
        if span_id not in self.active_spans:
            logger.warning(f"Span not found: {span_id}")
            return
        
        span = self.active_spans[span_id]
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        
        if error:
            span.error = error
        
        del self.active_spans[span_id]
        
        logger.debug(f"Span ended: {span_id} (duration: {span.duration_ms:.2f}ms)")
    
    def add_tag(self, span_id: str, key: str, value: Any):
        """Add tag to span."""
        if span_id in self.active_spans:
            self.active_spans[span_id].tags[key] = value
    
    def add_log(self, span_id: str, message: str, level: str = "INFO"):
        """Add log to span."""
        if span_id in self.active_spans:
            self.active_spans[span_id].logs.append({
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get trace by ID."""
        if trace_id not in self.traces:
            return None
        
        return [
            {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_span_id": span.parent_span_id,
                "operation_name": span.operation_name,
                "start_time": span.start_time.isoformat(),
                "end_time": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": span.duration_ms,
                "tags": span.tags,
                "logs": span.logs,
                "error": span.error
            }
            for span in self.traces[trace_id]
        ]
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID from context."""
        return trace_id_var.get()
    
    def cleanup_old_traces(self, max_age_hours: int = 24):
        """Cleanup old traces."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        traces_to_remove = []
        for trace_id, spans in self.traces.items():
            if spans and spans[0].start_time < cutoff:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.traces[trace_id]
        
        if traces_to_remove:
            logger.info(f"Cleaned up {len(traces_to_remove)} old traces")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        total_traces = len(self.traces)
        total_spans = sum(len(spans) for spans in self.traces.values())
        active_spans = len(self.active_spans)
        
        return {
            "total_traces": total_traces,
            "total_spans": total_spans,
            "active_spans": active_spans,
            "avg_spans_per_trace": round(total_spans / total_traces, 2) if total_traces > 0 else 0
        }

# Global instance
request_tracer = RequestTracer()
















