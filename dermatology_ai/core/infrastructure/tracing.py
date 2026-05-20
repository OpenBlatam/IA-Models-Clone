"""
Distributed Tracing Support
Provides tracing capabilities for distributed systems
"""

import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: list[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def finish(self, error: Optional[str] = None):
        """Finish span"""
        self.end_time = datetime.utcnow()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000
        if error:
            self.error = error
            self.tags["error"] = True
    
    def add_tag(self, key: str, value: Any):
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, **kwargs):
        """Add log entry to span"""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            **kwargs
        })


class Tracer:
    """Distributed tracer"""
    
    def __init__(self, service_name: str = "dermatology_ai"):
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
    
    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span
        
        Args:
            operation_name: Name of the operation
            trace_id: Trace ID (generated if not provided)
            parent_span_id: Parent span ID for nested spans
            tags: Initial tags
            
        Returns:
            Span instance
        """
        trace_id = trace_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        span.add_tag("service.name", self.service_name)
        
        self.spans[span_id] = span
        logger.debug(f"Started span: {operation_name} (trace_id: {trace_id}, span_id: {span_id})")
        
        return span
    
    def finish_span(self, span_id: str, error: Optional[str] = None):
        """Finish a span"""
        if span_id in self.spans:
            self.spans[span_id].finish(error)
            logger.debug(f"Finished span: {span_id}")
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID"""
        return self.spans.get(span_id)
    
    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace"""
        return [span for span in self.spans.values() if span.trace_id == trace_id]
    
    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """Export trace in standard format"""
        spans = self.get_trace(trace_id)
        
        return {
            "trace_id": trace_id,
            "service": self.service_name,
            "spans": [
                {
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "operation": span.operation_name,
                    "start_time": span.start_time.isoformat() if span.start_time else None,
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "duration_ms": span.duration_ms,
                    "tags": span.tags,
                    "logs": span.logs,
                    "error": span.error
                }
                for span in spans
            ]
        }


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "dermatology_ai") -> Tracer:
    """Get global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name=service_name)
    return _tracer


def trace_operation(operation_name: str):
    """
    Decorator to trace an operation
    
    Args:
        operation_name: Name of the operation
    """
    def decorator(func):
        import functools
        import asyncio
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.start_span(operation_name)
            
            try:
                result = await func(*args, **kwargs)
                tracer.finish_span(span.span_id)
                return result
            except Exception as e:
                tracer.finish_span(span.span_id, error=str(e))
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.start_span(operation_name)
            
            try:
                result = func(*args, **kwargs)
                tracer.finish_span(span.span_id)
                return result
            except Exception as e:
                tracer.finish_span(span.span_id, error=str(e))
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator















