"""
Tracer

Utilities for distributed tracing.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Span:
    """Tracing span."""
    
    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_id: Optional[str] = None
    ):
        """
        Initialize span.
        
        Args:
            name: Span name
            trace_id: Trace ID
            span_id: Span ID
            parent_id: Parent span ID
        """
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.tags: Dict[str, Any] = {}
        self.logs: list = []
    
    def finish(self) -> None:
        """Finish span."""
        self.end_time = time.time()
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span."""
        self.tags[key] = value
    
    def log(self, message: str, **kwargs) -> None:
        """Add log to span."""
        self.logs.append({
            'message': message,
            'timestamp': time.time(),
            **kwargs
        })
    
    def get_duration(self) -> float:
        """Get span duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class Tracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str = "music_generation"):
        """
        Initialize tracer.
        
        Args:
            service_name: Service name
        """
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self.current_trace_id: Optional[str] = None
    
    def start_trace(self, trace_name: str) -> str:
        """
        Start new trace.
        
        Args:
            trace_name: Trace name
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        self.current_trace_id = trace_id
        
        span = Span(trace_name, trace_id, str(uuid.uuid4()))
        span.add_tag("service", self.service_name)
        self.spans[span.span_id] = span
        
        logger.debug(f"Started trace: {trace_id}")
        
        return trace_id
    
    @contextmanager
    def span(
        self,
        name: str,
        parent_id: Optional[str] = None
    ):
        """
        Create span context.
        
        Args:
            name: Span name
            parent_id: Parent span ID
            
        Yields:
            Span instance
        """
        if not self.current_trace_id:
            self.start_trace("default")
        
        span_id = str(uuid.uuid4())
        span = Span(
            name,
            self.current_trace_id,
            span_id,
            parent_id
        )
        
        self.spans[span_id] = span
        
        try:
            yield span
        finally:
            span.finish()
            logger.debug(f"Finished span: {name} (duration: {span.get_duration():.3f}s)")
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Get trace information.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace information
        """
        trace_spans = [
            span for span in self.spans.values()
            if span.trace_id == trace_id
        ]
        
        return {
            'trace_id': trace_id,
            'spans': [
                {
                    'name': span.name,
                    'span_id': span.span_id,
                    'duration': span.get_duration(),
                    'tags': span.tags
                }
                for span in trace_spans
            ]
        }


def trace_function(tracer: Tracer, name: Optional[str] = None):
    """
    Trace function decorator.
    
    Args:
        tracer: Tracer instance
        name: Span name (uses function name if None)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            
            with tracer.span(span_name) as span:
                span.add_tag("function", func.__name__)
                result = func(*args, **kwargs)
                span.add_tag("success", True)
                return result
        
        return wrapper
    
    return decorator


def create_span(
    tracer: Tracer,
    name: str
):
    """Create span context."""
    return tracer.span(name)


def get_trace_context(tracer: Tracer) -> Dict[str, str]:
    """Get current trace context."""
    return {
        'trace_id': tracer.current_trace_id or '',
        'service': tracer.service_name
    }



