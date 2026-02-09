"""Observability utilities."""

from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import time

from utils.logger import get_logger

logger = get_logger(__name__)


class Span:
    """Represents a tracing span."""
    
    def __init__(self, name: str, parent: Optional['Span'] = None):
        self.name = name
        self.parent = parent
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.tags: Dict[str, Any] = {}
        self.logs: list = []
    
    def start(self) -> None:
        """Start the span."""
        self.start_time = time.time()
    
    def finish(self) -> None:
        """Finish the span."""
        self.end_time = time.time()
    
    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self.tags[key] = value
    
    def log(self, message: str, **kwargs) -> None:
        """Log an event in the span."""
        self.logs.append({
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
    
    @property
    def duration(self) -> Optional[float]:
        """Get span duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class Tracer:
    """Simple tracer for distributed tracing."""
    
    def __init__(self):
        self.spans: list = []
        self.current_span: Optional[Span] = None
    
    def start_span(self, name: str) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            
        Returns:
            Span instance
        """
        span = Span(name, parent=self.current_span)
        span.start()
        self.current_span = span
        self.spans.append(span)
        return span
    
    def finish_span(self, span: Span) -> None:
        """
        Finish a span.
        
        Args:
            span: Span to finish
        """
        span.finish()
        if span == self.current_span:
            self.current_span = span.parent
    
    @asynccontextmanager
    async def span(self, name: str):
        """
        Context manager for a span.
        
        Args:
            name: Span name
            
        Yields:
            Span instance
        """
        span = self.start_span(name)
        try:
            yield span
        finally:
            self.finish_span(span)
    
    def get_trace(self) -> Dict[str, Any]:
        """
        Get trace information.
        
        Returns:
            Dictionary with trace data
        """
        return {
            "spans": [
                {
                    "name": span.name,
                    "duration": span.duration,
                    "tags": span.tags,
                    "logs": span.logs
                }
                for span in self.spans
            ],
            "total_spans": len(self.spans),
            "total_duration": sum(
                s.duration for s in self.spans if s.duration
            )
        }


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer

