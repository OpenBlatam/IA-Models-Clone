"""
Observability Utilities
=======================

Utilities for system observability and monitoring.
"""

import time
import logging
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Trace span for distributed tracing."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get span duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


class Tracer:
    """
    Distributed tracing utility.
    
    Features:
    - Span creation and management
    - Tag and log support
    - Context propagation
    """
    
    def __init__(self):
        """Initialize tracer."""
        self.spans: List[TraceSpan] = []
        self.active_spans: Dict[str, TraceSpan] = {}
        self._lock = threading.Lock()
    
    def start_span(self, name: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new span.
        
        Args:
            name: Span name
            tags: Optional tags
        
        Returns:
            Span ID
        """
        span_id = f"{name}_{int(time.time() * 1000000)}"
        span = TraceSpan(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self.active_spans[span_id] = span
        
        return span_id
    
    def finish_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Finish a span.
        
        Args:
            span_id: Span ID
            tags: Optional additional tags
        """
        with self._lock:
            if span_id in self.active_spans:
                span = self.active_spans.pop(span_id)
                span.end_time = time.time()
                if tags:
                    span.tags.update(tags)
                self.spans.append(span)
    
    def add_tag(self, span_id: str, key: str, value: Any) -> None:
        """Add tag to span."""
        with self._lock:
            if span_id in self.active_spans:
                self.active_spans[span_id].tags[key] = value
    
    def add_log(self, span_id: str, message: str, fields: Optional[Dict[str, Any]] = None) -> None:
        """Add log to span."""
        with self._lock:
            if span_id in self.active_spans:
                self.active_spans[span_id].logs.append({
                    "timestamp": time.time(),
                    "message": message,
                    "fields": fields or {}
                })
    
    def get_trace(self) -> List[TraceSpan]:
        """Get all completed spans."""
        with self._lock:
            return self.spans.copy()
    
    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self.spans.clear()
            self.active_spans.clear()


class MetricsCollector:
    """
    Advanced metrics collector.
    
    Features:
    - Counters
    - Gauges
    - Histograms
    - Timers
    - Aggregation
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window
        """
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment counter."""
        with self._lock:
            self.counters[name] += value
    
    def decrement(self, name: str, value: int = 1) -> None:
        """Decrement counter."""
        with self._lock:
            self.counters[name] -= value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record histogram value."""
        with self._lock:
            self.histograms[name].append(value)
    
    def record_timer(self, name: str, duration: float) -> None:
        """Record timer duration."""
        with self._lock:
            self.timers[name].append(duration)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
    
    @contextmanager
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record_timer(name, duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            metrics = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timers": {}
            }
            
            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    metrics["histograms"][name] = {
                        "count": len(sorted_values),
                        "min": min(sorted_values),
                        "max": max(sorted_values),
                        "mean": sum(sorted_values) / len(sorted_values),
                        "p50": sorted_values[len(sorted_values) // 2],
                        "p95": sorted_values[int(len(sorted_values) * 0.95)],
                        "p99": sorted_values[int(len(sorted_values) * 0.99)]
                    }
            
            # Calculate timer statistics
            for name, durations in self.timers.items():
                if durations:
                    sorted_durations = sorted(durations)
                    metrics["timers"][name] = {
                        "count": len(sorted_durations),
                        "min": min(sorted_durations),
                        "max": max(sorted_durations),
                        "mean": sum(sorted_durations) / len(sorted_durations),
                        "p50": sorted_durations[len(sorted_durations) // 2],
                        "p95": sorted_durations[int(len(sorted_durations) * 0.95)],
                        "p99": sorted_durations[int(len(sorted_durations) * 0.99)]
                    }
            
            return metrics


def trace_function(tracer: Optional[Tracer] = None):
    """
    Decorator to trace function execution.
    
    Args:
        tracer: Optional tracer instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if tracer is None:
                return func(*args, **kwargs)
            
            span_id = tracer.start_span(func.__name__)
            try:
                result = func(*args, **kwargs)
                tracer.finish_span(span_id, tags={"status": "success"})
                return result
            except Exception as e:
                tracer.finish_span(span_id, tags={"status": "error", "error": str(e)})
                raise
        
        return wrapper
    return decorator

