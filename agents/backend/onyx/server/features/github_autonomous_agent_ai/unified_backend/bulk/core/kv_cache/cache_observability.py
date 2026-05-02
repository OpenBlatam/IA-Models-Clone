"""
Advanced cache observability.

Provides comprehensive observability for cache operations.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric definition."""
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CacheObservability:
    """
    Cache observability manager.
    
    Provides comprehensive observability.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize observability.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.traces: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.spans: List[Dict[str, Any]] = []
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Metric type
            labels: Optional labels
        """
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=time.time()
        )
        self.metrics[name].append(metric)
    
    def record_trace(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record trace.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        trace = {
            "operation": operation,
            "duration": duration,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.traces.append(trace)
        
        # Keep only recent traces
        if len(self.traces) > 10000:
            self.traces = self.traces[-10000:]
    
    def record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record event.
        
        Args:
            event_type: Event type
            description: Event description
            metadata: Optional metadata
        """
        event = {
            "type": event_type,
            "description": description,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
    
    def start_span(self, operation: str) -> str:
        """
        Start span.
        
        Args:
            operation: Operation name
            
        Returns:
            Span ID
        """
        import uuid
        span_id = str(uuid.uuid4())
        
        span = {
            "span_id": span_id,
            "operation": operation,
            "start_time": time.time(),
            "status": "started"
        }
        self.spans.append(span)
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "completed") -> None:
        """
        Finish span.
        
        Args:
            span_id: Span ID
            status: Span status
        """
        for span in self.spans:
            if span["span_id"] == span_id:
                span["end_time"] = time.time()
                span["duration"] = span["end_time"] - span["start_time"]
                span["status"] = status
                break
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Metrics summary
        """
        summary = {}
        
        for name, metric_list in self.metrics.items():
            if not metric_list:
                continue
            
            values = [m.value for m in metric_list]
            
            summary[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0
            }
        
        return summary
    
    def get_traces_summary(self) -> Dict[str, Any]:
        """
        Get traces summary.
        
        Returns:
            Traces summary
        """
        if not self.traces:
            return {}
        
        operations = defaultdict(list)
        
        for trace in self.traces:
            operations[trace["operation"]].append(trace["duration"])
        
        summary = {}
        for op, durations in operations.items():
            summary[op] = {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                "p99": sorted(durations)[int(len(durations) * 0.99)] if durations else 0
            }
        
        return summary
    
    def get_events_summary(self) -> Dict[str, int]:
        """
        Get events summary.
        
        Returns:
            Events summary
        """
        summary = defaultdict(int)
        
        for event in self.events:
            summary[event["type"]] += 1
        
        return dict(summary)


class CacheMetricsExporter:
    """
    Cache metrics exporter.
    
    Exports metrics to various formats.
    """
    
    def __init__(self, observability: CacheObservability):
        """
        Initialize exporter.
        
        Args:
            observability: Observability instance
        """
        self.observability = observability
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus format string
        """
        lines = []
        
        for name, metric_list in self.observability.metrics.items():
            if not metric_list:
                continue
            
            latest = metric_list[-1]
            
            # Format labels
            labels_str = ""
            if latest.labels:
                labels_str = "{" + ",".join(f'{k}="{v}"' for k, v in latest.labels.items()) + "}"
            
            lines.append(f"{name}{labels_str} {latest.value}")
        
        return "\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """
        Export metrics in JSON format.
        
        Returns:
            JSON dictionary
        """
        return {
            "metrics": self.observability.get_metrics_summary(),
            "traces": self.observability.get_traces_summary(),
            "events": self.observability.get_events_summary()
        }
    
    def export_csv(self) -> str:
        """
        Export metrics in CSV format.
        
        Returns:
            CSV format string
        """
        lines = ["metric_name,value,timestamp"]
        
        for name, metric_list in self.observability.metrics.items():
            for metric in metric_list:
                lines.append(f"{name},{metric.value},{metric.timestamp}")
        
        return "\n".join(lines)

