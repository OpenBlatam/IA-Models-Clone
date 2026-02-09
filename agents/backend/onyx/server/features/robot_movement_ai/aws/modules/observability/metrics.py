"""
Metrics Collector
=================

Advanced metrics collection.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Metrics collector."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._metrics: Dict[str, List[Metric]] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
    
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter."""
        key = self._build_key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {}
        )
        self._record_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value."""
        key = self._build_key(name, labels)
        self._gauges[key] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        )
        self._record_metric(metric)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram value."""
        key = self._build_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {}
        )
        self._record_metric(metric)
    
    def record_duration(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record duration as histogram."""
        self.observe_histogram(f"{name}_duration", duration, labels)
    
    @staticmethod
    def _build_key(name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build metric key."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def _record_metric(self, metric: Metric):
        """Record metric."""
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []
        self._metrics[metric.name].append(metric)
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics."""
        if name:
            return {
                "name": name,
                "metrics": self._metrics.get(name, []),
                "counter": self._counters.get(name, 0.0),
                "gauge": self._gauges.get(name, 0.0),
                "histogram": self._histograms.get(name, [])
            }
        
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": {k: len(v) for k, v in self._histograms.items()},
            "all_metrics": {k: len(v) for k, v in self._metrics.items()}
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms
        for key, values in self._histograms.items():
            if values:
                lines.append(f"# TYPE {key} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")
        
        return "\n".join(lines)















