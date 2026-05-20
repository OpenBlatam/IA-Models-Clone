"""
Telemetry
Metrics collection and reporting.
"""

import time
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class Metric:
    """Single metric value."""
    
    def __init__(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.value = value
        self.tags = tags or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


class Counter:
    """Counter metric."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tags = tags or {}
        self.value = 0
    
    def increment(self, amount: int = 1):
        """Increment counter."""
        self.value += amount
    
    def get(self) -> Metric:
        """Get current metric."""
        return Metric(self.name, self.value, self.tags)


class Gauge:
    """Gauge metric."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tags = tags or {}
        self.value = 0.0
    
    def set(self, value: float):
        """Set gauge value."""
        self.value = value
    
    def get(self) -> Metric:
        """Get current metric."""
        return Metric(self.name, self.value, self.tags)


class Histogram:
    """Histogram metric."""
    
    def __init__(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.tags = tags or {}
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.values = deque(maxlen=1000)  # Keep last 1000 values
    
    def observe(self, value: float):
        """Record a value."""
        self.values.append(value)
    
    def get(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        if not self.values:
            return {
                "name": self.name,
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }
        
        values_list = list(self.values)
        return {
            "name": self.name,
            "count": len(values_list),
            "sum": sum(values_list),
            "min": min(values_list),
            "max": max(values_list),
            "avg": sum(values_list) / len(values_list),
            "buckets": self._calculate_buckets(values_list),
        }
    
    def _calculate_buckets(self, values: List[float]) -> Dict[str, int]:
        """Calculate bucket counts."""
        buckets = {str(bucket): 0 for bucket in self.buckets}
        buckets["+Inf"] = 0
        
        for value in values:
            placed = False
            for bucket in self.buckets:
                if value <= bucket:
                    buckets[str(bucket)] += 1
                    placed = True
                    break
            if not placed:
                buckets["+Inf"] += 1
        
        return buckets


class TelemetryCollector:
    """Collect and manage metrics."""
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
    
    def counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create counter."""
        key = self._make_key(name, tags)
        if key not in self.counters:
            self.counters[key] = Counter(name, tags)
        return self.counters[key]
    
    def gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create gauge."""
        key = self._make_key(name, tags)
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, tags)
        return self.gauges[key]
    
    def histogram(self, name: str, buckets: Optional[List[float]] = None, tags: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create histogram."""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = Histogram(name, buckets, tags)
        return self.histograms[key]
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Make key from name and tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}:{tag_str}"
        return name
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": [
                counter.get().to_dict()
                for counter in self.counters.values()
            ],
            "gauges": [
                gauge.get().to_dict()
                for gauge in self.gauges.values()
            ],
            "histograms": [
                histogram.get()
                for histogram in self.histograms.values()
            ],
        }
    
    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()


# Global collector instance
_global_collector = TelemetryCollector()


def get_collector() -> TelemetryCollector:
    """Get global telemetry collector."""
    return _global_collector



