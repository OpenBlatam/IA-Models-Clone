"""
Metrics Exporter
Export metrics for Prometheus and other monitoring systems
"""

from typing import Dict, Any, Optional
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and export application metrics"""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.timers: Dict[str, list] = defaultdict(list)
    
    def increment(self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        key = self._build_key(metric_name, labels)
        self.counters[key] += value
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        key = self._build_key(metric_name, labels)
        self.gauges[key] = value
    
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        key = self._build_key(metric_name, labels)
        self.histograms[key].append(value)
    
    def start_timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> 'Timer':
        """Start a timer"""
        return Timer(metric_name, labels, self)
    
    def _build_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build metric key with labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{metric_name}{{{label_str}}}"
        return metric_name
    
    def get_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms (simplified)
        for key, values in self.histograms.items():
            if values:
                lines.append(f"# TYPE {key.split('{')[0]} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")
        
        # Timers (as histograms)
        for key, values in self.timers.items():
            if values:
                lines.append(f"# TYPE {key.split('{')[0]} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")
        
        return "\n".join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                }
                for k, v in self.histograms.items()
            },
            "timers": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                }
                for k, v in self.timers.items()
            },
        }
    
    def reset(self):
        """Reset all metrics"""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]], collector: MetricsCollector):
        self.metric_name = metric_name
        self.labels = labels
        self.collector = collector
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            key = self.collector._build_key(self.metric_name, self.labels)
            self.collector.timers[key].append(duration)
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            key = self.collector._build_key(self.metric_name, self.labels)
            self.collector.timers[key].append(duration)


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics_collector















