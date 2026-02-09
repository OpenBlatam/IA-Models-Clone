"""
Prometheus Metrics for Piel Mejorador AI SAM3
=============================================

Prometheus metrics exporter for monitoring.
"""

import time
import logging
from typing import Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics collector.
    
    Features:
    - Counter metrics
    - Histogram metrics
    - Gauge metrics
    - Summary metrics
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment
            labels: Optional labels (for compatibility)
        """
        key = self._format_key(name, labels)
        self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        key = self._format_key(name, labels)
        self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Observe a histogram value.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        key = self._format_key(name, labels)
        self._histograms[key].append(value)
    
    def start_timer(self, name: str, labels: Dict[str, str] = None) -> str:
        """
        Start a timer.
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Returns:
            Timer ID
        """
        timer_id = self._format_key(name, labels)
        self._start_times[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str):
        """
        Stop a timer and record duration.
        
        Args:
            timer_id: Timer ID from start_timer
        """
        if timer_id in self._start_times:
            duration = time.time() - self._start_times[timer_id]
            # Extract metric name from timer_id
            metric_name = timer_id.split("{")[0] if "{" in timer_id else timer_id
            self.observe_histogram(f"{metric_name}_duration_seconds", duration)
            del self._start_times[timer_id]
    
    def _format_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Format metric key with labels."""
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus metrics string
        """
        lines = []
        
        # Counters
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms (simplified - just count and sum)
        for key, values in self._histograms.items():
            if values:
                metric_name = key.split("{")[0]
                lines.append(f"# TYPE {metric_name} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")
        
        return "\n".join(lines) + "\n"
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                }
                for k, v in self._histograms.items()
            }
        }




