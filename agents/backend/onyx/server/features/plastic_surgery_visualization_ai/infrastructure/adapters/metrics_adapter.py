"""Adapter for metrics collector to implement IMetricsCollector interface."""

from core.interfaces import IMetricsCollector
from utils.metrics import metrics_collector


class MetricsCollectorAdapter(IMetricsCollector):
    """Adapter for metrics collector."""
    
    def increment(self, metric_name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        metrics_collector.increment(metric_name, value)
    
    def record_timing(self, metric_name: str, duration: float) -> None:
        """Record a timing metric."""
        metrics_collector.record_timing(metric_name, duration)
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """Set a gauge metric."""
        metrics_collector.set_gauge(metric_name, value)

