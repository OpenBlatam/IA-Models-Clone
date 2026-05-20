"""Prometheus metrics integration"""
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        # Counters
        self.conversions_total = Counter(
            'markdown_conversions_total',
            'Total number of conversions',
            ['format', 'status']
        )
        
        self.requests_total = Counter(
            'markdown_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.errors_total = Counter(
            'markdown_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        # Histograms
        self.conversion_duration = Histogram(
            'markdown_conversion_duration_seconds',
            'Conversion duration in seconds',
            ['format'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.request_duration = Histogram(
            'markdown_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Gauges
        self.active_conversions = Gauge(
            'markdown_active_conversions',
            'Number of active conversions'
        )
        
        self.cache_size = Gauge(
            'markdown_cache_size',
            'Cache size in bytes'
        )
        
        self.queue_size = Gauge(
            'markdown_queue_size',
            'Queue size'
        )
    
    def record_conversion(
        self,
        format: str,
        status: str,
        duration: float
    ):
        """
        Record conversion metric
        
        Args:
            format: Output format
            status: Conversion status (success, failed)
            duration: Duration in seconds
        """
        self.conversions_total.labels(format=format, status=status).inc()
        self.conversion_duration.labels(format=format).observe(duration)
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """
        Record request metric
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Duration in seconds
        """
        status = 'success' if status_code < 400 else 'error'
        self.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_error(self, error_type: str):
        """
        Record error metric
        
        Args:
            error_type: Error type
        """
        self.errors_total.labels(error_type=error_type).inc()
    
    def set_active_conversions(self, count: int):
        """Set active conversions count"""
        self.active_conversions.set(count)
    
    def set_cache_size(self, size: int):
        """Set cache size"""
        self.cache_size.set(size)
    
    def set_queue_size(self, size: int):
        """Set queue size"""
        self.queue_size.set(size)
    
    def get_metrics(self) -> bytes:
        """
        Get Prometheus metrics
        
        Returns:
            Metrics in Prometheus format
        """
        return generate_latest()


# Global metrics
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics"""
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics

