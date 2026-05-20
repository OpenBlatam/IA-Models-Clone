"""
Prometheus Metrics
Prometheus metrics collection
"""

from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary
import logging

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collection"""
    
    def __init__(self):
        # Counters
        self.video_generation_total = Counter(
            'video_generation_total',
            'Total video generations',
            ['status', 'user_id']
        )
        
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'service']
        )
        
        # Histograms
        self.video_generation_duration = Histogram(
            'video_generation_duration_seconds',
            'Video generation duration',
            ['status'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
        )
        
        # Gauges
        self.active_video_generations = Gauge(
            'active_video_generations',
            'Currently active video generations'
        )
        
        self.queue_size = Gauge(
            'video_queue_size',
            'Size of video generation queue'
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate (0-1)'
        )
        
        # Summaries
        self.video_file_size = Summary(
            'video_file_size_bytes',
            'Video file size',
            ['status']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_video_generation(
        self,
        status: str,
        duration: float,
        user_id: Optional[str] = None,
        file_size: Optional[int] = None
    ):
        """Record video generation metric"""
        self.video_generation_total.labels(status=status, user_id=user_id or 'anonymous').inc()
        self.video_generation_duration.labels(status=status).observe(duration)
        
        if file_size:
            self.video_file_size.labels(status=status).observe(file_size)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """Record API request metric"""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_error(self, error_type: str, service: str):
        """Record error metric"""
        self.errors_total.labels(error_type=error_type, service=service).inc()
    
    def set_active_generations(self, count: int):
        """Set active video generations count"""
        self.active_video_generations.set(count)
    
    def set_queue_size(self, size: int):
        """Set queue size"""
        self.queue_size.set(size)
    
    def set_cache_hit_rate(self, rate: float):
        """Set cache hit rate"""
        self.cache_hit_rate.set(rate)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "video_generations": {
                "total": self.video_generation_total._value.get(),
                "active": self.active_video_generations._value.get(),
            },
            "api_requests": {
                "total": self.api_requests_total._value.get(),
            },
            "errors": {
                "total": self.errors_total._value.get(),
            },
            "queue": {
                "size": self.queue_size._value.get(),
            },
            "cache": {
                "hit_rate": self.cache_hit_rate._value.get(),
            },
        }


_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get Prometheus metrics instance (singleton)"""
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics

