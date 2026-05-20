"""
PDF Variantes - Prometheus Metrics
"""

import time
from typing import Dict, Any, Optional
from functools import wraps
from fastapi import Request
import logging

logger = logging.getLogger(__name__)

# Try to import Prometheus (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus not available. Install with: pip install prometheus-client")


class MetricsCollector:
    """Prometheus metrics collector"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            self._setup_metrics()
        else:
            self._create_dummy_metrics()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=(0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0)
        )
        
        # Business metrics
        self.pdf_uploads_total = Counter(
            'pdf_uploads_total',
            'Total PDF uploads',
            ['status']
        )
        
        self.variants_generated_total = Counter(
            'variants_generated_total',
            'Total variants generated'
        )
        
        self.topics_extracted_total = Counter(
            'topics_extracted_total',
            'Total topics extracted'
        )
        
        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def _create_dummy_metrics(self):
        """Create dummy metrics when Prometheus is not available"""
        class DummyMetric:
            def inc(self, *args, **kwargs): pass
            def observe(self, *args, **kwargs): pass
            def set(self, *args, **kwargs): pass
        
        self.http_requests_total = DummyMetric()
        self.http_request_duration = DummyMetric()
        self.pdf_uploads_total = DummyMetric()
        self.variants_generated_total = DummyMetric()
        self.topics_extracted_total = DummyMetric()
        self.active_connections = DummyMetric()
        self.cache_hits_total = DummyMetric()
        self.cache_misses_total = DummyMetric()
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if self.enabled:
            self.http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_pdf_upload(self, status: str = "success"):
        """Record PDF upload metric"""
        if self.enabled:
            self.pdf_uploads_total.labels(status=status).inc()
    
    def record_variant_generation(self):
        """Record variant generation metric"""
        if self.enabled:
            self.variants_generated_total.inc()
    
    def record_topic_extraction(self):
        """Record topic extraction metric"""
        if self.enabled:
            self.topics_extracted_total.inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        if self.enabled:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        if self.enabled:
            self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_active_connections(self, count: int):
        """Set active connections gauge"""
        if self.enabled:
            self.active_connections.set(count)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics as bytes"""
        if self.enabled:
            return generate_latest()
        return b"# Prometheus not available\n"


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def metrics_middleware(request: Request, call_next):
    """Middleware to collect HTTP metrics"""
    collector = get_metrics_collector()
    
    start_time = time.time()
    
    async def process():
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        method = request.method
        endpoint = request.url.path.split('?')[0]  # Remove query params
        status = response.status_code
        
        collector.record_http_request(method, endpoint, status, duration)
        
        return response
    
    return process()






