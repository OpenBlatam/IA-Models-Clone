"""
Prometheus metrics collection for Logistics AI Platform

This module provides Prometheus metrics collection and export functionality.
"""

import time
from typing import Dict, Any, Optional
from functools import wraps

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    CollectorRegistry = None
    REGISTRY = None

from utils.logger import logger


class MetricsCollector:
    """
    Prometheus metrics collector for Logistics AI Platform
    
    Collects and exposes metrics for:
    - HTTP requests (count, duration)
    - Business operations (quotes, bookings, shipments)
    - System metrics (cache, database)
    - Error rates
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics will be disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.registry = REGISTRY
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'logistics_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration_seconds = Histogram(
            'logistics_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Business metrics
        self.quotes_created_total = Counter(
            'logistics_quotes_created_total',
            'Total quotes created'
        )
        
        self.bookings_created_total = Counter(
            'logistics_bookings_created_total',
            'Total bookings created'
        )
        
        self.shipments_created_total = Counter(
            'logistics_shipments_created_total',
            'Total shipments created'
        )
        
        self.containers_created_total = Counter(
            'logistics_containers_created_total',
            'Total containers created'
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'logistics_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'logistics_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        self.cache_size = Gauge(
            'logistics_cache_size',
            'Current cache size',
            ['cache_type']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'logistics_errors_total',
            'Total errors',
            ['error_type', 'endpoint']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'logistics_active_connections',
            'Number of active connections'
        )
        
        self.background_tasks_queued = Gauge(
            'logistics_background_tasks_queued',
            'Number of queued background tasks'
        )
        
        self.background_tasks_processed = Counter(
            'logistics_background_tasks_processed_total',
            'Total background tasks processed'
        )
        
        logger.info("Prometheus metrics collector initialized")
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record HTTP request metrics
        
        Args:
            method: HTTP method
            endpoint: Endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        if not self.enabled:
            return
        
        try:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        except Exception as e:
            logger.warning(f"Error recording request metrics: {e}")
    
    def record_quote_created(self) -> None:
        """Record quote creation"""
        if self.enabled:
            self.quotes_created_total.inc()
    
    def record_booking_created(self) -> None:
        """Record booking creation"""
        if self.enabled:
            self.bookings_created_total.inc()
    
    def record_shipment_created(self) -> None:
        """Record shipment creation"""
        if self.enabled:
            self.shipments_created_total.inc()
    
    def record_container_created(self) -> None:
        """Record container creation"""
        if self.enabled:
            self.containers_created_total.inc()
    
    def record_cache_hit(self, cache_type: str = "default") -> None:
        """Record cache hit"""
        if self.enabled:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "default") -> None:
        """Record cache miss"""
        if self.enabled:
            self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_cache_size(self, size: int, cache_type: str = "default") -> None:
        """Set cache size metric"""
        if self.enabled:
            self.cache_size.labels(cache_type=cache_type).set(size)
    
    def record_error(self, error_type: str, endpoint: str = "unknown") -> None:
        """Record error"""
        if self.enabled:
            self.errors_total.labels(
                error_type=error_type,
                endpoint=endpoint
            ).inc()
    
    def set_active_connections(self, count: int) -> None:
        """Set active connections count"""
        if self.enabled:
            self.active_connections.set(count)
    
    def set_background_tasks_queued(self, count: int) -> None:
        """Set background tasks queued count"""
        if self.enabled:
            self.background_tasks_queued.set(count)
    
    def record_background_task_processed(self) -> None:
        """Record background task processed"""
        if self.enabled:
            self.background_tasks_processed.inc()
    
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format
        
        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return b"# Prometheus metrics not available\n"
        
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return b"# Error generating metrics\n"
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get metrics as dictionary
        
        Returns:
            Dictionary with metrics summary
        """
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "http_requests": "logistics_http_requests_total",
            "http_duration": "logistics_http_request_duration_seconds",
            "business_metrics": {
                "quotes": "logistics_quotes_created_total",
                "bookings": "logistics_bookings_created_total",
                "shipments": "logistics_shipments_created_total",
                "containers": "logistics_containers_created_total"
            },
            "cache_metrics": {
                "hits": "logistics_cache_hits_total",
                "misses": "logistics_cache_misses_total",
                "size": "logistics_cache_size"
            },
            "error_metrics": "logistics_errors_total",
            "system_metrics": {
                "connections": "logistics_active_connections",
                "background_tasks": "logistics_background_tasks_queued"
            }
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Helper function to record request metrics"""
    get_metrics_collector().record_request(method, endpoint, status_code, duration)


def metrics_middleware_decorator(func):
    """Decorator to automatically record metrics for endpoint functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "UNKNOWN"
        endpoint = func.__name__
        
        # Try to extract method and endpoint from request
        for arg in args:
            if hasattr(arg, 'method'):
                method = arg.method
            if hasattr(arg, 'url'):
                endpoint = str(arg.url.path)
        
        try:
            result = await func(*args, **kwargs)
            status_code = 200
            if hasattr(result, 'status_code'):
                status_code = result.status_code
            
            duration = time.time() - start_time
            record_request_metrics(method, endpoint, status_code, duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            status_code = 500
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            record_request_metrics(method, endpoint, status_code, duration)
            get_metrics_collector().record_error(type(e).__name__, endpoint)
            raise
    
    return wrapper

