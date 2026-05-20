"""
Prometheus Monitoring Middleware for metrics collection
"""

import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_openmetrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus not available. Install with: pip install prometheus-client")

# Export for use in main.py
__all__ = ["MonitoringMiddleware", "get_metrics_endpoint", "PROMETHEUS_AVAILABLE"]


# Metrics (only create if Prometheus is available)
if PROMETHEUS_AVAILABLE:
    # HTTP request metrics
    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"]
    )
    
    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    )
    
    http_request_size_bytes = Histogram(
        "http_request_size_bytes",
        "HTTP request size in bytes",
        ["method", "endpoint"],
        buckets=(100, 1000, 10000, 100000, 1000000)
    )
    
    http_response_size_bytes = Histogram(
        "http_response_size_bytes",
        "HTTP response size in bytes",
        ["method", "endpoint"],
        buckets=(100, 1000, 10000, 100000, 1000000)
    )
    
    # Active requests gauge
    active_requests = Gauge(
        "active_requests",
        "Number of active requests"
    )
else:
    # Dummy metrics for when Prometheus is not available
    http_requests_total = None
    http_request_duration_seconds = None
    http_request_size_bytes = None
    http_response_size_bytes = None
    active_requests = None


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for Prometheus metrics collection.
    Falls back to simple logging if Prometheus is not available.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection"""
        start_time = time.time()
        
        # Get endpoint path (normalize for metrics)
        endpoint = request.url.path
        method = request.method
        
        # Increment active requests
        if active_requests:
            active_requests.inc()
        
        # Measure request size
        request_size = 0
        if hasattr(request, "_body"):
            request_size = len(request._body) if request._body else 0
        
        if http_request_size_bytes:
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Record metrics
            if http_requests_total:
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
            
            if http_request_duration_seconds:
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
            
            # Measure response size
            response_size = 0
            if hasattr(response, "body"):
                response_size = len(response.body) if response.body else 0
            
            if http_response_size_bytes:
                http_response_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
            # Add performance headers
            response.headers["X-Process-Time"] = f"{duration:.4f}"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            status_code = 500
            
            # Record error metrics
            if http_requests_total:
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
            
            if http_request_duration_seconds:
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
            
            logger.error(
                f"Request error: {method} {endpoint} - {e} - Duration: {duration:.3f}s",
                exc_info=True
            )
            raise
            
        finally:
            # Decrement active requests
            if active_requests:
                active_requests.dec()


def get_metrics_endpoint():
    """Get FastAPI endpoint function for /metrics"""
    if not PROMETHEUS_AVAILABLE:
        async def metrics_endpoint():
            return {"error": "Prometheus not available"}
        return metrics_endpoint
    
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    return metrics_endpoint

