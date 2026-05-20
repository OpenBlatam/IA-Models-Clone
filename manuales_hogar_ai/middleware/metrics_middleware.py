"""
Metrics Middleware
==================

Implements Prometheus metrics collection.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    http_requests_in_progress = Gauge(
        "http_requests_in_progress",
        "HTTP requests currently in progress",
        ["method", "endpoint"],
    )


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics middleware."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.prometheus_available = PROMETHEUS_AVAILABLE

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection."""
        if not self.prometheus_available:
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        # Increment in-progress gauge
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        # Start timer
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()

            http_request_duration_seconds.labels(
                method=method, endpoint=endpoint
            ).observe(duration)

            return response

        except Exception as e:
            # Record error
            duration = time.time() - start_time
            http_requests_total.labels(
                method=method, endpoint=endpoint, status_code=500
            ).inc()
            http_request_duration_seconds.labels(
                method=method, endpoint=endpoint
            ).observe(duration)
            raise

        finally:
            # Decrement in-progress gauge
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()




