"""
Prometheus Metrics Configuration
================================

Exposes metrics for monitoring with Prometheus and Grafana.
"""

import time
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client import make_asgi_app
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

logger = logging.getLogger(__name__)

# HTTP Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
)

# Application Metrics
robot_movements_total = Counter(
    "robot_movements_total",
    "Total robot movements",
    ["status", "robot_brand"]
)

trajectory_optimizations_total = Counter(
    "trajectory_optimizations_total",
    "Total trajectory optimizations",
    ["algorithm", "status"]
)

chat_messages_total = Counter(
    "chat_messages_total",
    "Total chat messages",
    ["status"]
)

# System Metrics
active_connections = Gauge(
    "active_connections",
    "Number of active connections"
)

active_tasks = Gauge(
    "active_tasks",
    "Number of active background tasks",
    ["task_type"]
)

# Performance Metrics
trajectory_optimization_duration = Histogram(
    "trajectory_optimization_duration_seconds",
    "Trajectory optimization duration",
    ["algorithm"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

model_inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration",
    ["model_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0]
)

# Info metric
app_info = Info(
    "app_info",
    "Application information"
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        # Get request size
        request_size = 0
        if hasattr(request, "_body"):
            request_size = len(request._body) if request._body else 0
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            duration = time.time() - start_time
            
            # Get response size
            response_size = 0
            if hasattr(response, "body"):
                response_size = len(response.body) if response.body else 0
            
            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            http_request_size_bytes.labels(
                method=method,
                endpoint=path
            ).observe(request_size)
            
            http_response_size_bytes.labels(
                method=method,
                endpoint=path
            ).observe(response_size)
            
            return response
            
        except Exception as e:
            status_code = 500
            duration = time.time() - start_time
            
            # Record error metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            raise


def setup_prometheus(app, app_name: str = "robot-movement-ai", app_version: str = "1.0.0"):
    """Setup Prometheus metrics."""
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Set app info
    app_info.info({
        "name": app_name,
        "version": app_version,
    })
    
    # Create metrics endpoint
    metrics_app = make_asgi_app(REGISTRY)
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain"
        )
    
    logger.info("Prometheus metrics configured")
    
    return app















