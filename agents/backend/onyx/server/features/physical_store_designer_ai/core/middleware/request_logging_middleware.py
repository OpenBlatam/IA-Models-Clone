"""
Request Logging Middleware

Middleware for request and response logging with metrics.
"""

import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging_config import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests con métricas"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Log request and response with metrics"""
        start_time = time.perf_counter()
        
        # Skip metrics for health checks
        is_health_check = request.url.path in ["/health", "/health/live", "/health/ready", "/metrics"]
        
        if not is_health_check:
            from ..metrics import get_metrics_collector
            metrics = get_metrics_collector()
            metrics.increment("http.requests.total", tags={"method": request.method})
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client": request.client.host if request.client else None
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000
        
        # Track metrics
        if not is_health_check:
            from ..metrics import get_metrics_collector
            metrics = get_metrics_collector()
            metrics.timer("http.request.duration", duration_ms, tags={
                "method": request.method,
                "status": str(response.status_code),
                "path": request.url.path
            })
            metrics.increment("http.requests", tags={
                "method": request.method,
                "status": str(response.status_code)
            })
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2)
            }
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        return response

