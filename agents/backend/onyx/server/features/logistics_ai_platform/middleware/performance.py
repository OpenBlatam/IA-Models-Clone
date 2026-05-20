"""Performance monitoring middleware"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
from utils.logger import logger
from utils.metrics import get_metrics_collector, record_request_metrics


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring with Prometheus integration"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with performance tracking and metrics collection"""
        start_time = time.time()
        
        # Skip metrics for health and metrics endpoints
        path = str(request.url.path)
        skip_metrics = path in ["/health", "/ready", "/metrics", "/metrics/info"]
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            process_time = time.time() - start_time
            
            # Record Prometheus metrics
            if not skip_metrics:
                try:
                    record_request_metrics(
                        method=request.method,
                        endpoint=path,
                        status_code=response.status_code,
                        duration=process_time
                    )
                except Exception as e:
                    logger.warning(f"Failed to record metrics: {e}")
            
            # Add performance headers
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
            
            # Log slow requests
            if process_time > 1.0:
                logger.warning(
                    f"Slow request: {request.method} {path} took {process_time:.4f}s",
                    extra={
                        "method": request.method,
                        "path": path,
                        "process_time": process_time,
                        "status_code": response.status_code
                    }
                )
            
            return response
        except Exception as e:
            # Record error metrics
            process_time = time.time() - start_time
            if not skip_metrics:
                try:
                    collector = get_metrics_collector()
                    collector.record_error(
                        error_type=type(e).__name__,
                        endpoint=path
                    )
                    record_request_metrics(
                        method=request.method,
                        endpoint=path,
                        status_code=500,
                        duration=process_time
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record error metrics: {metrics_error}")
            
            # Re-raise the exception
            raise








