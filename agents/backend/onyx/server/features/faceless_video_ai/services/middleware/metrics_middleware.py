"""
Metrics Middleware
Record metrics for all requests
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
from ..metrics import get_prometheus_metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to record metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        metrics = get_prometheus_metrics()
        metrics.record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response

