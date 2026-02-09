"""
Performance monitoring middleware
"""

import time
import logging
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API performance metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(
            f"Request processed: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        # Warn on slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request detected: {request.url.path} took {process_time:.2f}s",
                extra={
                    "path": request.url.path,
                    "process_time": process_time
                }
            )
        
        return response

