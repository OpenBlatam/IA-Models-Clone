"""Logging middleware for request/response logging."""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import get_logger, log_request
from config.settings import settings

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        start_time = time.time()
        
        # Skip logging for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else None
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log request
            log_request(
                logger,
                method=method,
                path=path,
                status_code=response.status_code,
                duration=duration,
                client_ip=client_ip
            )
            
            # Add performance header
            response.headers["X-Process-Time"] = str(round(duration, 3))
            
            # Log slow requests
            if settings.log_slow_requests and duration > settings.slow_request_threshold:
                logger.warning(
                    "slow_request",
                    method=method,
                    path=path,
                    duration=duration,
                    threshold=settings.slow_request_threshold
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "request_failed",
                method=method,
                path=path,
                error=str(e),
                duration=duration,
                client_ip=client_ip,
                exc_info=True
            )
            raise

