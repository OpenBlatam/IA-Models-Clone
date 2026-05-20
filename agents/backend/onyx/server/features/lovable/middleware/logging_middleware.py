"""
Logging middleware for request/response logging with performance metrics.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time
import logging
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses with performance tracking."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response with performance metrics."""
        start_time = time.time()
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] Request: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
                "query_params": dict(request.query_params) if request.query_params else None
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            logger.error(
                f"[{request_id}] Request failed: {e}",
                exc_info=True,
                extra={"request_id": request_id, "method": request.method, "path": request.url.path}
            )
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        try:
            from ..utils.performance_metrics import get_metrics
            metrics = get_metrics()
            metrics.record_request(
                method=request.method,
                path=request.url.path,
                duration=duration,
                status_code=status_code
            )
        except Exception:
            # Metrics not critical, continue if it fails
            pass
        
        # Log response
        log_level = logger.warning if status_code >= 400 else logger.info
        log_level(
            f"[{request_id}] Response: {request.method} {request.url.path} - {status_code} ({duration:.3f}s)",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration": duration
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response




