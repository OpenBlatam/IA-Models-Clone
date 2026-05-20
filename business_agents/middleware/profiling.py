"""Performance profiling middleware."""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware to profile request processing time and log slow requests."""
    
    def __init__(
        self,
        app,
        slow_request_threshold: float = 1.0,
        log_slow_requests: bool = True
    ):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.log_slow_requests = log_slow_requests
    
    async def dispatch(self, request: Request, call_next):
        """Measure request processing time."""
        start_time = time.perf_counter()
        
        # Add profiling header
        request.state.profiling_start = start_time
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        
        # Add timing headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-Id"] = request.headers.get("X-Request-ID", "N/A")
        
        # Log slow requests
        if self.log_slow_requests and process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"took {process_time:.4f}s (threshold: {self.slow_request_threshold}s)",
                extra={
                    "request_id": request.headers.get("X-Request-ID"),
                    "method": request.method,
                    "path": str(request.url.path),
                    "process_time": process_time,
                    "status_code": response.status_code
                }
            )
        
        return response


