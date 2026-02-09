"""
Enhanced Middleware

Additional middleware for API enhancement.
"""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to add timing information to responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Add timing headers to response."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to requests."""
    
    async def dispatch(self, request: Request, call_next):
        """Add request ID if not present."""
        import uuid
        
        if "X-Request-ID" not in request.headers:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
        else:
            request.state.request_id = request.headers["X-Request-ID"]
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response



