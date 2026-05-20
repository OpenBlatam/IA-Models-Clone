"""
Logging Middleware
=================

Middleware for comprehensive request/response logging.
"""

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request details and response timing."""
        start_time = time.time()
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        
        # Log request
        logger.info(
            f"Request: {method} {url} - {client_ip} - "
            f"RequestID: {getattr(request.state, 'request_id', 'unknown')}"
        )
        
        if query_params:
            logger.debug(f"Query params: {query_params}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s - "
            f"RequestID: {getattr(request.state, 'request_id', 'unknown')}"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
