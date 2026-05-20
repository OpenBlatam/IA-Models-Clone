from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            from fastapi.responses import JSONResponse
from typing import Any, List, Dict, Optional
import asyncio
"""
API Middleware
=============

Custom middleware for the FastAPI application.
"""


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response logging"""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
        self.rate_limit_store = {}
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 100
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make request"""
        now = time.time()
        
        # Clean old entries
        self.rate_limit_store = {
            client: timestamps for client, timestamps in self.rate_limit_store.items()
            if any(ts > now - self.rate_limit_window for ts in timestamps)
        }
        
        # Get client requests
        client_requests = self.rate_limit_store.get(client_id, [])
        client_requests = [ts for ts in client_requests if ts > now - self.rate_limit_window]
        
        # Check if limit exceeded
        if len(client_requests) >= self.max_requests_per_window:
            return False
        
        # Add current request
        client_requests.append(now)
        self.rate_limit_store[client_id] = client_requests
        
        return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_id = request.client.host if request.client else "unknown"
        
        if not self.is_allowed(client_id):
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "timestamp": time.time()
                }
            )
        
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for additional security headers"""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Error handling middleware"""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": time.time()
                }
            ) 