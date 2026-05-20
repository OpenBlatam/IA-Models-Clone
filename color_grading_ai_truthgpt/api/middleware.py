"""
Middleware for Color Grading API
==================================

Rate limiting, authentication, and request logging.
"""

import time
import logging
from typing import Callable, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier (IP address, API key, etc.)
            
        Returns:
            True if allowed
        """
        async with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests
            self._requests[client_id] = [
                req_time for req_time in self._requests[client_id]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self._requests[client_id]) >= self.max_requests:
                return False
            
            # Add current request
            self._requests[client_id].append(now)
            return True
    
    async def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        async with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            self._requests[client_id] = [
                req_time for req_time in self._requests[client_id]
                if req_time > window_start
            ]
            
            return max(0, self.max_requests - len(self._requests[client_id]))


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


async def rate_limit_middleware(request: Request, call_next: Callable):
    """
    Rate limiting middleware.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint
        
    Returns:
        Response
    """
    # Get client identifier (IP address)
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not await rate_limiter.is_allowed(client_id):
        remaining = await rate_limiter.get_remaining(client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "retry_after": 60,
                "remaining": remaining
            },
            headers={
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "Retry-After": "60"
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    remaining = await rate_limiter.get_remaining(client_id)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return response


async def request_logging_middleware(request: Request, call_next: Callable):
    """
    Request logging middleware.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint
        
    Returns:
        Response
    """
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"in {duration:.3f}s for {request.method} {request.url.path}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    
    return response


async def error_handling_middleware(request: Request, call_next: Callable):
    """
    Error handling middleware.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint
        
    Returns:
        Response
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": str(e) if logger.level == logging.DEBUG else "An error occurred"
            }
        )




