"""
Rate Limit Middleware
=====================

FastAPI middleware for rate limiting.
"""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Callable, Optional

from services.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.
    
    Features:
    - Automatic rate limiting
    - Client identification
    - Customizable limits
    - Rate limit headers in response
    """
    
    def __init__(
        self,
        app,
        default_limit: int = 100,
        default_window: float = 60.0,
        client_identifier: Optional[Callable[[Request], str]] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            default_limit: Default requests per window
            default_window: Default window size in seconds
            client_identifier: Optional function to extract client ID from request
        """
        super().__init__(app)
        self.rate_limiter = get_rate_limiter()
        self.default_limit = default_limit
        self.default_window = default_window
        self.client_identifier = client_identifier or self._default_client_identifier
    
    def _default_client_identifier(self, request: Request) -> str:
        """
        Default client identifier function.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier string
        """
        # Try to get client ID from header
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return client_id
        
        # Try to get from query parameter
        client_id = request.query_params.get("client_id")
        if client_id:
            return client_id
        
        # Fallback to IP address
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response with rate limit headers
        """
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Get client identifier
        client_id = self.client_identifier(request)
        
        # Check rate limit
        is_allowed, info = self.rate_limiter.is_allowed(
            client_id=client_id,
            limit=self.default_limit,
            window=self.default_window
        )
        
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(info["limit"]),
            "X-RateLimit-Remaining": str(info["remaining"]),
            "X-RateLimit-Reset": str(int(info["reset_in"]))
        }
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {info['limit']} per {info['window_seconds']}s",
                    "retry_after": int(info["reset_in"])
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response

