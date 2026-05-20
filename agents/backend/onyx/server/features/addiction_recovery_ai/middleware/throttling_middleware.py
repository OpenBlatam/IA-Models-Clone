"""
Throttling Middleware
Intelligent request throttling
"""

import logging
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from scalability.throttling import get_throttler, ThrottleConfig

logger = logging.getLogger(__name__)


class ThrottlingMiddleware(BaseHTTPMiddleware):
    """
    Intelligent throttling middleware
    
    Features:
    - Token bucket algorithm
    - Priority-based throttling
    - Per-endpoint configuration
    - Adaptive rate limiting
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.throttler = get_throttler()
        self._setup_default_configs()
    
    def _setup_default_configs(self) -> None:
        """Setup default throttling configurations"""
        # Default config for all endpoints
        default_config = ThrottleConfig(
            requests_per_second=10,
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=20
        )
        self.throttler.configure("default", default_config)
        
        # High-traffic endpoints
        high_traffic_config = ThrottleConfig(
            requests_per_second=50,
            requests_per_minute=300,
            requests_per_hour=5000,
            burst_size=100
        )
        self.throttler.configure("/recovery/health", high_traffic_config)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with throttling"""
        endpoint = request.url.path
        user_id = request.headers.get("x-user-id")
        priority = request.headers.get("x-priority", "medium")
        
        # Get endpoint-specific config or use default
        config_key = endpoint if endpoint in self.throttler._configs else "default"
        
        # Check throttling
        is_allowed, info = self.throttler.is_allowed(config_key, user_id, priority)
        
        if not is_allowed:
            retry_after = info.get("retry_after", 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.throttler._configs[config_key].requests_per_second),
                    "X-RateLimit-Remaining": str(int(info.get("tokens_available", 0)))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add throttling headers
        status_info = self.throttler.get_throttle_status(config_key)
        response.headers["X-RateLimit-Limit"] = str(status_info.get("rate", 0))
        response.headers["X-RateLimit-Remaining"] = str(int(status_info.get("tokens", 0)))
        
        return response















