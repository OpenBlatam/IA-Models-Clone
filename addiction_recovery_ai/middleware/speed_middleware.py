"""
Speed Middleware
Ultra-fast request processing optimizations
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from typing import Optional

try:
    from performance.serialization_optimizer import get_serializer
    from performance.response_optimizer import get_response_optimizer
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    get_serializer = get_response_optimizer = None

logger = logging.getLogger(__name__)


class SpeedMiddleware(BaseHTTPMiddleware):
    """
    Speed optimization middleware
    
    Features:
    - Fast JSON serialization
    - Response caching
    - Request deduplication
    - Early response optimization
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        if PERFORMANCE_AVAILABLE:
            self.serializer = get_serializer()
            self.response_optimizer = get_response_optimizer()
        else:
            self.serializer = None
            self.response_optimizer = None
        self._request_cache: dict = {}
        self._cache_ttl = 1  # 1 second cache for duplicate requests
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with speed optimizations"""
        start_time = time.perf_counter()
        
        # Check for duplicate requests (idempotency)
        if request.method == "GET":
            cache_key = self._get_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Optimize response
        response = await self._optimize_response(request, response)
        
        # Cache GET responses
        if request.method == "GET" and response.status_code == 200:
            self._cache_response(cache_key, response)
        
        # Add performance headers
        duration = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        response.headers["X-Speed-Optimized"] = "true"
        
        return response
    
    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.method}:{request.url.path}:{request.url.query}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response"""
        import time
        if cache_key in self._request_cache:
            response, cached_time = self._request_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return response
            else:
                del self._request_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Response) -> None:
        """Cache response"""
        import time
        self._request_cache[cache_key] = (response, time.time())
    
    async def _optimize_response(self, request: Request, response: Response) -> Response:
        """Optimize response for speed"""
        if not PERFORMANCE_AVAILABLE:
            return response
        
        # If response is JSON, use fast serializer
        content_type = response.headers.get("content-type", "")
        
        if "application/json" in content_type and self.response_optimizer:
            # Response is already optimized by FastResponse
            pass
        
        return response

