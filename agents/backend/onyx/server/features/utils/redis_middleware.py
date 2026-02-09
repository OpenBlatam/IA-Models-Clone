from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json
import hashlib
import time
import logging
from .redis_utils import RedisUtils
from .redis_config import get_config
from fastapi import FastAPI
from .redis_middleware import RedisMiddleware
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Middleware - Onyx Integration
Middleware for Redis caching in Onyx.
"""

logger = logging.getLogger(__name__)

class RedisMiddleware(BaseHTTPMiddleware):
    """Middleware for Redis caching in Onyx."""
    
    def __init__(
        self,
        app: ASGIApp,
        config: Any = None
    ):
        """Initialize Redis middleware."""
        super().__init__(app)
        self.config = config or {}
        self.redis_utils = RedisUtils(get_config())
        # Support both dict and Pydantic model
        if isinstance(self.config, dict):
            self.cache_ttl = self.config.get("cache_ttl", 3600)
            self.exclude_paths = self.config.get("exclude_paths", [])
            self.include_paths = self.config.get("include_paths", [])
            self.cache_headers = self.config.get("cache_headers", True)
        else:
            self.cache_ttl = getattr(self.config, "cache_ttl", 3600)
            self.exclude_paths = getattr(self.config, "exclude_paths", [])
            self.include_paths = getattr(self.config, "include_paths", [])
            self.cache_headers = getattr(self.config, "cache_headers", True)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process the request and response with Redis caching."""
        # Skip caching for excluded paths
        if self._should_skip_caching(request.url.path):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get cached response
        cached_response = self.redis_utils.get_cached_data(
            prefix="response",
            identifier=cache_key
        )
        
        if cached_response:
            logger.debug(f"Cache hit for {request.url.path}")
            return self._create_cached_response(cached_response)
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache response if it's cacheable
        if self._is_cacheable_response(response):
            await self._cache_response(response, cache_key)
        
        return response
    
    def _should_skip_caching(self, path: str) -> bool:
        """Check if the path should be skipped for caching."""
        # Skip if path is in exclude_paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return True
        
        # Skip if include_paths is set and path is not included
        if self.include_paths and not any(path.startswith(included) for included in self.include_paths):
            return True
        
        return False
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate a cache key for the request."""
        # Combine request components
        key_parts = [
            request.method,
            request.url.path,
            str(request.query_params),
            request.headers.get("authorization", ""),
            request.headers.get("accept", "")
        ]
        
        # Add body if present
        if request.method in ["POST", "PUT", "PATCH"]:
            body = request.body()
            if body:
                key_parts.append(body.decode())
        
        # Create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cacheable_response(self, response: Response) -> bool:
        """Check if the response should be cached."""
        # Don't cache error responses
        if response.status_code >= 400:
            return False
        
        # Don't cache streaming responses
        if "transfer-encoding" in response.headers:
            return False
        
        # Check cache control headers
        cache_control = response.headers.get("cache-control", "")
        if "no-store" in cache_control or "private" in cache_control:
            return False
        
        return True
    
    async def _cache_response(self, response: Response, cache_key: str) -> None:
        """Cache the response in Redis."""
        try:
            # Get response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Create cache data
            cache_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body.decode(),
                "timestamp": time.time()
            }
            
            # Cache response
            self.redis_utils.cache_data(
                data=cache_data,
                prefix="response",
                identifier=cache_key,
                expire=self.cache_ttl
            )
            
            logger.debug(f"Cached response for {cache_key}")
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def _create_cached_response(self, cached_data: Dict[str, Any]) -> Response:
        """Create a response from cached data."""
        # Create response
        response = Response(
            content=cached_data["body"],
            status_code=cached_data["status_code"],
            headers=cached_data["headers"]
        )
        
        # Add cache headers if enabled
        if self.cache_headers:
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Timestamp"] = str(cached_data["timestamp"])
        
        return response

# Example usage:
"""

app = FastAPI()

# Add Redis middleware
app.add_middleware(
    RedisMiddleware,
    config={
        "cache_ttl": 3600,  # 1 hour
        "exclude_paths": ["/admin", "/api/v1/auth"],
        "include_paths": ["/api/v1"],
        "cache_headers": True
    }
)

@app.get("/api/v1/users")
async def get_users():
    
    """get_users function."""
# This response will be cached
    return {"users": [...]}

@app.post("/api/v1/users")
async def create_user():
    
    """create_user function."""
# This response won't be cached
    return {"user": {...}}
""" 