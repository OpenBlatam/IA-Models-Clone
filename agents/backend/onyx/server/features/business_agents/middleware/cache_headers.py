"""Cache headers middleware for HTTP responses."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Dict, Set, Optional

class CacheHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add cache control headers based on route patterns."""
    
    def __init__(
        self,
        app,
        cache_rules: Optional[Dict[str, str]] = None,
        default_cache: str = "no-cache, no-store, must-revalidate"
    ):
        super().__init__(app)
        self.cache_rules = cache_rules or {}
        self.default_cache = default_cache
    
    async def dispatch(self, request: Request, call_next):
        """Add cache headers to response."""
        response = await call_next(request)
        
        # Check if route matches any cache rule
        path = request.url.path
        
        cache_header = self.default_cache
        for pattern, cache_value in self.cache_rules.items():
            if pattern in path or path.startswith(pattern):
                cache_header = cache_value
                break
        
        # Add cache headers
        response.headers["Cache-Control"] = cache_header
        response.headers["Pragma"] = "no-cache" if "no-cache" in cache_header else ""
        
        # Add ETag header for cacheable resources
        if "public" in cache_header or "private" in cache_header:
            import hashlib
            etag = hashlib.md5(str(response.body).encode()).hexdigest()
            response.headers["ETag"] = f'"{etag}"'
        
        return response


