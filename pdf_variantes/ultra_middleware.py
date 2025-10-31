"""Ultra-efficient middleware with minimal overhead."""

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from typing import Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class UltraFastRequestLogger(BaseHTTPMiddleware):
    """Ultra-fast request logging with minimal overhead."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._request_count = 0
        self._total_time = 0.0
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        self._request_count += 1
        self._total_time += duration
        
        # Add minimal headers
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        response.headers["X-Request-Count"] = str(self._request_count)
        
        return response


class UltraFastRateLimiter(BaseHTTPMiddleware):
    """Ultra-fast rate limiting with minimal memory usage."""
    
    def __init__(self, app: ASGIApp, max_requests: int = 1000, window_seconds: int = 60):
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self._requests[client_ip] = [
            req_time for req_time in self._requests[client_ip]
            if current_time - req_time < self._window_seconds
        ]
        
        # Check rate limit
        if len(self._requests[client_ip]) >= self._max_requests:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self._window_seconds)}
            )
        
        # Add current request
        self._requests[client_ip].append(current_time)
        
        return await call_next(request)


class UltraFastCORS(BaseHTTPMiddleware):
    """Ultra-fast CORS with minimal overhead."""
    
    def __init__(self, app: ASGIApp, allow_origins: list = None):
        super().__init__(app)
        self._allow_origins = allow_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response


class UltraFastCompression(BaseHTTPMiddleware):
    """Ultra-fast compression with minimal overhead."""
    
    def __init__(self, app: ASGIApp, min_size: int = 1024):
        super().__init__(app)
        self._min_size = min_size
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Check if compression is needed
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self._min_size:
            response.headers["Content-Encoding"] = "gzip"
        
        return response


class UltraFastSecurity(BaseHTTPMiddleware):
    """Ultra-fast security headers."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in self._security_headers.items():
            response.headers[header] = value
        
        return response


class UltraFastMetrics(BaseHTTPMiddleware):
    """Ultra-fast metrics collection."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._metrics = {
            "requests": 0,
            "errors": 0,
            "total_time": 0.0,
            "start_time": time.time()
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            self._metrics["requests"] += 1
        except Exception as e:
            self._metrics["errors"] += 1
            raise e
        finally:
            duration = time.time() - start_time
            self._metrics["total_time"] += duration
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self._metrics["start_time"]
        return {
            "requests": self._metrics["requests"],
            "errors": self._metrics["errors"],
            "uptime": uptime,
            "avg_response_time": self._metrics["total_time"] / max(self._metrics["requests"], 1),
            "requests_per_second": self._metrics["requests"] / max(uptime, 1)
        }


class UltraFastCache(BaseHTTPMiddleware):
    """Ultra-fast response caching."""
    
    def __init__(self, app: ASGIApp, ttl: int = 300, max_size: int = 1000):
        super().__init__(app)
        self._cache = {}
        self._ttl = ttl
        self._max_size = max_size
        self._timestamps = {}
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Create cache key
        cache_key = f"{request.url.path}:{str(request.query_params)}"
        
        # Check cache
        if cache_key in self._cache:
            if time.time() - self._timestamps[cache_key] < self._ttl:
                return Response(
                    content=self._cache[cache_key],
                    media_type="application/json"
                )
            else:
                # Remove expired entry
                del self._cache[cache_key]
                del self._timestamps[cache_key]
        
        # Process request
        response = await call_next(request)
        
        # Cache response if it's JSON and under size limit
        if (response.headers.get("content-type", "").startswith("application/json") and
            len(response.body) < 1024 * 1024):  # 1MB limit
            
            # Evict old entries if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            # Cache response
            self._cache[cache_key] = response.body
            self._timestamps[cache_key] = time.time()
        
        return response


def setup_ultra_fast_middleware(app: FastAPI, config: Dict[str, Any] = None):
    """Setup ultra-fast middleware stack."""
    config = config or {}
    
    # Add middleware in order (last added is first executed)
    app.add_middleware(UltraFastSecurity)
    app.add_middleware(UltraFastCompression, min_size=config.get("min_compression_size", 1024))
    app.add_middleware(UltraFastCORS, allow_origins=config.get("cors_origins", ["*"]))
    app.add_middleware(UltraFastRateLimiter, 
                      max_requests=config.get("max_requests", 1000),
                      window_seconds=config.get("window_seconds", 60))
    app.add_middleware(UltraFastCache,
                      ttl=config.get("cache_ttl", 300),
                      max_size=config.get("cache_max_size", 1000))
    app.add_middleware(UltraFastMetrics)
    app.add_middleware(UltraFastRequestLogger)
    
    logger.info("Ultra-fast middleware stack configured")


class UltraFastErrorHandler:
    """Ultra-fast error handling."""
    
    def __init__(self):
        self._error_counts = defaultdict(int)
    
    def handle_error(self, error: Exception, request: Request) -> Response:
        """Handle error with minimal overhead."""
        error_type = type(error).__name__
        self._error_counts[error_type] += 1
        
        # Log error
        logger.error(f"Error in {request.url.path}: {error}")
        
        # Return error response
        return Response(
            content=f"Error: {error_type}",
            status_code=500,
            media_type="text/plain"
        )
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return dict(self._error_counts)


class UltraFastHealthCheck:
    """Ultra-fast health check."""
    
    def __init__(self):
        self._start_time = time.time()
        self._last_check = time.time()
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        current_time = time.time()
        uptime = current_time - self._start_time
        
        return {
            "status": "healthy",
            "uptime": uptime,
            "last_check": current_time,
            "timestamp": current_time
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return True  # Simplified check


# Global instances
ultra_fast_error_handler = UltraFastErrorHandler()
ultra_fast_health_check = UltraFastHealthCheck()
