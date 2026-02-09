from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import functools
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.responses import StreamingResponse
import structlog
import redis.asyncio as redis
            import gzip
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Performance Optimization Middleware for HeyGen AI API
Caching, compression, rate limiting, and performance monitoring.
"""



logger = structlog.get_logger()

# =============================================================================
# Performance Metrics
# =============================================================================

class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self) -> Any:
        self.request_times: deque = deque(maxlen=1000)
        self.response_sizes: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_performance: Dict[str, List[float]] = defaultdict(list)
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.slow_requests: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)
    
    def record_request(self, duration: float, response_size: Optional[int] = None, endpoint: str = ""):
        """Record request performance metrics."""
        self.request_times.append(duration)
        
        if response_size:
            self.response_sizes.append(response_size)
        
        if endpoint:
            self.endpoint_performance[endpoint].append(duration)
            # Keep only last 100 measurements per endpoint
            if len(self.endpoint_performance[endpoint]) > 100:
                self.endpoint_performance[endpoint] = self.endpoint_performance[endpoint][-100:]
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        self.error_counts[error_type] += 1
    
    def record_cache_hit(self) -> Any:
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> Any:
        """Record cache miss."""
        self.cache_misses += 1
    
    def record_slow_request(self, request_data: Dict[str, Any]):
        """Record slow request."""
        self.slow_requests.append(request_data)
        # Keep only last 100 slow requests
        if len(self.slow_requests) > 100:
            self.slow_requests = self.slow_requests[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.request_times:
            return {"error": "No data available"}
        
        request_times = list(self.request_times)
        response_sizes = list(self.response_sizes)
        
        stats = {
            "total_requests": len(request_times),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "average_response_time_ms": sum(request_times) / len(request_times),
            "min_response_time_ms": min(request_times),
            "max_response_time_ms": max(request_times),
            "p95_response_time_ms": self._percentile(request_times, 95),
            "p99_response_time_ms": self._percentile(request_times, 99),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_counts": dict(self.error_counts),
            "slow_requests_count": len(self.slow_requests),
        }
        
        if response_sizes:
            stats.update({
                "average_response_size_bytes": sum(response_sizes) / len(response_sizes),
                "total_response_size_bytes": sum(response_sizes),
            })
        
        return stats
    
    def get_endpoint_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by endpoint."""
        endpoint_stats = {}
        
        for endpoint, times in self.endpoint_performance.items():
            if times:
                endpoint_stats[endpoint] = {
                    "count": len(times),
                    "average_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": self._percentile(times, 95),
                    "p99_ms": self._percentile(times, 99),
                }
        
        return endpoint_stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

# =============================================================================
# Caching System
# =============================================================================

class CacheManager:
    """Cache management system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def generate_cache_key(self, request: Request, include_headers: bool = False) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items())),
        ]
        
        if include_headers:
            # Include relevant headers in cache key
            relevant_headers = ["authorization", "content-type", "accept"]
            header_values = []
            for header in relevant_headers:
                if header in request.headers:
                    header_values.append(f"{header}:{request.headers[header]}")
            key_parts.append(str(sorted(header_values)))
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
            
            # Try memory cache
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if datetime.now(timezone.utc) < expiry:
                    self.cache_stats["hits"] += 1
                    return value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            self.cache_stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache."""
        try:
            # Set in Redis
            if self.redis_client:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            
            # Set in memory cache
            expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            self.memory_cache[key] = (value, expiry)
            
            return True
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    return len(keys)
            
            # Invalidate memory cache entries
            invalidated = 0
            keys_to_remove = [key for key in self.memory_cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
            
            return invalidated
            
        except Exception as e:
            logger.error("Cache invalidation error", pattern=pattern, error=str(e))
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
        }

# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Rate limiting system."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_limit: int = 100,
        default_window: int = 60
    ):
        
    """__init__ function."""
self.redis_client = redis_client
        self.default_limit = default_limit
        self.default_window = default_window
        self.memory_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.limit_configs: Dict[str, Dict[str, int]] = {}
    
    def set_limit(self, key: str, limit: int, window: int):
        """Set rate limit for specific key."""
        self.limit_configs[key] = {"limit": limit, "window": window}
    
    def get_client_key(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get from headers first
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return f"client:{client_id}"
        
        # Use IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def is_allowed(self, key: str, client_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        try:
            config = self.limit_configs.get(key, {
                "limit": self.default_limit,
                "window": self.default_window
            })
            
            current_time = time.time()
            window_start = current_time - config["window"]
            
            # Try Redis first
            if self.redis_client:
                redis_key = f"rate_limit:{key}:{client_key}"
                
                # Get current requests in window
                requests = await self.redis_client.zrangebyscore(
                    redis_key, window_start, current_time
                )
                
                # Remove expired entries
                await self.redis_client.zremrangebyscore(
                    redis_key, 0, window_start - 1
                )
                
                current_count = len(requests)
                
                if current_count < config["limit"]:
                    # Add current request
                    await self.redis_client.zadd(redis_key, {str(current_time): current_time})
                    await self.redis_client.expire(redis_key, config["window"])
                    
                    return True, {
                        "limit": config["limit"],
                        "remaining": config["limit"] - current_count - 1,
                        "reset_time": current_time + config["window"],
                        "window": config["window"]
                    }
                else:
                    return False, {
                        "limit": config["limit"],
                        "remaining": 0,
                        "reset_time": current_time + config["window"],
                        "window": config["window"]
                    }
            
            # Fallback to memory-based rate limiting
            requests = self.memory_limits[f"{key}:{client_key}"]
            
            # Remove expired requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            current_count = len(requests)
            
            if current_count < config["limit"]:
                requests.append(current_time)
                return True, {
                    "limit": config["limit"],
                    "remaining": config["limit"] - current_count - 1,
                    "reset_time": current_time + config["window"],
                    "window": config["window"]
                }
            else:
                return False, {
                    "limit": config["limit"],
                    "remaining": 0,
                    "reset_time": current_time + config["window"],
                    "window": config["window"]
                }
                
        except Exception as e:
            logger.error("Rate limiting error", key=key, error=str(e))
            # Allow request on error
            return True, {"error": "Rate limiting unavailable"}

# =============================================================================
# Compression
# =============================================================================

class CompressionManager:
    """Response compression management."""
    
    def __init__(self, min_size: int = 1024, compression_level: int = 6):
        
    """__init__ function."""
self.min_size = min_size
        self.compression_level = compression_level
        self.compressible_types = {
            "text/", "application/json", "application/xml", "application/javascript"
        }
    
    def should_compress(self, response: Response) -> bool:
        """Check if response should be compressed."""
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(compressible in content_type for compressible in self.compressible_types):
            return False
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.min_size:
            return False
        
        return True
    
    def compress_response(self, response: Response) -> Response:
        """Compress response content."""
        try:
            
            # Get response content
            if hasattr(response, 'body'):
                content = response.body
            else:
                return response
            
            # Compress content
            compressed_content = gzip.compress(content, self.compression_level)
            
            # Create new response
            compressed_response = Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
            # Update headers
            compressed_response.headers["content-encoding"] = "gzip"
            compressed_response.headers["content-length"] = str(len(compressed_content))
            
            return compressed_response
            
        except Exception as e:
            logger.error("Compression error", error=str(e))
            return response

# =============================================================================
# Main Performance Middleware
# =============================================================================

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Comprehensive performance optimization middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_client: Optional[redis.Redis] = None,
        enable_caching: bool = True,
        enable_compression: bool = True,
        enable_rate_limiting: bool = True,
        enable_metrics: bool = True,
        cache_ttl: int = 300,
        slow_request_threshold_ms: float = 1000.0,
        exclude_paths: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None,
    ):
        
    """__init__ function."""
super().__init__(app)
        
        # Initialize components
        self.cache_manager = CacheManager(redis_client) if enable_caching else None
        self.rate_limiter = RateLimiter(redis_client) if enable_rate_limiting else None
        self.compression_manager = CompressionManager() if enable_compression else None
        self.metrics = PerformanceMetrics() if enable_metrics else None
        
        # Configuration
        self.cache_ttl = cache_ttl
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs"]
        self.exclude_methods = exclude_methods or ["OPTIONS"]
        
        # Setup rate limits
        if self.rate_limiter:
            self._setup_rate_limits()
    
    def _setup_rate_limits(self) -> Any:
        """Setup rate limits for different endpoints."""
        # General API limit
        self.rate_limiter.set_limit("api", 1000, 60)  # 1000 requests per minute
        
        # Authentication endpoints
        self.rate_limiter.set_limit("auth", 10, 60)  # 10 auth attempts per minute
        
        # Video processing endpoints
        self.rate_limiter.set_limit("video_processing", 50, 60)  # 50 video requests per minute
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance optimizations."""
        
        # Check if request should be excluded
        if self._should_exclude_request(request):
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            # Rate limiting
            if self.rate_limiter:
                response = await self._handle_rate_limiting(request)
                if response:
                    return response
            
            # Caching
            if self.cache_manager:
                cached_response = await self._handle_caching(request)
                if cached_response:
                    return cached_response
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if self.cache_manager and response.status_code == 200:
                await self._cache_response(request, response)
            
            # Compression
            if self.compression_manager:
                response = self._handle_compression(request, response)
            
            # Metrics
            if self.metrics:
                self._record_metrics(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            if self.metrics:
                self.metrics.record_error(type(e).__name__)
            
            raise
    
    async def _should_exclude_request(self, request: Request) -> bool:
        """Check if request should be excluded from performance optimizations."""
        # Check path exclusions
        for exclude_path in self.exclude_paths:
            if request.url.path.startswith(exclude_path):
                return True
        
        # Check method exclusions
        if request.method in self.exclude_methods:
            return True
        
        return False
    
    async def _handle_rate_limiting(self, request: Request) -> Optional[Response]:
        """Handle rate limiting."""
        client_key = self.rate_limiter.get_client_key(request)
        
        # Determine rate limit key based on endpoint
        if request.url.path.startswith("/auth"):
            limit_key = "auth"
        elif request.url.path.startswith("/videos"):
            limit_key = "video_processing"
        else:
            limit_key = "api"
        
        is_allowed, limit_info = await self.rate_limiter.is_allowed(limit_key, client_key)
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "retry_after": limit_info.get("reset_time", 0)
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info.get("limit", 0)),
                    "X-RateLimit-Remaining": str(limit_info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(limit_info.get("reset_time", 0)),
                }
            )
        
        return None
    
    async def _handle_caching(self, request: Request) -> Optional[Response]:
        """Handle request caching."""
        # Only cache GET requests
        if request.method != "GET":
            return None
        
        cache_key = self.cache_manager.generate_cache_key(request)
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            return JSONResponse(
                content=cached_data,
                headers={"X-Cache": "HIT"}
            )
        
        return None
    
    async def _cache_response(self, request: Request, response: Response):
        """Cache successful response."""
        try:
            cache_key = self.cache_manager.generate_cache_key(request)
            
            # Extract response content
            if hasattr(response, 'body'):
                content = response.body
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                # Parse JSON if possible
                try:
                    content_data = json.loads(content)
                    await self.cache_manager.set(cache_key, content_data, self.cache_ttl)
                except json.JSONDecodeError:
                    # Cache as string if not JSON
                    await self.cache_manager.set(cache_key, content, self.cache_ttl)
            
        except Exception as e:
            logger.error("Caching error", error=str(e))
    
    def _handle_compression(self, request: Request, response: Response) -> Response:
        """Handle response compression."""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        if self.compression_manager.should_compress(response):
            return self.compression_manager.compress_response(response)
        
        return response
    
    def _record_metrics(self, request: Request, response: Response, start_time: float):
        """Record performance metrics."""
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        endpoint = request.url.path
        
        # Get response size
        response_size = None
        content_length = response.headers.get("content-length")
        if content_length:
            response_size = int(content_length)
        
        # Record metrics
        self.metrics.record_request(duration, response_size, endpoint)
        
        # Record slow requests
        if duration > self.slow_request_threshold_ms:
            self.metrics.record_slow_request({
                "url": str(request.url),
                "method": request.method,
                "duration_ms": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, 'request_id', None)
            })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics:
            return {"error": "Metrics not enabled"}
        
        stats = self.metrics.get_statistics()
        
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_stats()
        
        return stats
    
    def get_endpoint_performance(self) -> Dict[str, Dict[str, float]]:
        """Get endpoint-specific performance statistics."""
        if not self.metrics:
            return {}
        
        return self.metrics.get_endpoint_performance()

# =============================================================================
# Utility Functions
# =============================================================================

def create_performance_middleware(
    redis_client: Optional[redis.Redis] = None,
    enable_caching: bool = True,
    enable_compression: bool = True,
    enable_rate_limiting: bool = True,
    enable_metrics: bool = True,
    cache_ttl: int = 300,
    slow_request_threshold_ms: float = 1000.0,
    exclude_paths: Optional[List[str]] = None,
    exclude_methods: Optional[List[str]] = None,
) -> PerformanceMiddleware:
    """Create performance middleware with configuration."""
    
    return PerformanceMiddleware(
        redis_client=redis_client,
        enable_caching=enable_caching,
        enable_compression=enable_compression,
        enable_rate_limiting=enable_rate_limiting,
        enable_metrics=enable_metrics,
        cache_ttl=cache_ttl,
        slow_request_threshold_ms=slow_request_threshold_ms,
        exclude_paths=exclude_paths,
        exclude_methods=exclude_methods,
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "PerformanceMiddleware",
    "CacheManager",
    "RateLimiter",
    "CompressionManager",
    "PerformanceMetrics",
    "create_performance_middleware",
] 