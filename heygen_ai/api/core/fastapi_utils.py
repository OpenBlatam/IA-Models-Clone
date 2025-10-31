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

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from typing import Any, List, Dict, Optional
"""
FastAPI Utilities - Advanced Features and Middleware
Comprehensive utilities for enhanced FastAPI applications with production-ready features.
"""



logger = structlog.get_logger()

# =============================================================================
# REDIS-BACKED RATE LIMITING
# =============================================================================

class RedisRateLimiter:
    """Redis-backed rate limiter with sliding window."""
    
    def __init__(self, redis_client: redis.Redis, prefix: str = "rate_limit"):
        
    """__init__ function."""
self.redis = redis_client
        self.prefix = prefix
    
    async def is_allowed(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed within rate limit."""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Create Redis key
        redis_key = f"{self.prefix}:{key}"
        
        # Remove old entries and add current request
        await self.redis.zremrangebyscore(redis_key, 0, window_start)
        await self.redis.zadd(redis_key, {str(current_time): current_time})
        await self.redis.expire(redis_key, window_seconds)
        
        # Count requests in window
        request_count = await self.redis.zcard(redis_key)
        
        # Get remaining requests
        remaining = max(0, max_requests - request_count)
        
        # Check if allowed
        allowed = request_count < max_requests
        
        return allowed, {
            "limit": max_requests,
            "remaining": remaining,
            "reset_time": current_time + window_seconds,
            "window_seconds": window_seconds
        }


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting middleware."""
    
    def __init__(
        self, 
        app: ASGIApp, 
        redis_client: redis.Redis,
        default_limit: int = 100,
        default_window: int = 60,
        key_func: Optional[Callable[[Request], str]] = None
    ):
        
    """__init__ function."""
super().__init__(app)
        self.rate_limiter = RedisRateLimiter(redis_client)
        self.default_limit = default_limit
        self.default_window = default_window
        self.key_func = key_func or self._default_key_func
    
    def _default_key_func(self, request: Request) -> str:
        """Default key function for rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        return f"{client_ip}:{request.url.path}"
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        
    """dispatch function."""
# Get rate limit key
        key = self.key_func(request)
        
        # Check rate limit
        allowed, rate_info = await self.rate_limiter.is_allowed(
            key, self.default_limit, self.default_window
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(rate_info["reset_time"] - time.time()),
                    "limit": rate_info["limit"],
                    "window_seconds": rate_info["window_seconds"]
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(int(rate_info["reset_time"]))
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers.update({
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(int(rate_info["reset_time"]))
        })
        
        return response


# =============================================================================
# REDIS-BACKED CACHING
# =============================================================================

class RedisCache:
    """Redis-backed caching with TTL and compression."""
    
    def __init__(self, redis_client: redis.Redis, prefix: str = "cache"):
        
    """__init__ function."""
self.redis = redis_client
        self.prefix = prefix
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = await self.redis.get(self._make_key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL."""
        try:
            data = json.dumps(value)
            return await self.redis.setex(self._make_key(key), ttl, data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(await self.redis.delete(self._make_key(key)))
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis.exists(self._make_key(key)))
        except Exception as e:
            logger.warning(f"Cache exists error: {e}")
            return False


class RedisCacheMiddleware(BaseHTTPMiddleware):
    """Redis-backed caching middleware."""
    
    def __init__(
        self, 
        app: ASGIApp, 
        redis_client: redis.Redis,
        default_ttl: int = 300,
        cacheable_paths: Optional[List[str]] = None
    ):
        
    """__init__ function."""
super().__init__(app)
        self.cache = RedisCache(redis_client)
        self.default_ttl = default_ttl
        self.cacheable_paths = cacheable_paths or ["/api/"]
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable."""
        if request.method != "GET":
            return False
        
        path = request.url.path
        return any(path.startswith(prefix) for prefix in self.cacheable_paths)
    
    def _make_cache_key(self, request: Request) -> str:
        """Create cache key from request."""
        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items()))
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        
    """dispatch function."""
if not self._is_cacheable(request):
            return await call_next(request)
        
        # Check cache
        cache_key = self._make_cache_key(request)
        cached_response = await self.cache.get(cache_key)
        
        if cached_response:
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"]
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                cache_data = {
                    "content": json.loads(body.decode()),
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                
                await self.cache.set(cache_key, cache_data, self.default_ttl)
                
                # Return response with body
                return JSONResponse(
                    content=cache_data["content"],
                    status_code=cache_data["status_code"],
                    headers=cache_data["headers"]
                )
            except Exception as e:
                logger.warning(f"Cache storage error: {e}")
        
        return response


# =============================================================================
# BACKGROUND TASK MANAGEMENT
# =============================================================================

class BackgroundTaskManager:
    """Background task manager with monitoring and error handling."""
    
    def __init__(self) -> Any:
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def add_task(
        self, 
        task_id: str, 
        coro: Awaitable, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """Add a background task."""
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        
        self.task_metadata[task_id] = {
            "created_at": datetime.now(),
            "status": "running",
            "metadata": metadata or {}
        }
        
        # Add done callback
        task.add_done_callback(lambda t: self._task_done_callback(task_id, t))
        
        logger.info(f"Background task added", task_id=task_id)
        return task
    
    def _task_done_callback(self, task_id: str, task: asyncio.Task):
        """Callback when task is done."""
        try:
            if task.cancelled():
                self.task_metadata[task_id]["status"] = "cancelled"
            elif task.exception():
                self.task_metadata[task_id]["status"] = "failed"
                self.task_metadata[task_id]["error"] = str(task.exception())
            else:
                self.task_metadata[task_id]["status"] = "completed"
            
            self.task_metadata[task_id]["completed_at"] = datetime.now()
            
            logger.info(f"Background task completed", 
                       task_id=task_id, 
                       status=self.task_metadata[task_id]["status"])
            
        except Exception as e:
            logger.error(f"Task callback error: {e}")
    
    def get_task(self, task_id: str) -> Optional[asyncio.Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information."""
        return self.task_metadata.get(task_id)
    
    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all tasks with their status."""
        return {
            task_id: {
                "task": task,
                "done": task.done(),
                "cancelled": task.cancelled(),
                "exception": task.exception() if task.done() else None,
                **metadata
            }
            for task_id, task in self.tasks.items()
            for metadata in [self.task_metadata.get(task_id, {})]
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up completed tasks older than max_age_hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, metadata in self.task_metadata.items():
            if (metadata.get("status") in ["completed", "failed", "cancelled"] and
                metadata.get("completed_at", datetime.now()) < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            self.tasks.pop(task_id, None)
            self.task_metadata.pop(task_id, None)
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")


# =============================================================================
# REQUEST VALIDATION AND SANITIZATION
# =============================================================================

class RequestValidator:
    """Request validation and sanitization utilities."""
    
    @staticmethod
    def validate_content_length(request: Request, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate request content length."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                return size <= max_size
            except ValueError:
                return False
        return True
    
    @staticmethod
    def validate_content_type(request: Request, allowed_types: List[str]) -> bool:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "")
        return any(allowed_type in content_type for allowed_type in allowed_types)
    
    @staticmethod
    def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize request headers."""
        sanitized = {}
        for key, value in headers.items():
            # Remove potentially dangerous headers
            if key.lower() in ["x-forwarded-for", "x-real-ip", "x-forwarded-host"]:
                continue
            
            # Sanitize header values
            sanitized_value = str(value).strip()[:1000]  # Limit length
            sanitized[key] = sanitized_value
        
        return sanitized
    
    @staticmethod
    def validate_json_payload(data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate JSON payload structure."""
        return all(field in data for field in required_fields)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation middleware."""
    
    def __init__(
        self, 
        app: ASGIApp,
        max_content_length: int = 10 * 1024 * 1024,
        allowed_content_types: Optional[List[str]] = None
    ):
        
    """__init__ function."""
super().__init__(app)
        self.max_content_length = max_content_length
        self.allowed_content_types = allowed_content_types or ["application/json", "multipart/form-data"]
        self.validator = RequestValidator()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        
    """dispatch function."""
# Validate content length
        if not self.validator.validate_content_length(request, self.max_content_length):
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Request payload too large"}
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            if not self.validator.validate_content_type(request, self.allowed_content_types):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={"error": "Unsupported content type"}
                )
        
        # Sanitize headers
        request.headers._list = [
            (name, value) for name, value in request.headers.raw
            if name.lower() not in ["x-forwarded-for", "x-real-ip", "x-forwarded-host"]
        ]
        
        return await call_next(request)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Performance monitoring with detailed metrics."""
    
    def __init__(self) -> Any:
        # Request metrics
        self.request_counter = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
        self.request_size = Histogram('http_request_size_bytes', 'HTTP request size', ['method', 'endpoint'])
        self.response_size = Histogram('http_response_size_bytes', 'HTTP response size', ['method', 'endpoint'])
        
        # Error metrics
        self.error_counter = Counter('http_errors_total', 'Total HTTP errors', ['method', 'endpoint', 'error_type'])
        
        # Cache metrics
        self.cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
        self.cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
        
        # Background task metrics
        self.background_tasks = Counter('background_tasks_total', 'Total background tasks', ['task_type', 'status'])
        
        # System metrics
        self.active_connections = Gauge('active_connections', 'Active connections')
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_error(self, method: str, endpoint: str, error_type: str):
        """Record error metrics."""
        self.error_counter.labels(method=method, endpoint=endpoint, error_type=error_type).inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_background_task(self, task_type: str, status: str):
        """Record background task metrics."""
        self.background_tasks.labels(task_type=task_type, status=status).inc()


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware."""
    
    def __init__(self, app: ASGIApp, monitor: Optional[PerformanceMonitor] = None):
        
    """__init__ function."""
super().__init__(app)
        self.monitor = monitor or PerformanceMonitor()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        
    """dispatch function."""
start_time = time.time()
        
        # Record request start
        method = request.method
        endpoint = request.url.path
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            
            # Record metrics
            self.monitor.record_request(method, endpoint, response.status_code, duration)
            
            # Add performance headers
            response.headers["X-Processing-Time"] = f"{duration:.4f}"
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self.monitor.record_error(method, endpoint, type(e).__name__)
            raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    # In production, this would use connection pooling and configuration
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def get_background_task_manager() -> BackgroundTaskManager:
    """Get background task manager instance."""
    return BackgroundTaskManager()


def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance."""
    return PerformanceMonitor()


# Dependency injection functions
async def get_rate_limiter() -> RedisRateLimiter:
    """Get rate limiter dependency."""
    redis_client = get_redis_client()
    return RedisRateLimiter(redis_client)


async def get_cache() -> RedisCache:
    """Get cache dependency."""
    redis_client = get_redis_client()
    return RedisCache(redis_client)


async def get_task_manager() -> BackgroundTaskManager:
    """Get task manager dependency."""
    return get_background_task_manager()


# Decorators
def rate_limit(max_requests: int, window_seconds: int = 60):
    """Rate limiting decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get rate limiter from dependencies
            rate_limiter = await get_rate_limiter()
            
            # Create key (in production, use user ID or IP)
            key = "default"
            
            # Check rate limit
            allowed, rate_info = await rate_limiter.is_allowed(key, max_requests, window_seconds)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def cache_response(ttl: int = 300, key_func: Optional[Callable] = None):
    """Response caching decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get cache from dependencies
            cache_client = await get_cache()
            
            # Create cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached_result = await cache_client.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_client.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def background_task(task_type: str = "default"):
    """Background task decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get task manager from dependencies
            task_manager = await get_task_manager()
            
            # Create task ID
            task_id = f"{task_type}_{int(time.time() * 1000)}"
            
            # Add task
            task = await task_manager.add_task(
                task_id,
                func(*args, **kwargs),
                metadata={"task_type": task_type, "function": func.__name__}
            )
            
            return {"task_id": task_id, "status": "started"}
        return wrapper
    return decorator 