"""
Advanced FastAPI Patterns for BUL API
=====================================

Implementation of advanced FastAPI patterns following best practices:
- Functional programming approach
- RORO pattern (Receive an Object, Return an Object)
- Advanced dependency injection
- Performance optimization patterns
- Error handling strategies
"""

from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import asyncio
import time
import logging
from datetime import datetime, timedelta
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt
import orjson
from starlette.middleware.base import BaseHTTPMiddleware

# Type variables for generic patterns
T = TypeVar('T')
R = TypeVar('R')

# Advanced Pydantic Models with RORO pattern
class RequestContext(BaseModel):
    """Request context following RORO pattern"""
    request_id: str
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ResponseContext(BaseModel, Generic[T]):
    """Response context following RORO pattern"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Advanced Error Handling
class APIError(Exception):
    """Base API error with context"""
    def __init__(self, message: str, status_code: int = 500, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.context = context or {}
        super().__init__(self.message)

class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, 400, {"field": field})

class BusinessLogicError(APIError):
    """Business logic error"""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message, 422, {"code": code})

class ExternalServiceError(APIError):
    """External service error"""
    def __init__(self, message: str, service: str):
        super().__init__(message, 503, {"service": service})

# Advanced Dependency Injection Patterns
class DependencyContainer:
    """Advanced dependency container with lifecycle management"""
    
    def __init__(self):
        self._dependencies: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton dependency"""
        self._singletons[name] = instance
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory dependency"""
        self._factories[name] = factory
    
    def get(self, name: str) -> Any:
        """Get dependency by name"""
        if name in self._singletons:
            return self._singletons[name]
        
        if name in self._factories:
            return self._factories[name]()
        
        raise ValueError(f"Dependency '{name}' not found")
    
    def get_or_create(self, name: str, factory: Callable) -> Any:
        """Get or create dependency"""
        if name not in self._dependencies:
            self._dependencies[name] = factory()
        return self._dependencies[name]

# Global dependency container
container = DependencyContainer()

# Advanced Caching Patterns
class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

class AdvancedCache:
    """Advanced caching with multiple strategies"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU, max_size: int = 1000):
        self.strategy = strategy
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._ttl_map: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with strategy-specific logic"""
        if key not in self._cache:
            return None
        
        # TTL check
        if self.strategy == CacheStrategy.TTL:
            if time.time() > self._ttl_map.get(key, 0):
                await self.delete(key)
                return None
        
        # Update access time for LRU
        if self.strategy == CacheStrategy.LRU:
            self._access_times[key] = time.time()
        
        return self._cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with strategy-specific logic"""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            await self._evict()
        
        self._cache[key] = value
        
        if self.strategy == CacheStrategy.TTL and ttl:
            self._ttl_map[key] = time.time() + ttl
        
        if self.strategy == CacheStrategy.LRU:
            self._access_times[key] = time.time()
    
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._ttl_map.pop(key, None)
    
    async def _evict(self) -> None:
        """Evict items based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            await self.delete(oldest_key)
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired items
            current_time = time.time()
            expired_keys = [k for k, expiry in self._ttl_map.items() if current_time > expiry]
            for key in expired_keys:
                await self.delete(key)

# Advanced Performance Monitoring
class PerformanceMetrics:
    """Advanced performance metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
    
    def record_timing(self, operation: str, duration: float) -> None:
        """Record timing metric"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
        # Keep only last 1000 measurements
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation] = self.metrics[operation][-1000:]
    
    def increment_counter(self, counter: str, value: int = 1) -> None:
        """Increment counter metric"""
        self.counters[counter] = self.counters.get(counter, 0) + value
    
    def set_gauge(self, gauge: str, value: float) -> None:
        """Set gauge metric"""
        self.gauges[gauge] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for operation, timings in self.metrics.items():
            if timings:
                stats[operation] = {
                    "count": len(timings),
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "p95": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0]
                }
        
        stats["counters"] = self.counters.copy()
        stats["gauges"] = self.gauges.copy()
        
        return stats

# Global performance metrics
performance_metrics = PerformanceMetrics()

# Advanced Decorators
def monitor_performance(operation: str):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_metrics.record_timing(operation, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_metrics.record_timing(operation, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def cache_result(ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorator to cache function results"""
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(CacheStrategy.TTL)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                performance_metrics.increment_counter("cache_hits")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            performance_metrics.increment_counter("cache_misses")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = asyncio.run(cache.get(cache_key))
            if cached_result is not None:
                performance_metrics.increment_counter("cache_hits")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            asyncio.run(cache.set(cache_key, result, ttl))
            performance_metrics.increment_counter("cache_misses")
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                        performance_metrics.increment_counter(f"{func.__name__}_retries")
                    else:
                        performance_metrics.increment_counter(f"{func.__name__}_failures")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                        performance_metrics.increment_counter(f"{func.__name__}_retries")
                    else:
                        performance_metrics.increment_counter(f"{func.__name__}_failures")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Advanced Middleware
class AdvancedLoggingMiddleware(BaseHTTPMiddleware):
    """Advanced logging middleware with structured logging"""
    
    def __init__(self, app, logger_name: str = "bul_api"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)
    
    async def dispatch(self, request: Request, call_next):
        # Extract request information
        request_id = request.headers.get("X-Request-ID", f"req_{int(time.time())}")
        start_time = time.time()
        
        # Log request
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Record metrics
            performance_metrics.record_timing("request_duration", duration)
            performance_metrics.increment_counter("requests_total")
            performance_metrics.increment_counter(f"status_{response.status_code}")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Record error metrics
            performance_metrics.increment_counter("request_errors")
            performance_metrics.record_timing("error_duration", duration)
            
            raise

class CompressionMiddleware(BaseHTTPMiddleware):
    """Advanced compression middleware"""
    
    def __init__(self, app, min_size: int = 1000, compression_level: int = 6):
        super().__init__(app)
        self.min_size = min_size
        self.compression_level = compression_level
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Check if response should be compressed
        if self._should_compress(request, response):
            # Add compression headers
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """Determine if response should be compressed"""
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return False
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.min_size:
            return False
        
        # Check if already compressed
        if response.headers.get("content-encoding"):
            return False
        
        return True

# Advanced Route Patterns
class AdvancedRoute(APIRoute):
    """Advanced route with custom behavior"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_tracker = True
    
    async def handle_request(self, request: Request, call_next):
        """Custom request handling with performance tracking"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Track performance
            if self.performance_tracker:
                performance_metrics.record_timing(f"route_{self.name}", duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            performance_metrics.record_timing(f"route_{self.name}_error", duration)
            raise

# Advanced Error Handlers
def create_error_handler(status_code: int, error_type: str):
    """Factory function to create error handlers"""
    def error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_type,
                "message": str(exc),
                "request_id": request.headers.get("X-Request-ID"),
                "timestamp": datetime.now().isoformat()
            }
        )
    return error_handler

# Advanced Utility Functions
def create_response_context(
    data: Any,
    request_id: str,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ResponseContext:
    """Create response context following RORO pattern"""
    return ResponseContext(
        success=success,
        data=data,
        error=error,
        request_id=request_id,
        metadata=metadata or {}
    )

def extract_request_context(request: Request) -> RequestContext:
    """Extract request context from FastAPI request"""
    return RequestContext(
        request_id=request.headers.get("X-Request-ID", f"req_{int(time.time())}"),
        user_id=request.headers.get("X-User-ID"),
        metadata={
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent")
        }
    )

# Advanced Async Context Managers
@asynccontextmanager
async def database_transaction():
    """Database transaction context manager"""
    # This would integrate with actual database
    try:
        # Begin transaction
        yield
        # Commit transaction
    except Exception:
        # Rollback transaction
        raise

@asynccontextmanager
async def cache_operation(cache_key: str, ttl: int = 300):
    """Cache operation context manager"""
    cache = AdvancedCache(CacheStrategy.TTL)
    
    # Try to get from cache
    cached_value = await cache.get(cache_key)
    if cached_value is not None:
        yield cached_value
        return
    
    # Execute operation and cache result
    try:
        result = yield
        await cache.set(cache_key, result, ttl)
    except Exception:
        # Don't cache errors
        raise

# Advanced Functional Patterns
def create_pipeline(*functions: Callable) -> Callable:
    """Create a pipeline of functions"""
    def pipeline(data: Any) -> Any:
        result = data
        for func in functions:
            result = func(result)
        return result
    return pipeline

def create_async_pipeline(*functions: Callable) -> Callable:
    """Create an async pipeline of functions"""
    async def pipeline(data: Any) -> Any:
        result = data
        for func in functions:
            if asyncio.iscoroutinefunction(func):
                result = await func(result)
            else:
                result = func(result)
        return result
    return pipeline

# Advanced Validation
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate required fields in data"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> None:
    """Validate field types in data"""
    for field, expected_type in field_types.items():
        if field in data and not isinstance(data[field], expected_type):
            raise ValidationError(
                f"Field '{field}' must be of type {expected_type.__name__}",
                field=field
            )

# Advanced Serialization
def serialize_response(data: Any) -> Dict[str, Any]:
    """Serialize response data with advanced handling"""
    if isinstance(data, BaseModel):
        return data.dict()
    elif isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return [serialize_response(item) for item in data]
    else:
        return {"value": data}

# Performance Optimization Utilities
@lru_cache(maxsize=128)
def get_cached_config(key: str) -> Any:
    """Get cached configuration value"""
    # This would integrate with actual config system
    return None

async def batch_process(items: List[Any], processor: Callable, batch_size: int = 10) -> List[Any]:
    """Process items in batches for better performance"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)
    
    return results

def optimize_json_serialization(data: Any) -> bytes:
    """Optimize JSON serialization using orjson"""
    return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)

# Advanced Health Check
async def advanced_health_check() -> Dict[str, Any]:
    """Advanced health check with detailed metrics"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "uptime": time.time(),
        "metrics": performance_metrics.get_stats(),
        "dependencies": {
            "database": await _check_database_health(),
            "cache": await _check_cache_health(),
            "external_apis": await _check_external_apis_health()
        }
    }
    
    # Determine overall health
    all_healthy = all(
        dep.get("status") == "healthy" 
        for dep in health_status["dependencies"].values()
    )
    
    if not all_healthy:
        health_status["status"] = "degraded"
    
    return health_status

async def _check_database_health() -> Dict[str, Any]:
    """Check database health"""
    # This would integrate with actual database
    return {"status": "healthy", "response_time": 0.1}

async def _check_cache_health() -> Dict[str, Any]:
    """Check cache health"""
    # This would integrate with actual cache
    return {"status": "healthy", "response_time": 0.05}

async def _check_external_apis_health() -> Dict[str, Any]:
    """Check external APIs health"""
    # This would integrate with actual external APIs
    return {"status": "healthy", "response_time": 0.2}

# Export all advanced patterns
__all__ = [
    # Models
    "RequestContext",
    "ResponseContext",
    
    # Error Handling
    "APIError",
    "ValidationError", 
    "BusinessLogicError",
    "ExternalServiceError",
    
    # Dependency Injection
    "DependencyContainer",
    "container",
    
    # Caching
    "CacheStrategy",
    "AdvancedCache",
    
    # Performance
    "PerformanceMetrics",
    "performance_metrics",
    
    # Decorators
    "monitor_performance",
    "cache_result",
    "retry_on_failure",
    
    # Middleware
    "AdvancedLoggingMiddleware",
    "CompressionMiddleware",
    
    # Routes
    "AdvancedRoute",
    
    # Error Handlers
    "create_error_handler",
    
    # Utilities
    "create_response_context",
    "extract_request_context",
    "database_transaction",
    "cache_operation",
    "create_pipeline",
    "create_async_pipeline",
    "validate_required_fields",
    "validate_field_types",
    "serialize_response",
    "get_cached_config",
    "batch_process",
    "optimize_json_serialization",
    "advanced_health_check"
]












