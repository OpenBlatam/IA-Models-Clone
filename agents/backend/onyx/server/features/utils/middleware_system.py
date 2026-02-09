from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import hashlib
import json
import traceback
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis.asyncio as redis
from typing import Any, List, Dict, Optional
import logging
"""
Middleware System - FastAPI Middleware for Logging, Error Monitoring, and Performance
Comprehensive middleware system providing logging, error monitoring, performance optimization,
security, and observability features for FastAPI applications.
"""



logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class MiddlewareConfig(BaseModel):
    """Configuration for middleware system."""
    
    # Logging
    logging_enabled: bool = Field(default=True, description="Enable request logging")
    log_request_body: bool = Field(default=False, description="Log request body")
    log_response_body: bool = Field(default=False, description="Log response body")
    log_headers: bool = Field(default=False, description="Log request/response headers")
    sensitive_headers: List[str] = Field(
        default=["authorization", "cookie", "x-api-key"],
        description="Headers to mask in logs"
    )
    
    # Performance monitoring
    performance_monitoring_enabled: bool = Field(default=True, description="Enable performance monitoring")
    slow_request_threshold: float = Field(default=1.0, description="Slow request threshold in seconds")
    performance_metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    
    # Error monitoring
    error_monitoring_enabled: bool = Field(default=True, description="Enable error monitoring")
    error_sampling_rate: float = Field(default=1.0, description="Error sampling rate (0.0-1.0)")
    error_retention_days: int = Field(default=30, description="Error retention in days")
    
    # Security
    security_enabled: bool = Field(default=True, description="Enable security middleware")
    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # CORS
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # Compression
    compression_enabled: bool = Field(default=True, description="Enable response compression")
    compression_min_size: int = Field(default=1000, description="Minimum size for compression")
    
    # Trusted hosts
    trusted_hosts_enabled: bool = Field(default=False, description="Enable trusted hosts")
    trusted_hosts: List[str] = Field(default=["*"], description="Trusted host patterns")
    
    # Redis (for rate limiting and caching)
    redis_enabled: bool = Field(default=False, description="Enable Redis integration")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_ttl: int = Field(default=3600, description="Redis TTL in seconds")

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """Performance metrics collection using Prometheus."""
    
    def __init__(self) -> Any:
        # Request counters
        self.request_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        # Request duration
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Response size
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint']
        )
        
        # Error rate
        self.error_rate = Counter(
            'http_errors_total',
            'Total HTTP errors',
            ['method', 'endpoint', 'error_type']
        )
        
        # Active requests
        self.active_requests = Gauge(
            'http_active_requests',
            'Number of active HTTP requests',
            ['method', 'endpoint']
        )
        
        # Memory usage
        self.memory_usage = Gauge(
            'app_memory_usage_bytes',
            'Application memory usage in bytes'
        )
        
        # CPU usage
        self.cpu_usage = Gauge(
            'app_cpu_usage_percent',
            'Application CPU usage percentage'
        )

# =============================================================================
# REQUEST CONTEXT
# =============================================================================

@dataclass
class RequestContext:
    """Context for tracking request information."""
    request_id: str
    start_time: float
    method: str
    url: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    status_code: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive request/response logging middleware."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.config.logging_enabled:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create request context
        context = RequestContext(
            request_id=request_id,
            start_time=start_time,
            method=request.method,
            url=str(request.url),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            request_size=self._get_request_size(request)
        )
        
        # Log request
        self._log_request(request, context)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update context
            context.status_code = response.status_code
            context.response_size = self._get_response_size(response)
            context.duration = time.time() - start_time
            
            # Log response
            self._log_response(response, context)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Update context with error
            context.duration = time.time() - start_time
            context.error = str(e)
            context.status_code = 500
            
            # Log error
            self._log_error(request, context, e)
            
            # Re-raise the exception
            raise
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else None
    
    async def _get_request_size(self, request: Request) -> int:
        """Get request size in bytes."""
        size = 0
        # URL size
        size += len(str(request.url))
        # Headers size
        for name, value in request.headers.items():
            size += len(name) + len(value)
        return size
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        size = 0
        # Headers size
        for name, value in response.headers.items():
            size += len(name) + len(value)
        return size
    
    def _log_request(self, request: Request, context: RequestContext):
        """Log incoming request."""
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "url": context.url,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
            "request_size": context.request_size
        }
        
        if self.config.log_headers:
            log_data["headers"] = self._sanitize_headers(request.headers)
        
        if self.config.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    log_data["body"] = body.decode()[:1000]  # Limit body size
            except Exception:
                log_data["body"] = "[Unable to read body]"
        
        self.logger.info("Incoming request", **log_data)
    
    def _log_response(self, response: Response, context: RequestContext):
        """Log response."""
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "url": context.url,
            "status_code": context.status_code,
            "duration": context.duration,
            "response_size": context.response_size
        }
        
        if self.config.log_headers:
            log_data["response_headers"] = dict(response.headers)
        
        self.logger.info("Request completed", **log_data)
    
    def _log_error(self, request: Request, context: RequestContext, error: Exception):
        """Log error."""
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "url": context.url,
            "error": str(error),
            "duration": context.duration,
            "traceback": traceback.format_exc()
        }
        
        self.logger.error("Request failed", **log_data)
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers by masking sensitive information."""
        sanitized = {}
        for name, value in headers.items():
            if name.lower() in self.config.sensitive_headers:
                sanitized[name] = "[REDACTED]"
            else:
                sanitized[name] = value
        return sanitized

# =============================================================================
# PERFORMANCE MONITORING MIDDLEWARE
# =============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring and metrics collection middleware."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig, metrics: PerformanceMetrics):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.metrics = metrics
        self.logger = structlog.get_logger(__name__)
        self.slow_requests = deque(maxlen=100)  # Keep last 100 slow requests
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.config.performance_monitoring_enabled:
            return await call_next(request)
        
        start_time = time.time()
        endpoint = self._get_endpoint(request)
        
        # Increment active requests
        self.metrics.active_requests.labels(
            method=request.method,
            endpoint=endpoint
        ).inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self._record_metrics(request, response, duration, endpoint)
            
            # Check for slow requests
            if duration > self.config.slow_request_threshold:
                self._log_slow_request(request, response, duration, endpoint)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record error metrics
            self._record_error_metrics(request, e, duration, endpoint)
            
            # Re-raise the exception
            raise
        finally:
            # Decrement active requests
            self.metrics.active_requests.labels(
                method=request.method,
                endpoint=endpoint
            ).dec()
    
    def _get_endpoint(self, request: Request) -> str:
        """Extract endpoint name from request."""
        path = request.url.path
        # Remove version prefix if present
        if path.startswith("/api/v"):
            parts = path.split("/")
            if len(parts) >= 3:
                return "/".join(parts[3:])
        return path
    
    def _record_metrics(self, request: Request, response: Response, duration: float, endpoint: str):
        """Record performance metrics."""
        if not self.config.performance_metrics_enabled:
            return
        
        # Request counter
        self.metrics.request_total.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()
        
        # Request duration
        self.metrics.request_duration.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)
        
        # Response size
        response_size = self._get_response_size(response)
        if response_size > 0:
            self.metrics.response_size.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(response_size)
    
    def _record_error_metrics(self, request: Request, error: Exception, duration: float, endpoint: str):
        """Record error metrics."""
        if not self.config.performance_metrics_enabled:
            return
        
        error_type = type(error).__name__
        self.metrics.error_rate.labels(
            method=request.method,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        size = 0
        for name, value in response.headers.items():
            size += len(name) + len(value)
        return size
    
    def _log_slow_request(self, request: Request, response: Response, duration: float, endpoint: str):
        """Log slow request details."""
        slow_request_info = {
            "method": request.method,
            "url": str(request.url),
            "endpoint": endpoint,
            "duration": duration,
            "status_code": response.status_code,
            "client_ip": self._get_client_ip(request)
        }
        
        self.slow_requests.append(slow_request_info)
        
        self.logger.warning("Slow request detected", **slow_request_info)
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else None
    
    async def get_slow_requests_summary(self) -> Dict[str, Any]:
        """Get summary of slow requests."""
        if not self.slow_requests:
            return {"count": 0, "requests": []}
        
        durations = [req["duration"] for req in self.slow_requests]
        return {
            "count": len(self.slow_requests),
            "average_duration": statistics.mean(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "recent_requests": list(self.slow_requests)[-10:]  # Last 10 requests
        }

# =============================================================================
# ERROR MONITORING MIDDLEWARE
# =============================================================================

class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Error monitoring and alerting middleware."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.redis_client = redis_client
        self.logger = structlog.get_logger(__name__)
        self.error_counts = defaultdict(int)
        self.error_samples = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.config.error_monitoring_enabled:
            return await call_next(request)
        
        try:
            return await call_next(request)
        except Exception as e:
            await self._handle_error(request, e)
            raise
    
    async def _handle_error(self, request: Request, error: Exception):
        """Handle and monitor errors."""
        error_type = type(error).__name__
        error_key = f"{request.method}:{request.url.path}:{error_type}"
        
        # Increment error count
        self.error_counts[error_key] += 1
        
        # Sample error details
        if len(self.error_samples[error_key]) < 10:  # Keep last 10 samples
            error_sample = {
                "timestamp": time.time(),
                "method": request.method,
                "url": str(request.url),
                "error_type": error_type,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent")
            }
            self.error_samples[error_key].append(error_sample)
        
        # Log error
        self.logger.error("Error monitored",
                         error_type=error_type,
                         error_message=str(error),
                         method=request.method,
                         url=str(request.url),
                         count=self.error_counts[error_key])
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_error_in_redis(error_key, error)
    
    async def _store_error_in_redis(self, error_key: str, error: Exception):
        """Store error information in Redis."""
        try:
            error_data = {
                "timestamp": time.time(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "count": self.error_counts[error_key]
            }
            
            # Store error data
            await self.redis_client.hset(
                f"errors:{error_key}",
                mapping=error_data
            )
            
            # Set expiration
            await self.redis_client.expire(
                f"errors:{error_key}",
                self.config.error_retention_days * 24 * 3600
            )
            
        except Exception as e:
            self.logger.error("Failed to store error in Redis", error=str(e))
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error monitoring summary."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "error_samples": dict(self.error_samples)
        }

# =============================================================================
# RATE LIMITING MIDDLEWARE
# =============================================================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis or in-memory storage."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.redis_client = redis_client
        self.logger = structlog.get_logger(__name__)
        self.local_limits = defaultdict(lambda: deque(maxlen=config.rate_limit_requests))
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.config.rate_limiting_enabled:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if await self._is_rate_limited(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": self.config.rate_limit_window
                },
                headers={"Retry-After": str(self.config.rate_limit_window)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.config.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(await self._get_remaining_requests(client_id))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.config.rate_limit_window))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Try to get from API key first
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        
        if self.redis_client:
            return await self._check_redis_rate_limit(client_id, current_time)
        else:
            return self._check_local_rate_limit(client_id, current_time)
    
    async def _check_redis_rate_limit(self, client_id: str, current_time: float) -> bool:
        """Check rate limit using Redis."""
        try:
            key = f"rate_limit:{client_id}"
            
            # Get current requests
            requests = await self.redis_client.zrangebyscore(
                key,
                current_time - self.config.rate_limit_window,
                current_time
            )
            
            if len(requests) >= self.config.rate_limit_requests:
                return True
            
            # Add current request
            await self.redis_client.zadd(key, {str(current_time): current_time})
            await self.redis_client.expire(key, self.config.rate_limit_window)
            
            return False
            
        except Exception as e:
            self.logger.error("Redis rate limit check failed", error=str(e))
            # Fall back to local rate limiting
            return self._check_local_rate_limit(client_id, current_time)
    
    def _check_local_rate_limit(self, client_id: str, current_time: float) -> bool:
        """Check rate limit using local storage."""
        # Remove old requests
        while (self.local_limits[client_id] and 
               current_time - self.local_limits[client_id][0] > self.config.rate_limit_window):
            self.local_limits[client_id].popleft()
        
        # Check if limit exceeded
        if len(self.local_limits[client_id]) >= self.config.rate_limit_requests:
            return True
        
        # Add current request
        self.local_limits[client_id].append(current_time)
        return False
    
    async async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        current_time = time.time()
        
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}"
                requests = await self.redis_client.zrangebyscore(
                    key,
                    current_time - self.config.rate_limit_window,
                    current_time
                )
                return max(0, self.config.rate_limit_requests - len(requests))
            except Exception:
                pass
        
        # Fall back to local calculation
        while (self.local_limits[client_id] and 
               current_time - self.local_limits[client_id][0] > self.config.rate_limit_window):
            self.local_limits[client_id].popleft()
        
        return max(0, self.config.rate_limit_requests - len(self.local_limits[client_id]))

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and basic protection."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            if header not in response.headers:
                response.headers[header] = value

# =============================================================================
# MIDDLEWARE MANAGER
# =============================================================================

class MiddlewareManager:
    """Manager for configuring and applying middleware."""
    
    def __init__(self, config: MiddlewareConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.config = config
        self.redis_client = redis_client
        self.metrics = PerformanceMetrics() if config.performance_metrics_enabled else None
        self.logger = structlog.get_logger(__name__)
    
    def setup_middleware(self, app: FastAPI) -> None:
        """Setup all middleware for the application."""
        self.logger.info("Setting up middleware", config=self.config.dict())
        
        # CORS middleware
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=self.config.cors_credentials,
                allow_methods=self.config.cors_methods,
                allow_headers=self.config.cors_headers,
            )
        
        # Trusted hosts middleware
        if self.config.trusted_hosts_enabled:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.trusted_hosts
            )
        
        # Compression middleware
        if self.config.compression_enabled:
            app.add_middleware(
                GZipMiddleware,
                minimum_size=self.config.compression_min_size
            )
        
        # Custom middleware (in order of execution)
        if self.config.logging_enabled:
            app.add_middleware(LoggingMiddleware, config=self.config)
        
        if self.config.performance_monitoring_enabled:
            app.add_middleware(PerformanceMonitoringMiddleware, config=self.config, metrics=self.metrics)
        
        if self.config.error_monitoring_enabled:
            app.add_middleware(ErrorMonitoringMiddleware, config=self.config, redis_client=self.redis_client)
        
        if self.config.rate_limiting_enabled:
            app.add_middleware(RateLimitingMiddleware, config=self.config, redis_client=self.redis_client)
        
        if self.config.security_enabled:
            app.add_middleware(SecurityMiddleware, config=self.config)
        
        self.logger.info("Middleware setup completed")
    
    def get_metrics(self) -> Optional[str]:
        """Get Prometheus metrics."""
        if self.metrics:
            return generate_latest()
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.metrics:
            return {"enabled": False}
        
        # Get slow requests summary from performance middleware
        # This would need to be accessed from the middleware instance
        return {
            "enabled": True,
            "metrics_available": True
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error monitoring summary."""
        if not self.config.error_monitoring_enabled:
            return {"enabled": False}
        
        # This would need to be accessed from the error monitoring middleware instance
        return {
            "enabled": True,
            "error_monitoring_active": True
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_middleware_config(
    logging_enabled: bool = True,
    performance_monitoring_enabled: bool = True,
    error_monitoring_enabled: bool = True,
    security_enabled: bool = True,
    rate_limiting_enabled: bool = True,
    **kwargs
) -> MiddlewareConfig:
    """Create middleware configuration with sensible defaults."""
    return MiddlewareConfig(
        logging_enabled=logging_enabled,
        performance_monitoring_enabled=performance_monitoring_enabled,
        error_monitoring_enabled=error_monitoring_enabled,
        security_enabled=security_enabled,
        rate_limiting_enabled=rate_limiting_enabled,
        **kwargs
    )

def create_production_middleware_config() -> MiddlewareConfig:
    """Create production-optimized middleware configuration."""
    return MiddlewareConfig(
        logging_enabled=True,
        log_request_body=False,
        log_response_body=False,
        log_headers=False,
        performance_monitoring_enabled=True,
        slow_request_threshold=2.0,
        performance_metrics_enabled=True,
        error_monitoring_enabled=True,
        error_sampling_rate=0.1,  # Sample 10% of errors
        security_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=100,
        cors_enabled=True,
        cors_origins=["https://yourdomain.com"],
        compression_enabled=True,
        compression_min_size=1000,
        trusted_hosts_enabled=True,
        trusted_hosts=["yourdomain.com", "*.yourdomain.com"],
        redis_enabled=True
    )

def create_development_middleware_config() -> MiddlewareConfig:
    """Create development-optimized middleware configuration."""
    return MiddlewareConfig(
        logging_enabled=True,
        log_request_body=True,
        log_response_body=True,
        log_headers=True,
        performance_monitoring_enabled=True,
        slow_request_threshold=0.5,
        performance_metrics_enabled=True,
        error_monitoring_enabled=True,
        error_sampling_rate=1.0,  # Log all errors
        security_enabled=True,
        rate_limiting_enabled=False,  # Disable rate limiting in dev
        cors_enabled=True,
        cors_origins=["*"],
        compression_enabled=True,
        compression_min_size=500,
        trusted_hosts_enabled=False,
        redis_enabled=False
    ) 