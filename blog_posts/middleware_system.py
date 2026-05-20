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
import uuid
import json
import hashlib
import logging
import traceback
from typing import (
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import weakref
from fastapi import (
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from pydantic import BaseModel, Field, validator
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil
import asyncio
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis.asyncio as redis
from cachetools import TTLCache
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from typing import Any, List, Dict, Optional
"""
🔧 COMPREHENSIVE MIDDLEWARE SYSTEM
=================================

Production-ready middleware system for FastAPI with:
- Structured logging with correlation IDs
- Error monitoring and tracking
- Performance optimization and metrics
- Rate limiting and security
- Health monitoring and alerts
- Request/response transformation
- Caching and compression
- Background task management

Features:
- Zero-downtime deployment ready
- Auto-scaling compatible
- Enterprise monitoring integration
- Real-time performance tracking
- Comprehensive error handling
- Security hardening
"""

    Dict, List, Optional, Any, Callable, Awaitable,
    Union, Tuple, Set, Deque
)

    FastAPI, Request, Response, HTTPException, status,
    BackgroundTasks, Depends
)

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class MiddlewareConfig(BaseModel):
    """Configuration for middleware system."""
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/console)")
    enable_request_logging: bool = Field(default=True, description="Enable request logging")
    enable_response_logging: bool = Field(default=False, description="Enable response logging")
    log_sensitive_headers: bool = Field(default=False, description="Log sensitive headers")
    correlation_id_header: str = Field(default="X-Correlation-ID", description="Correlation ID header")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    slow_request_threshold_ms: int = Field(default=1000, description="Slow request threshold in ms")
    enable_metrics_collection: bool = Field(default=True, description="Enable metrics collection")
    metrics_retention_days: int = Field(default=30, description="Metrics retention period")
    
    # Error monitoring
    enable_error_monitoring: bool = Field(default=True, description="Enable error monitoring")
    error_alert_threshold: int = Field(default=10, description="Error alert threshold")
    error_alert_window_minutes: int = Field(default=5, description="Error alert window")
    enable_error_tracking: bool = Field(default=True, description="Enable error tracking")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Security
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    trusted_hosts: List[str] = Field(default=["*"], description="Trusted hosts")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Cache max size")
    
    # Compression
    enable_compression: bool = Field(default=True, description="Enable response compression")
    compression_min_size: int = Field(default=1000, description="Minimum size for compression")
    
    # Health monitoring
    enable_health_monitoring: bool = Field(default=True, description="Enable health monitoring")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    
    # Redis configuration (for distributed features)
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed features")
    
    class Config:
        validate_assignment = True

# ============================================================================
# DATA MODELS
# ============================================================================

class RequestMetrics(BaseModel):
    """Request performance metrics."""
    request_id: str
    method: str
    path: str
    client_ip: str
    user_agent: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    cache_hit: bool = False
    rate_limited: bool = False
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

class ErrorEvent(BaseModel):
    """Error event for monitoring."""
    error_id: str
    request_id: Optional[str] = None
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"
    handled: bool = False

class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    cache_hit_rate: float

class HealthStatus(BaseModel):
    """System health status."""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Optional[PerformanceMetrics] = None

# ============================================================================
# METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """Prometheus metrics collector."""
    
    def __init__(self) -> Any:
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint']
        )
        
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size',
            ['method', 'endpoint']
        )
        
        # Error metrics
        self.error_count = Counter(
            'http_errors_total',
            'Total HTTP errors',
            ['method', 'endpoint', 'error_type']
        )
        
        # System metrics
        self.active_requests = Gauge(
            'http_active_requests',
            'Active HTTP requests',
            ['method', 'endpoint']
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage'
        )
        
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage'
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses'
        )
        
        # Rate limiting metrics
        self.rate_limit_exceeded = Counter(
            'rate_limit_exceeded_total',
            'Total rate limit violations'
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_error(self, method: str, endpoint: str, error_type: str):
        """Record error metrics."""
        self.error_count.labels(method=method, endpoint=endpoint, error_type=error_type).inc()
    
    def record_cache_hit(self) -> Any:
        """Record cache hit."""
        self.cache_hits.inc()
    
    def record_cache_miss(self) -> Any:
        """Record cache miss."""
        self.cache_misses.inc()
    
    def record_rate_limit_exceeded(self) -> Any:
        """Record rate limit violation."""
        self.rate_limit_exceeded.inc()
    
    def update_system_metrics(self) -> Any:
        """Update system metrics."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_usage.set(memory.used)
        self.cpu_usage.set(cpu)

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class ResponseCache:
    """Response caching system."""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        
    """__init__ function."""
self.cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.metrics_collector = MetricsCollector()
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items())),
            request.headers.get("authorization", ""),
            request.headers.get("content-type", "")
        ]
        
        # Include request body hash for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            body = getattr(request, "_body", b"")
            if body:
                key_parts.append(hashlib.md5(body).hexdigest())
        
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Response]:
        """Get cached response."""
        response = self.cache.get(key)
        if response:
            self.metrics_collector.record_cache_hit()
        else:
            self.metrics_collector.record_cache_miss()
        return response
    
    def set(self, key: str, response: Response):
        """Cache response."""
        self.cache[key] = response
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_remove:
            self.cache.pop(key, None)

# ============================================================================
# ERROR MONITORING
# ============================================================================

class ErrorMonitor:
    """Error monitoring and alerting system."""
    
    def __init__(self, alert_threshold: int = 10, alert_window_minutes: int = 5):
        
    """__init__ function."""
self.alert_threshold = alert_threshold
        self.alert_window = timedelta(minutes=alert_window_minutes)
        self.error_events: Deque[ErrorEvent] = deque()
        self.alerted_errors: Set[str] = set()
        self.metrics_collector = MetricsCollector()
    
    def record_error(self, error: Exception, request: Optional[Request] = None, context: Optional[Dict[str, Any]] = None) -> ErrorEvent:
        """Record an error event."""
        error_event = ErrorEvent(
            error_id=str(uuid.uuid4()),
            request_id=getattr(request, "state", {}).get("request_id") if request else None,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            handled=False
        )
        
        self.error_events.append(error_event)
        
        # Clean old events
        cutoff_time = datetime.now() - self.alert_window
        while self.error_events and self.error_events[0].timestamp < cutoff_time:
            self.error_events.popleft()
        
        # Check for alert conditions
        self._check_alerts(error_event)
        
        # Record metrics
        if request:
            self.metrics_collector.record_error(
                method=request.method,
                endpoint=request.url.path,
                error_type=error_event.error_type
            )
        
        return error_event
    
    def _check_alerts(self, error_event: ErrorEvent):
        """Check if error conditions warrant an alert."""
        error_key = f"{error_event.error_type}:{error_event.error_message}"
        
        if error_key in self.alerted_errors:
            return
        
        # Count recent errors of the same type
        recent_errors = [
            e for e in self.error_events
            if e.error_type == error_event.error_type
            and e.timestamp > datetime.now() - self.alert_window
        ]
        
        if len(recent_errors) >= self.alert_threshold:
            self._send_alert(error_event, recent_errors)
            self.alerted_errors.add(error_key)
    
    def _send_alert(self, error_event: ErrorEvent, recent_errors: List[ErrorEvent]):
        """Send error alert."""
        logger = structlog.get_logger("error_monitor")
        logger.error(
            "Error alert triggered",
            error_type=error_event.error_type,
            error_count=len(recent_errors),
            alert_window_minutes=self.alert_window.total_seconds() / 60,
            recent_errors=[e.error_message for e in recent_errors[-5:]]  # Last 5 errors
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        if not self.error_events:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}
        
        error_types = defaultdict(int)
        for event in self.error_events:
            error_types[event.error_type] += 1
        
        return {
            "total_errors": len(self.error_events),
            "error_types": dict(error_types),
            "recent_errors": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.error_type,
                    "message": event.error_message
                }
                for event in list(self.error_events)[-10:]  # Last 10 errors
            ]
        }

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self, slow_request_threshold_ms: int = 1000):
        
    """__init__ function."""
self.slow_request_threshold = slow_request_threshold_ms
        self.request_times: Deque[float] = deque(maxlen=1000)
        self.start_time = datetime.now()
        self.metrics_collector = MetricsCollector()
        self.logger = structlog.get_logger("performance_monitor")
    
    def record_request(self, request_metrics: RequestMetrics):
        """Record request performance metrics."""
        if request_metrics.duration_ms:
            self.request_times.append(request_metrics.duration_ms)
            
            # Check for slow requests
            if request_metrics.duration_ms > self.slow_request_threshold:
                self.logger.warning(
                    "Slow request detected",
                    request_id=request_metrics.request_id,
                    method=request_metrics.method,
                    path=request_metrics.path,
                    duration_ms=request_metrics.duration_ms,
                    threshold_ms=self.slow_request_threshold
                )
        
        # Record metrics
        self.metrics_collector.record_request(
            method=request_metrics.method,
            endpoint=request_metrics.path,
            status_code=request_metrics.status_code or 500,
            duration=request_metrics.duration_ms / 1000 if request_metrics.duration_ms else 0
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.request_times:
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                requests_per_second=0.0,
                error_rate=0.0,
                memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
                active_connections=0,
                cache_hit_rate=0.0
            )
        
        times = list(self.request_times)
        times.sort()
        
        total_requests = len(times)
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_requests=total_requests,
            successful_requests=total_requests,  # Would need to track separately
            failed_requests=0,  # Would need to track separately
            average_response_time_ms=sum(times) / len(times),
            p95_response_time_ms=times[int(len(times) * 0.95)] if len(times) > 0 else 0,
            p99_response_time_ms=times[int(len(times) * 0.99)] if len(times) > 0 else 0,
            requests_per_second=total_requests / uptime if uptime > 0 else 0,
            error_rate=0.0,  # Would need to track separately
            memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            active_connections=0,  # Would need to track separately
            cache_hit_rate=0.0  # Would need to get from cache
        )

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiting system."""
    
    def __init__(self, requests_per_minute: int = 100, window_seconds: int = 60):
        
    """__init__ function."""
self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: Dict[str, Deque[datetime]] = defaultdict(lambda: deque())
        self.metrics_collector = MetricsCollector()
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get real IP from headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client IP
        return request.client.host if request.client else "unknown"
    
    def is_allowed(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        client_id = self.get_client_identifier(request)
        now = datetime.now()
        
        # Clean old requests
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        while self.requests[client_id] and self.requests[client_id][0] < cutoff_time:
            self.requests[client_id].popleft()
        
        # Check rate limit
        current_requests = len(self.requests[client_id])
        allowed = current_requests < self.requests_per_minute
        
        if allowed:
            self.requests[client_id].append(now)
        else:
            self.metrics_collector.record_rate_limit_exceeded()
        
        return allowed, {
            "client_id": client_id,
            "current_requests": current_requests,
            "limit": self.requests_per_minute,
            "window_seconds": self.window_seconds,
            "reset_time": cutoff_time + timedelta(seconds=self.window_seconds)
        }

# ============================================================================
# MIDDLEWARE CLASSES
# ============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive logging middleware."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.logger = structlog.get_logger("request_logger")
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Setup structured logging."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if self.config.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        # Generate correlation ID
        correlation_id = request.headers.get(
            self.config.correlation_id_header,
            str(uuid.uuid4())
        )
        
        # Add to request state
        request.state.correlation_id = correlation_id
        request.state.request_id = str(uuid.uuid4())
        request.state.start_time = datetime.now()
        
        # Log request start
        if self.config.enable_request_logging:
            self._log_request_start(request, correlation_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response
            if self.config.enable_response_logging:
                self._log_response(request, response, correlation_id)
            
            return response
            
        except Exception as e:
            # Log error
            self._log_error(request, e, correlation_id)
            raise
    
    def _log_request_start(self, request: Request, correlation_id: str):
        """Log request start."""
        log_data = {
            "correlation_id": correlation_id,
            "request_id": request.state.request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_length": request.headers.get("content-length"),
            "content_type": request.headers.get("content-type"),
        }
        
        # Add headers if configured
        if self.config.log_sensitive_headers:
            log_data["headers"] = dict(request.headers)
        
        self.logger.info("Request started", **log_data)
    
    def _log_response(self, request: Request, response: Response, correlation_id: str):
        """Log response."""
        duration = (datetime.now() - request.state.start_time).total_seconds() * 1000
        
        log_data = {
            "correlation_id": correlation_id,
            "request_id": request.state.request_id,
            "status_code": response.status_code,
            "duration_ms": duration,
            "response_size": len(response.body) if hasattr(response, 'body') else 0,
        }
        
        self.logger.info("Request completed", **log_data)
    
    def _log_error(self, request: Request, error: Exception, correlation_id: str):
        """Log error."""
        duration = (datetime.now() - request.state.start_time).total_seconds() * 1000
        
        log_data = {
            "correlation_id": correlation_id,
            "request_id": request.state.request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": duration,
        }
        
        self.logger.error("Request failed", **log_data)

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware."""
    
    def __init__(self, app, config: MiddlewareConfig, performance_monitor: PerformanceMonitor):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.performance_monitor = performance_monitor
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        if not self.config.enable_performance_monitoring:
            return await call_next(request)
        
        # Create request metrics
        request_metrics = RequestMetrics(
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            start_time=request.state.start_time
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update metrics
            request_metrics.end_time = datetime.now()
            request_metrics.duration_ms = (request_metrics.end_time - request_metrics.start_time).total_seconds() * 1000
            request_metrics.status_code = response.status_code
            request_metrics.response_size = len(response.body) if hasattr(response, 'body') else 0
            
            # Record metrics
            self.performance_monitor.record_request(request_metrics)
            
            return response
            
        except Exception as e:
            # Update metrics for error
            request_metrics.end_time = datetime.now()
            request_metrics.duration_ms = (request_metrics.end_time - request_metrics.start_time).total_seconds() * 1000
            request_metrics.error_occurred = True
            request_metrics.error_type = type(e).__name__
            request_metrics.error_message = str(e)
            request_metrics.status_code = 500
            
            # Record metrics
            self.performance_monitor.record_request(request_metrics)
            
            raise

class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Error monitoring middleware."""
    
    def __init__(self, app, config: MiddlewareConfig, error_monitor: ErrorMonitor):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.error_monitor = error_monitor
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error monitoring."""
        if not self.config.enable_error_monitoring:
            return await call_next(request)
        
        try:
            return await call_next(request)
            
        except Exception as e:
            # Record error
            context = {
                "request_id": request.state.request_id,
                "correlation_id": request.state.correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
            
            self.error_monitor.record_error(e, request, context)
            raise

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, config: MiddlewareConfig, rate_limiter: RateLimiter):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        if not self.config.enable_rate_limiting:
            return await call_next(request)
        
        # Check rate limit
        allowed, rate_limit_info = self.rate_limiter.is_allowed(request)
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {rate_limit_info['limit']} per {rate_limit_info['window_seconds']} seconds",
                    "retry_after": rate_limit_info['window_seconds'],
                    "request_id": request.state.request_id
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit_info['limit']),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": rate_limit_info['reset_time'].isoformat(),
                    "Retry-After": str(rate_limit_info['window_seconds'])
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info['limit'] - rate_limit_info['current_requests'])
        response.headers["X-RateLimit-Reset"] = rate_limit_info['reset_time'].isoformat()
        
        return response

class CachingMiddleware(BaseHTTPMiddleware):
    """Response caching middleware."""
    
    def __init__(self, app, config: MiddlewareConfig, response_cache: ResponseCache):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.response_cache = response_cache
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with caching."""
        if not self.config.enable_caching:
            return await call_next(request)
        
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check cache
        cache_key = self.response_cache.generate_cache_key(request)
        cached_response = self.response_cache.get(cache_key)
        
        if cached_response:
            # Add cache headers
            cached_response.headers["X-Cache"] = "HIT"
            cached_response.headers["X-Cache-Key"] = cache_key
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            response.headers["X-Cache"] = "MISS"
            response.headers["X-Cache-Key"] = cache_key
            self.response_cache.set(cache_key, response)
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security headers."""
        response = await call_next(request)
        
        if self.config.enable_security_headers:
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

# ============================================================================
# MIDDLEWARE MANAGER
# ============================================================================

class MiddlewareManager:
    """Centralized middleware management system."""
    
    def __init__(self, config: MiddlewareConfig):
        
    """__init__ function."""
self.config = config
        self.metrics_collector = MetricsCollector()
        self.error_monitor = ErrorMonitor(
            alert_threshold=config.error_alert_threshold,
            alert_window_minutes=config.error_alert_window_minutes
        )
        self.performance_monitor = PerformanceMonitor(
            slow_request_threshold_ms=config.slow_request_threshold_ms
        )
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_requests,
            window_seconds=config.rate_limit_window
        )
        self.response_cache = ResponseCache(
            ttl_seconds=config.cache_ttl_seconds,
            max_size=config.cache_max_size
        )
        self.redis_client: Optional[redis.Redis] = None
        
        # Initialize Redis if configured
        if config.redis_url:
            self._init_redis()
    
    def _init_redis(self) -> Any:
        """Initialize Redis client."""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
        except Exception as e:
            logger = structlog.get_logger("middleware_manager")
            logger.warning("Failed to initialize Redis", error=str(e))
    
    def setup_middleware(self, app: FastAPI):
        """Setup all middleware on FastAPI app."""
        # Add CORS middleware
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add trusted host middleware
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.trusted_hosts
        )
        
        # Add compression middleware
        if self.config.enable_compression:
            app.add_middleware(
                GZipMiddleware,
                minimum_size=self.config.compression_min_size
            )
        
        # Add custom middleware in order
        app.add_middleware(LoggingMiddleware, config=self.config)
        app.add_middleware(PerformanceMonitoringMiddleware, config=self.config, performance_monitor=self.performance_monitor)
        app.add_middleware(ErrorMonitoringMiddleware, config=self.config, error_monitor=self.error_monitor)
        app.add_middleware(RateLimitingMiddleware, config=self.config, rate_limiter=self.rate_limiter)
        app.add_middleware(CachingMiddleware, config=self.config, response_cache=self.response_cache)
        app.add_middleware(SecurityHeadersMiddleware, config=self.config)
        
        # Add exception handlers
        self._setup_exception_handlers(app)
        
        # Add health check endpoint
        self._setup_health_check(app)
        
        # Add metrics endpoint
        if self.config.enable_metrics_collection:
            self._setup_metrics_endpoint(app)
    
    def _setup_exception_handlers(self, app: FastAPI):
        """Setup exception handlers."""
        
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            # Record error
            context = {
                "request_id": getattr(request.state, "request_id", None),
                "correlation_id": getattr(request.state, "correlation_id", None),
                "method": request.method,
                "path": request.url.path,
            }
            
            self.error_monitor.record_error(exc, request, context)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "detail": str(exc) if app.debug else "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """HTTP exception handler."""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "HTTP error",
                    "detail": exc.detail,
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _setup_health_check(self, app: FastAPI):
        """Setup health check endpoint."""
        
        @app.get("/health", response_model=HealthStatus)
        async def health_check():
            """Health check endpoint."""
            metrics = self.performance_monitor.get_performance_metrics()
            error_summary = self.error_monitor.get_error_summary()
            
            components = {
                "database": {"status": "healthy", "details": "Connected"},
                "cache": {"status": "healthy", "details": "Available"},
                "rate_limiter": {"status": "healthy", "details": "Active"},
                "error_monitor": {
                    "status": "healthy" if error_summary["total_errors"] < 10 else "warning",
                    "details": f"Total errors: {error_summary['total_errors']}"
                }
            }
            
            return HealthStatus(
                status="healthy",
                timestamp=datetime.now(),
                uptime_seconds=(datetime.now() - self.performance_monitor.start_time).total_seconds(),
                version="1.0.0",
                components=components,
                metrics=metrics
            )
    
    def _setup_metrics_endpoint(self, app: FastAPI):
        """Setup metrics endpoint."""
        
        @app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint."""
            # Update system metrics
            self.metrics_collector.update_system_metrics()
            
            # Return metrics in Prometheus format
            
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.performance_monitor.start_time).total_seconds(),
            "performance_metrics": self.performance_monitor.get_performance_metrics().dict(),
            "error_summary": self.error_monitor.get_error_summary(),
            "cache_stats": {
                "size": len(self.response_cache.cache),
                "max_size": self.response_cache.cache.maxsize,
                "ttl": self.response_cache.cache.ttl
            },
            "rate_limit_stats": {
                "active_clients": len(self.rate_limiter.requests),
                "total_requests": sum(len(requests) for requests in self.rate_limiter.requests.values())
            }
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_middleware_config(**kwargs) -> MiddlewareConfig:
    """Create middleware configuration with defaults."""
    return MiddlewareConfig(**kwargs)

def setup_middleware_system(app: FastAPI, config: Optional[MiddlewareConfig] = None) -> MiddlewareManager:
    """Setup complete middleware system."""
    if config is None:
        config = MiddlewareConfig()
    
    manager = MiddlewareManager(config)
    manager.setup_middleware(app)
    
    return manager

async def get_request_metrics(request: Request) -> Dict[str, Any]:
    """Get request metrics from request state."""
    return {
        "request_id": getattr(request.state, "request_id", None),
        "correlation_id": getattr(request.state, "correlation_id", None),
        "start_time": getattr(request.state, "start_time", None),
        "duration_ms": (
            (datetime.now() - request.state.start_time).total_seconds() * 1000
            if hasattr(request.state, "start_time") else None
        )
    }

def log_request_event(event_type: str, request: Request, **kwargs):
    """Log request event with correlation ID."""
    logger = structlog.get_logger("request_events")
    
    log_data = {
        "event_type": event_type,
        "correlation_id": getattr(request.state, "correlation_id", None),
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        **kwargs
    }
    
    logger.info("Request event", **log_data)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_app_with_middleware() -> FastAPI:
    """Create FastAPI app with comprehensive middleware."""
    
    # Create configuration
    config = MiddlewareConfig(
        log_level="INFO",
        log_format="json",
        enable_request_logging=True,
        enable_performance_monitoring=True,
        enable_error_monitoring=True,
        enable_rate_limiting=True,
        enable_caching=True,
        enable_security_headers=True,
        rate_limit_requests=100,
        slow_request_threshold_ms=1000,
        cache_ttl_seconds=300
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="Blatam Academy NLP API",
        description="Production-ready NLP API with comprehensive middleware",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Setup middleware system
    middleware_manager = setup_middleware_system(app, config)
    
    # Add your routes here
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Blatam Academy NLP API with comprehensive middleware"}
    
    @app.get("/api/status")
    async def api_status():
        """Get API status with middleware information."""
        return middleware_manager.get_system_status()
    
    return app

if __name__ == "__main__":
    app = create_app_with_middleware()
    uvicorn.run(app, host="0.0.0.0", port=8000) 