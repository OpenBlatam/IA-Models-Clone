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
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta
import weakref
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp
from pydantic import BaseModel, Field, validator
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import redis.asyncio as redis
from .http_exception_system import (
from .http_response_models import (
from .error_system import (
    from fastapi import FastAPI
from typing import Any, List, Dict, Optional
"""
🔧 Enhanced Middleware System
============================

Comprehensive middleware system for handling unexpected errors, logging, and error monitoring.
Integrates with the HTTPException system and provides advanced observability features.
"""



# Import our HTTPException system
    OnyxHTTPException, HTTPExceptionFactory, HTTPExceptionMapper,
    HTTPExceptionHandler, http_exception_handler
)
    ErrorResponse, ErrorDetail, ResponseFactory
)
    OnyxBaseError, ValidationError, AuthenticationError, AuthorizationError,
    ResourceNotFoundError, BusinessLogicError, DatabaseError,
    ErrorContext, ErrorFactory, ErrorSeverity, ErrorCategory
)

logger = structlog.get_logger(__name__)

# =============================================================================
# ENHANCED CONFIGURATION MODELS
# =============================================================================

class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling middleware."""
    
    # Error handling
    catch_unexpected_errors: bool = Field(default=True, description="Catch unexpected errors")
    log_full_traceback: bool = Field(default=True, description="Log full traceback for errors")
    sanitize_error_messages: bool = Field(default=True, description="Sanitize error messages in responses")
    include_error_codes: bool = Field(default=True, description="Include error codes in responses")
    
    # Error monitoring
    error_sampling_rate: float = Field(default=1.0, description="Error sampling rate (0.0-1.0)")
    error_retention_days: int = Field(default=30, description="Error retention in days")
    error_alert_threshold: int = Field(default=10, description="Error alert threshold per minute")
    
    # Error categorization
    categorize_errors: bool = Field(default=True, description="Automatically categorize errors")
    track_error_patterns: bool = Field(default=True, description="Track error patterns")
    error_pattern_window: int = Field(default=3600, description="Error pattern window in seconds")
    
    # Performance thresholds
    slow_request_threshold: float = Field(default=1.0, description="Slow request threshold in seconds")
    critical_request_threshold: float = Field(default=5.0, description="Critical request threshold in seconds")
    
    @validator('error_sampling_rate')
    def validate_sampling_rate(cls, v) -> bool:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Error sampling rate must be between 0.0 and 1.0')
        return v

class LoggingConfig(BaseModel):
    """Configuration for logging middleware."""
    
    # Request logging
    log_requests: bool = Field(default=True, description="Log all requests")
    log_responses: bool = Field(default=True, description="Log all responses")
    log_errors: bool = Field(default=True, description="Log all errors")
    
    # Request details
    log_request_headers: bool = Field(default=False, description="Log request headers")
    log_request_body: bool = Field(default=False, description="Log request body")
    log_response_headers: bool = Field(default=False, description="Log response headers")
    log_response_body: bool = Field(default=False, description="Log response body")
    
    # Sensitive data handling
    sensitive_headers: List[str] = Field(
        default=["authorization", "cookie", "x-api-key", "x-auth-token"],
        description="Headers to mask in logs"
    )
    sensitive_fields: List[str] = Field(
        default=["password", "token", "secret", "key"],
        description="Request body fields to mask in logs"
    )
    
    # Log levels
    request_log_level: str = Field(default="INFO", description="Request log level")
    error_log_level: str = Field(default="ERROR", description="Error log level")
    performance_log_level: str = Field(default="WARNING", description="Performance log level")
    
    # Structured logging
    use_structured_logging: bool = Field(default=True, description="Use structured logging")
    include_request_id: bool = Field(default=True, description="Include request ID in logs")
    include_user_context: bool = Field(default=True, description="Include user context in logs")

class MonitoringConfig(BaseModel):
    """Configuration for monitoring middleware."""
    
    # Metrics collection
    collect_metrics: bool = Field(default=True, description="Collect performance metrics")
    metrics_prefix: str = Field(default="blatam_academy", description="Metrics prefix")
    
    # Performance monitoring
    track_response_times: bool = Field(default=True, description="Track response times")
    track_memory_usage: bool = Field(default=True, description="Track memory usage")
    track_cpu_usage: bool = Field(default=True, description="Track CPU usage")
    
    # Error monitoring
    track_error_rates: bool = Field(default=True, description="Track error rates")
    track_error_types: bool = Field(default=True, description="Track error types")
    track_slow_requests: bool = Field(default=True, description="Track slow requests")
    
    # Alerting
    enable_alerts: bool = Field(default=True, description="Enable performance alerts")
    alert_thresholds: Dict[str, float] = Field(
        default={
            "error_rate": 0.05,  # 5% error rate
            "response_time_p95": 2.0,  # 95th percentile response time
            "memory_usage": 0.8,  # 80% memory usage
        },
        description="Alert thresholds"
    )

class EnhancedMiddlewareConfig(BaseModel):
    """Enhanced configuration for the complete middleware system."""
    
    # Core settings
    enabled: bool = Field(default=True, description="Enable enhanced middleware")
    environment: str = Field(default="production", description="Environment (production/development)")
    
    # Component configurations
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # External services
    redis_enabled: bool = Field(default=False, description="Enable Redis for caching and rate limiting")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_ttl: int = Field(default=3600, description="Redis TTL in seconds")
    
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

# =============================================================================
# ENHANCED METRICS
# =============================================================================

class EnhancedMetrics:
    """Enhanced metrics collection with error tracking."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.prefix = config.metrics_prefix
        
        # Request metrics
        self.request_total = Counter(
            f'{self.prefix}_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'error_type']
        )
        
        self.request_duration = Histogram(
            f'{self.prefix}_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code']
        )
        
        # Error metrics
        self.error_total = Counter(
            f'{self.prefix}_http_errors_total',
            'Total HTTP errors',
            ['method', 'endpoint', 'error_category', 'error_code']
        )
        
        self.error_rate = Gauge(
            f'{self.prefix}_http_error_rate',
            'HTTP error rate percentage',
            ['method', 'endpoint']
        )
        
        # Performance metrics
        self.response_size = Histogram(
            f'{self.prefix}_http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            f'{self.prefix}_http_active_requests',
            'Number of active HTTP requests',
            ['method', 'endpoint']
        )
        
        # System metrics
        self.memory_usage = Gauge(
            f'{self.prefix}_app_memory_usage_bytes',
            'Application memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            f'{self.prefix}_app_cpu_usage_percent',
            'Application CPU usage percentage'
        )
        
        # Error pattern metrics
        self.error_patterns = Counter(
            f'{self.prefix}_error_patterns_total',
            'Error patterns detected',
            ['pattern_type', 'error_category']
        )
        
        # Slow request metrics
        self.slow_requests = Counter(
            f'{self.prefix}_slow_requests_total',
            'Total slow requests',
            ['method', 'endpoint', 'duration_range']
        )

# =============================================================================
# ENHANCED REQUEST CONTEXT
# =============================================================================

@dataclass
class EnhancedRequestContext:
    """Enhanced context for tracking request information."""
    request_id: str
    start_time: float
    method: str
    url: str
    endpoint: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    status_code: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[Exception] = None
    error_category: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_data: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# ENHANCED ERROR HANDLING MIDDLEWARE
# =============================================================================

class EnhancedErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced error handling middleware that catches unexpected errors
    and converts them to proper HTTP exceptions.
    """
    
    def __init__(
        self, 
        app: ASGIApp, 
        config: EnhancedMiddlewareConfig,
        metrics: Optional[EnhancedMetrics] = None
    ):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.metrics = metrics
        self.logger = structlog.get_logger(__name__)
        self.error_handler = HTTPExceptionHandler()
        self.error_counts = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.last_error_reset = time.time()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with comprehensive error handling."""
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Create enhanced context
        context = EnhancedRequestContext(
            request_id=request_id,
            start_time=start_time,
            method=request.method,
            url=str(request.url),
            endpoint=self._get_endpoint(request),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            user_id=request.headers.get("X-User-ID"),
            session_id=request.headers.get("X-Session-ID")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update context
            context.status_code = response.status_code
            context.duration = time.time() - start_time
            context.response_size = self._get_response_size(response)
            
            # Record metrics
            if self.metrics:
                self._record_success_metrics(context, response)
            
            # Log request
            self._log_request(context, response)
            
            return response
            
        except OnyxHTTPException as e:
            # Handle Onyx HTTP exceptions
            return await self._handle_onyx_http_exception(request, e, context)
            
        except OnyxBaseError as e:
            # Convert Onyx errors to HTTP exceptions
            return await self._handle_onyx_error(request, e, context)
            
        except RequestValidationError as e:
            # Handle validation errors
            return await self._handle_validation_error(request, e, context)
            
        except Exception as e:
            # Handle unexpected errors
            return await self._handle_unexpected_error(request, e, context)
    
    async async def _handle_onyx_http_exception(
        self, 
        request: Request, 
        exc: OnyxHTTPException, 
        context: EnhancedRequestContext
    ) -> JSONResponse:
        """Handle Onyx HTTP exceptions."""
        context.error = exc
        context.status_code = exc.status_code
        context.duration = time.time() - context.start_time
        
        # Extract error details
        error_detail = exc.detail.get("error", {})
        context.error_category = error_detail.get("category", "unknown")
        context.error_code = error_detail.get("error_code", "UNKNOWN")
        
        # Record metrics
        if self.metrics:
            self._record_error_metrics(context, exc)
        
        # Log error
        self._log_error(context, exc)
        
        # Return response
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers
        )
    
    async def _handle_onyx_error(
        self, 
        request: Request, 
        exc: OnyxBaseError, 
        context: EnhancedRequestContext
    ) -> JSONResponse:
        """Handle Onyx base errors."""
        # Convert to HTTP exception
        http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(exc)
        
        # Update context
        context.error = exc
        context.error_category = exc.category.value
        context.error_code = exc.error_code
        context.status_code = http_exception.status_code
        context.duration = time.time() - context.start_time
        
        # Record metrics
        if self.metrics:
            self._record_error_metrics(context, http_exception)
        
        # Log error
        self._log_error(context, exc)
        
        # Return response
        return JSONResponse(
            status_code=http_exception.status_code,
            content=http_exception.detail,
            headers=http_exception.headers
        )
    
    async def _handle_validation_error(
        self, 
        request: Request, 
        exc: RequestValidationError, 
        context: EnhancedRequestContext
    ) -> JSONResponse:
        """Handle request validation errors."""
        # Create HTTP exception
        http_exception = HTTPExceptionFactory.unprocessable_entity(
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            validation_errors=[f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in exc.errors()],
            additional_data={
                "raw_errors": exc.errors(),
                "body": exc.body
            }
        )
        
        # Update context
        context.error = exc
        context.error_category = "validation"
        context.error_code = "VALIDATION_ERROR"
        context.status_code = http_exception.status_code
        context.duration = time.time() - context.start_time
        
        # Record metrics
        if self.metrics:
            self._record_error_metrics(context, http_exception)
        
        # Log error
        self._log_error(context, exc)
        
        # Return response
        return JSONResponse(
            status_code=http_exception.status_code,
            content=http_exception.detail,
            headers=http_exception.headers
        )
    
    async def _handle_unexpected_error(
        self, 
        request: Request, 
        exc: Exception, 
        context: EnhancedRequestContext
    ) -> JSONResponse:
        """Handle unexpected errors."""
        # Convert to HTTP exception
        http_exception = HTTPExceptionMapper.map_exception_to_http_exception(exc)
        
        # Update context
        context.error = exc
        context.error_category = "unexpected"
        context.error_code = "UNEXPECTED_ERROR"
        context.status_code = http_exception.status_code
        context.duration = time.time() - context.start_time
        
        # Record metrics
        if self.metrics:
            self._record_error_metrics(context, http_exception)
        
        # Log error
        self._log_error(context, exc)
        
        # Update error counts and patterns
        self._update_error_tracking(context)
        
        # Return response
        return JSONResponse(
            status_code=http_exception.status_code,
            content=http_exception.detail,
            headers=http_exception.headers
        )
    
    def _record_success_metrics(self, context: EnhancedRequestContext, response: Response):
        """Record success metrics."""
        self.metrics.request_total.labels(
            method=context.method,
            endpoint=context.endpoint,
            status_code=context.status_code,
            error_type="none"
        ).inc()
        
        self.metrics.request_duration.labels(
            method=context.method,
            endpoint=context.endpoint,
            status_code=context.status_code
        ).observe(context.duration)
        
        self.metrics.response_size.labels(
            method=context.method,
            endpoint=context.endpoint
        ).observe(context.response_size)
    
    def _record_error_metrics(self, context: EnhancedRequestContext, exc: Union[OnyxHTTPException, Exception]):
        """Record error metrics."""
        error_detail = exc.detail.get("error", {}) if hasattr(exc, 'detail') else {}
        error_category = error_detail.get("category", context.error_category or "unknown")
        error_code = error_detail.get("error_code", context.error_code or "UNKNOWN")
        
        self.metrics.request_total.labels(
            method=context.method,
            endpoint=context.endpoint,
            status_code=context.status_code,
            error_type=error_category
        ).inc()
        
        self.metrics.error_total.labels(
            method=context.method,
            endpoint=context.endpoint,
            error_category=error_category,
            error_code=error_code
        ).inc()
        
        # Track slow requests
        if context.duration and context.duration > self.config.error_handling.slow_request_threshold:
            duration_range = self._get_duration_range(context.duration)
            self.metrics.slow_requests.labels(
                method=context.method,
                endpoint=context.endpoint,
                duration_range=duration_range
            ).inc()
    
    def _update_error_tracking(self, context: EnhancedRequestContext):
        """Update error tracking for patterns and alerts."""
        current_time = time.time()
        
        # Reset counters if needed
        if current_time - self.last_error_reset > 60:  # Reset every minute
            self.error_counts.clear()
            self.last_error_reset = current_time
        
        # Update error counts
        error_key = f"{context.method}:{context.endpoint}:{context.error_code}"
        self.error_counts[error_key] += 1
        
        # Check for error patterns
        if self.config.error_handling.track_error_patterns:
            pattern_key = f"{context.error_category}:{context.error_code}"
            self.error_patterns[pattern_key] += 1
            
            if self.metrics:
                self.metrics.error_patterns.labels(
                    pattern_type=context.error_category,
                    error_category=context.error_code
                ).inc()
        
        # Check alert threshold
        total_errors = sum(self.error_counts.values())
        if total_errors > self.config.error_handling.error_alert_threshold:
            self._send_error_alert(total_errors, current_time)
    
    def _send_error_alert(self, error_count: int, timestamp: float):
        """Send error alert."""
        alert_message = {
            "type": "error_alert",
            "error_count": error_count,
            "timestamp": timestamp,
            "threshold": self.config.error_handling.error_alert_threshold,
            "message": f"High error rate detected: {error_count} errors in the last minute"
        }
        
        self.logger.error("Error alert triggered", **alert_message)
        
        # Here you could integrate with external alerting systems
        # like Slack, PagerDuty, etc.
    
    def _log_request(self, context: EnhancedRequestContext, response: Response):
        """Log successful request."""
        if not self.config.logging.log_requests:
            return
        
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "endpoint": context.endpoint,
            "status_code": context.status_code,
            "duration": context.duration,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
            "request_size": context.request_size,
            "response_size": context.response_size,
        }
        
        if context.user_id:
            log_data["user_id"] = context.user_id
        
        self.logger.info("Request completed", **log_data)
    
    def _log_error(self, context: EnhancedRequestContext, exc: Exception):
        """Log error with full context."""
        if not self.config.logging.log_errors:
            return
        
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "endpoint": context.endpoint,
            "status_code": context.status_code,
            "duration": context.duration,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_category": context.error_category,
            "error_code": context.error_code,
        }
        
        if context.user_id:
            log_data["user_id"] = context.user_id
        
        if self.config.error_handling.log_full_traceback:
            log_data["traceback"] = traceback.format_exc()
        
        self.logger.error("Request failed", **log_data)
    
    def _get_endpoint(self, request: Request) -> str:
        """Extract endpoint from request."""
        return request.url.path
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else None
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        if hasattr(response, 'body'):
            return len(response.body) if response.body else 0
        return 0
    
    def _get_duration_range(self, duration: float) -> str:
        """Get duration range for metrics."""
        if duration < 1.0:
            return "0-1s"
        elif duration < 5.0:
            return "1-5s"
        elif duration < 10.0:
            return "5-10s"
        else:
            return "10s+"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring."""
        return {
            "error_counts": dict(self.error_counts),
            "error_patterns": dict(self.error_patterns),
            "total_errors": sum(self.error_counts.values()),
            "last_reset": self.last_error_reset
        }

# =============================================================================
# ENHANCED LOGGING MIDDLEWARE
# =============================================================================

class EnhancedLoggingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced logging middleware with structured logging and context tracking.
    """
    
    def __init__(self, app: ASGIApp, config: EnhancedMiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.request_logger = structlog.get_logger("request")
        self.error_logger = structlog.get_logger("error")
        self.performance_logger = structlog.get_logger("performance")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with enhanced logging."""
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Create context
        context = EnhancedRequestContext(
            request_id=request_id,
            start_time=start_time,
            method=request.method,
            url=str(request.url),
            endpoint=self._get_endpoint(request),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            user_id=request.headers.get("X-User-ID"),
            session_id=request.headers.get("X-Session-ID")
        )
        
        # Log request
        if self.config.logging.log_requests:
            self._log_request_start(context, request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update context
            context.status_code = response.status_code
            context.duration = time.time() - start_time
            context.response_size = self._get_response_size(response)
            
            # Log response
            if self.config.logging.log_responses:
                self._log_request_success(context, response)
            
            return response
            
        except Exception as e:
            # Update context
            context.error = e
            context.duration = time.time() - start_time
            
            # Log error
            if self.config.logging.log_errors:
                self._log_request_error(context, e)
            
            raise
    
    def _log_request_start(self, context: EnhancedRequestContext, request: Request):
        """Log request start."""
        log_data = self._build_log_data(context, request)
        self.request_logger.info("Request started", **log_data)
    
    def _log_request_success(self, context: EnhancedRequestContext, response: Response):
        """Log successful request completion."""
        log_data = self._build_log_data(context, response=response)
        
        # Add performance data
        if context.duration:
            log_data["duration_ms"] = round(context.duration * 1000, 2)
            
            # Log slow requests
            if context.duration > self.config.error_handling.slow_request_threshold:
                self.performance_logger.warning("Slow request detected", **log_data)
        
        self.request_logger.info("Request completed", **log_data)
    
    def _log_request_error(self, context: EnhancedRequestContext, error: Exception):
        """Log request error."""
        log_data = self._build_log_data(context, error=error)
        log_data.update({
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
        
        if self.config.error_handling.log_full_traceback:
            log_data["traceback"] = traceback.format_exc()
        
        self.error_logger.error("Request failed", **log_data)
    
    def _build_log_data(
        self, 
        context: EnhancedRequestContext, 
        request: Optional[Request] = None,
        response: Optional[Response] = None,
        error: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Build structured log data."""
        log_data = {
            "request_id": context.request_id,
            "method": context.method,
            "endpoint": context.endpoint,
            "url": context.url,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
        }
        
        if context.user_id:
            log_data["user_id"] = context.user_id
        
        if context.session_id:
            log_data["session_id"] = context.session_id
        
        # Add request details
        if request and self.config.logging.log_request_headers:
            log_data["request_headers"] = self._sanitize_headers(dict(request.headers))
        
        if request and self.config.logging.log_request_body:
            log_data["request_body"] = self._sanitize_body(request)
        
        # Add response details
        if response:
            log_data["status_code"] = response.status_code
            log_data["response_size"] = context.response_size
            
            if self.config.logging.log_response_headers:
                log_data["response_headers"] = dict(response.headers)
        
        return log_data
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers."""
        sanitized = headers.copy()
        for sensitive in self.config.logging.sensitive_headers:
            if sensitive.lower() in sanitized:
                sanitized[sensitive] = "[REDACTED]"
        return sanitized
    
    def _sanitize_body(self, request: Request) -> Any:
        """Sanitize request body."""
        # This is a simplified version - in production you'd want to parse JSON
        # and remove sensitive fields
        return "[BODY_CONTENT]"
    
    def _get_endpoint(self, request: Request) -> str:
        """Extract endpoint from request."""
        return request.url.path
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else None
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        if hasattr(response, 'body'):
            return len(response.body) if response.body else 0
        return 0

# =============================================================================
# ENHANCED MIDDLEWARE MANAGER
# =============================================================================

class EnhancedMiddlewareManager:
    """
    Enhanced middleware manager for comprehensive error handling, logging, and monitoring.
    """
    
    def __init__(self, config: EnhancedMiddlewareConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.config = config
        self.redis_client = redis_client
        self.metrics = EnhancedMetrics(config.monitoring) if config.monitoring.collect_metrics else None
        self.logger = structlog.get_logger(__name__)
        
        # Initialize middleware components
        self.error_middleware = None
        self.logging_middleware = None
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def setup_middleware(self, app: FastAPI) -> None:
        """Setup all enhanced middleware for the application."""
        if not self.config.enabled:
            self.logger.info("Enhanced middleware disabled")
            return
        
        self.logger.info("Setting up enhanced middleware", config=self.config.dict())
        
        # Setup CORS middleware
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=self.config.cors_credentials,
                allow_methods=self.config.cors_methods,
                allow_headers=self.config.cors_headers
            )
        
        # Setup compression middleware
        if self.config.compression_enabled:
            app.add_middleware(
                GZipMiddleware,
                minimum_size=self.config.compression_min_size
            )
        
        # Setup trusted hosts middleware
        if self.config.trusted_hosts_enabled:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.trusted_hosts
            )
        
        # Setup custom middleware (in order of execution)
        
        # 1. Logging middleware (first to capture everything)
        if self.config.logging.log_requests or self.config.logging.log_responses:
            self.logging_middleware = EnhancedLoggingMiddleware(app, self.config)
            app.add_middleware(EnhancedLoggingMiddleware, config=self.config)
        
        # 2. Error handling middleware (catches and processes errors)
        if self.config.error_handling.catch_unexpected_errors:
            self.error_middleware = EnhancedErrorHandlingMiddleware(app, self.config, self.metrics)
            app.add_middleware(EnhancedErrorHandlingMiddleware, config=self.config, metrics=self.metrics)
        
        # 3. Security middleware
        if self.config.security_enabled:
            app.add_middleware(SecurityMiddleware, config=self.config)
        
        # 4. Rate limiting middleware
        if self.config.rate_limiting_enabled:
            app.add_middleware(RateLimitingMiddleware, config=self.config, redis_client=self.redis_client)
        
        self.logger.info("Enhanced middleware setup completed")
    
    def get_metrics(self) -> Optional[str]:
        """Get Prometheus metrics."""
        if self.metrics:
            return generate_latest(REGISTRY)
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "metrics_available": True
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.error_middleware:
            return {}
        
        return self.error_middleware.get_error_summary()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - self.start_time,
            "middleware_enabled": self.config.enabled,
            "metrics_enabled": self.metrics is not None,
            "redis_connected": self.redis_client is not None
        }

# =============================================================================
# CONFIGURATION FACTORIES
# =============================================================================

def create_enhanced_middleware_config(
    environment: str = "production",
    logging_enabled: bool = True,
    error_handling_enabled: bool = True,
    monitoring_enabled: bool = True,
    **kwargs
) -> EnhancedMiddlewareConfig:
    """Create enhanced middleware configuration with sensible defaults."""
    return EnhancedMiddlewareConfig(
        environment=environment,
        logging=LoggingConfig(
            log_requests=logging_enabled,
            log_errors=logging_enabled,
            use_structured_logging=True
        ),
        error_handling=ErrorHandlingConfig(
            catch_unexpected_errors=error_handling_enabled,
            log_full_traceback=True,
            sanitize_error_messages=True
        ),
        monitoring=MonitoringConfig(
            collect_metrics=monitoring_enabled,
            track_response_times=True,
            track_error_rates=True
        ),
        **kwargs
    )

def create_production_enhanced_config() -> EnhancedMiddlewareConfig:
    """Create production-optimized enhanced middleware configuration."""
    return EnhancedMiddlewareConfig(
        environment="production",
        logging=LoggingConfig(
            log_requests=True,
            log_responses=False,  # Reduce log volume in production
            log_errors=True,
            log_request_headers=False,
            log_request_body=False,
            use_structured_logging=True
        ),
        error_handling=ErrorHandlingConfig(
            catch_unexpected_errors=True,
            log_full_traceback=True,
            sanitize_error_messages=True,
            error_sampling_rate=1.0,
            error_alert_threshold=20
        ),
        monitoring=MonitoringConfig(
            collect_metrics=True,
            track_response_times=True,
            track_error_rates=True,
            enable_alerts=True,
            alert_thresholds={
                "error_rate": 0.05,
                "response_time_p95": 2.0,
                "memory_usage": 0.8
            }
        ),
        security_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=100,
        cors_enabled=True,
        compression_enabled=True
    )

def create_development_enhanced_config() -> EnhancedMiddlewareConfig:
    """Create development-optimized enhanced middleware configuration."""
    return EnhancedMiddlewareConfig(
        environment="development",
        logging=LoggingConfig(
            log_requests=True,
            log_responses=True,
            log_errors=True,
            log_request_headers=True,
            log_request_body=True,
            use_structured_logging=True
        ),
        error_handling=ErrorHandlingConfig(
            catch_unexpected_errors=True,
            log_full_traceback=True,
            sanitize_error_messages=False,  # Show full errors in development
            error_sampling_rate=1.0,
            error_alert_threshold=5
        ),
        monitoring=MonitoringConfig(
            collect_metrics=True,
            track_response_times=True,
            track_error_rates=True,
            enable_alerts=False  # No alerts in development
        ),
        security_enabled=False,  # Disable security in development
        rate_limiting_enabled=False,  # Disable rate limiting in development
        cors_enabled=True,
        compression_enabled=False  # Disable compression in development
    )

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_enhanced_middleware(
    app: FastAPI,
    config: Optional[EnhancedMiddlewareConfig] = None,
    redis_client: Optional[redis.Redis] = None
) -> EnhancedMiddlewareManager:
    """
    Setup enhanced middleware for a FastAPI application.
    
    Args:
        app: FastAPI application
        config: Middleware configuration (optional)
        redis_client: Redis client for caching and rate limiting (optional)
        
    Returns:
        EnhancedMiddlewareManager instance
    """
    if config is None:
        config = create_enhanced_middleware_config()
    
    manager = EnhancedMiddlewareManager(config, redis_client)
    manager.setup_middleware(app)
    
    return manager

# Example usage
def example_usage():
    """Example of how to use the enhanced middleware system."""
    
    
    # Create app
    app = FastAPI(title="Enhanced Middleware Example")
    
    # Create configuration
    config = create_production_enhanced_config()
    
    # Setup middleware
    manager = setup_enhanced_middleware(app, config)
    
    # Add endpoints
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Enhanced middleware example"}
    
    @app.get("/health")
    async def health():
        
    """health function."""
return manager.get_health_status()
    
    @app.get("/metrics")
    async def metrics():
        
    """metrics function."""
return manager.get_metrics()
    
    @app.get("/performance")
    async def performance():
        
    """performance function."""
return manager.get_performance_summary()
    
    @app.get("/errors")
    async def errors():
        
    """errors function."""
return manager.get_error_summary()
    
    return app

match __name__:
    case "__main__":
    example_usage() 