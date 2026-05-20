from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import traceback
import sys
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uuid
import json
from dataclasses import dataclass, asdict
from enum import Enum
from ..exceptions.http_exceptions import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Error Handling Middleware for HeyGen AI API
Comprehensive middleware for handling unexpected errors, logging, and monitoring.
"""


    BaseHTTPException, InternalServerError, ErrorResponse,
    ErrorCategory, ErrorSeverity
)

logger = structlog.get_logger()

# =============================================================================
# Error Types and Classification
# =============================================================================

class ErrorType(Enum):
    """Error types for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

class ErrorPriority(Enum):
    """Error priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    request_id: str
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    request_body: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# =============================================================================
# Error Classifier
# =============================================================================

class ErrorClassifier:
    """Classify errors based on exception type and message."""
    
    def __init__(self) -> Any:
        self.error_patterns = {
            # Database errors
            "database": [
                "connection", "timeout", "deadlock", "constraint",
                "foreign key", "unique", "not null", "sql", "postgresql",
                "mysql", "sqlite", "oracle"
            ],
            # Network errors
            "network": [
                "connection refused", "connection reset", "timeout",
                "dns", "host unreachable", "network unreachable"
            ],
            # External service errors
            "external_service": [
                "api", "service unavailable", "bad gateway",
                "gateway timeout", "heygen", "openai", "aws"
            ],
            # Authentication errors
            "authentication": [
                "unauthorized", "invalid token", "expired token",
                "missing token", "authentication failed"
            ],
            # Authorization errors
            "authorization": [
                "forbidden", "access denied", "permission denied",
                "insufficient permissions"
            ],
            # Validation errors
            "validation": [
                "validation", "invalid", "missing", "required",
                "format", "type error"
            ],
            # Rate limit errors
            "rate_limit": [
                "rate limit", "too many requests", "quota exceeded",
                "throttle"
            ],
            # Memory errors
            "memory": [
                "memory", "out of memory", "memory error",
                "insufficient memory"
            ],
            # Configuration errors
            "configuration": [
                "config", "environment", "setting", "parameter",
                "configuration error"
            ],
            # Timeout errors
            "timeout": [
                "timeout", "timed out", "deadline exceeded"
            ]
        }
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """Classify error based on exception type and message."""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Check for specific exception types first
        if "database" in exception_type or "sql" in exception_type:
            return ErrorType.DATABASE
        
        if "timeout" in exception_type:
            return ErrorType.TIMEOUT
        
        if "memory" in exception_type:
            return ErrorType.MEMORY
        
        if "config" in exception_type:
            return ErrorType.CONFIGURATION
        
        # Check error patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in error_message:
                    return ErrorType(error_type)
        
        # Default classification
        return ErrorType.UNKNOWN
    
    def get_priority(self, error_type: ErrorType, exception: Exception) -> ErrorPriority:
        """Get error priority based on type and context."""
        # Critical errors
        if error_type in [ErrorType.MEMORY, ErrorType.DATABASE]:
            return ErrorPriority.CRITICAL
        
        # High priority errors
        if error_type in [ErrorType.EXTERNAL_SERVICE, ErrorType.NETWORK]:
            return ErrorPriority.HIGH
        
        # Medium priority errors
        if error_type in [ErrorType.TIMEOUT, ErrorType.CONFIGURATION]:
            return ErrorPriority.MEDIUM
        
        # Low priority errors
        if error_type in [ErrorType.VALIDATION, ErrorType.AUTHENTICATION, ErrorType.AUTHORIZATION]:
            return ErrorPriority.LOW
        
        return ErrorPriority.MEDIUM

# =============================================================================
# Error Monitor
# =============================================================================

class ErrorMonitor:
    """Monitor and track errors for alerting and analysis."""
    
    def __init__(self) -> Any:
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            ErrorPriority.CRITICAL: 1,  # Alert immediately
            ErrorPriority.HIGH: 5,       # Alert after 5 errors
            ErrorPriority.MEDIUM: 20,    # Alert after 20 errors
            ErrorPriority.LOW: 100       # Alert after 100 errors
        }
        self.time_window = timedelta(minutes=5)
        self.max_history = 1000
    
    def record_error(
        self,
        error_type: ErrorType,
        priority: ErrorPriority,
        exception: Exception,
        context: ErrorContext
    ):
        """Record an error for monitoring."""
        error_key = f"{error_type.value}_{priority.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to history
        error_record = {
            "timestamp": context.timestamp,
            "error_type": error_type.value,
            "priority": priority.value,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "request_id": context.request_id,
            "endpoint": context.endpoint,
            "method": context.method,
            "user_id": context.user_id
        }
        
        self.error_history.append(error_record)
        
        # Trim history if too long
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Check for alerts
        self._check_alerts(error_type, priority, error_key)
    
    def _check_alerts(self, error_type: ErrorType, priority: ErrorPriority, error_key: str):
        """Check if alerts should be triggered."""
        threshold = self.alert_thresholds.get(priority, 10)
        current_count = self.error_counts.get(error_key, 0)
        
        if current_count >= threshold:
            self._trigger_alert(error_type, priority, current_count, error_key)
    
    def _trigger_alert(self, error_type: ErrorType, priority: ErrorPriority, count: int, error_key: str):
        """Trigger an alert for high error rates."""
        logger.critical(
            "Error alert triggered",
            error_type=error_type.value,
            priority=priority.value,
            count=count,
            threshold=self.alert_thresholds[priority],
            error_key=error_key
        )
        
        # Reset counter after alert
        self.error_counts[error_key] = 0
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        recent_errors = [
            error for error in self.error_history
            if error["timestamp"] > datetime.now(timezone.utc) - self.time_window
        ]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_counts": self.error_counts,
            "error_types": {
                error_type.value: len([e for e in recent_errors if e["error_type"] == error_type.value])
                for error_type in ErrorType
            },
            "priorities": {
                priority.value: len([e for e in recent_errors if e["priority"] == priority.value])
                for priority in ErrorPriority
            }
        }
    
    def clear_history(self) -> Any:
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()

# =============================================================================
# Error Recovery Strategies
# =============================================================================

class ErrorRecovery:
    """Error recovery strategies and fallbacks."""
    
    def __init__(self) -> Any:
        self.recovery_strategies = {
            ErrorType.DATABASE: self._database_recovery,
            ErrorType.CACHE: self._cache_recovery,
            ErrorType.NETWORK: self._network_recovery,
            ErrorType.TIMEOUT: self._timeout_recovery,
            ErrorType.EXTERNAL_SERVICE: self._external_service_recovery
        }
    
    async def attempt_recovery(
        self,
        error_type: ErrorType,
        exception: Exception,
        context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover from error."""
        recovery_strategy = self.recovery_strategies.get(error_type)
        
        if recovery_strategy:
            try:
                return await recovery_strategy(exception, context)
            except Exception as recovery_error:
                logger.error(
                    "Recovery attempt failed",
                    original_error=str(exception),
                    recovery_error=str(recovery_error),
                    request_id=context.request_id
                )
        
        return None
    
    async def _database_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Database recovery strategy."""
        # Try to reconnect to database
        # Return cached data if available
        # Fall back to default values
        logger.info("Attempting database recovery", request_id=context.request_id)
        return {"status": "recovered", "source": "cache"}
    
    async def _cache_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Cache recovery strategy."""
        # Try alternative cache
        # Fall back to database
        logger.info("Attempting cache recovery", request_id=context.request_id)
        return {"status": "recovered", "source": "database"}
    
    async def _network_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Network recovery strategy."""
        # Retry with exponential backoff
        # Try alternative endpoints
        logger.info("Attempting network recovery", request_id=context.request_id)
        return {"status": "retry", "backoff": 5}
    
    async def _timeout_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Timeout recovery strategy."""
        # Increase timeout
        # Use cached results
        logger.info("Attempting timeout recovery", request_id=context.request_id)
        return {"status": "recovered", "source": "cache"}
    
    async def _external_service_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """External service recovery strategy."""
        # Try alternative service
        # Use cached data
        # Return degraded response
        logger.info("Attempting external service recovery", request_id=context.request_id)
        return {"status": "degraded", "message": "Using cached data"}

# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Comprehensive error handling middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_recovery: bool = True,
        enable_monitoring: bool = True,
        log_request_body: bool = False,
        log_headers: bool = False
    ):
        
    """__init__ function."""
super().__init__(app)
        self.enable_recovery = enable_recovery
        self.enable_monitoring = enable_monitoring
        self.log_request_body = log_request_body
        self.log_headers = log_headers
        
        self.error_classifier = ErrorClassifier()
        self.error_monitor = ErrorMonitor() if enable_monitoring else None
        self.error_recovery = ErrorRecovery() if enable_recovery else None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        start_time = datetime.now(timezone.utc)
        request_id = self._get_request_id(request)
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful request
            await self._log_successful_request(request, response, start_time)
            
            return response
            
        except BaseHTTPException as http_exc:
            # Handle known HTTP exceptions
            return await self._handle_http_exception(http_exc, request, start_time)
            
        except Exception as exc:
            # Handle unexpected exceptions
            return await self._handle_unexpected_exception(exc, request, start_time)
    
    async def _get_request_id(self, request: Request) -> str:
        """Get or generate request ID."""
        return (
            request.headers.get("X-Request-ID") or
            str(uuid.uuid4())
        )
    
    async def _log_successful_request(
        self,
        request: Request,
        response: Response,
        start_time: datetime
    ):
        """Log successful request."""
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(
            "Request completed successfully",
            request_id=request.state.request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_ms=duration * 1000,
            user_agent=request.headers.get("user-agent"),
            ip_address=self._get_client_ip(request)
        )
    
    async async def _handle_http_exception(
        self,
        exception: BaseHTTPException,
        request: Request,
        start_time: datetime
    ) -> JSONResponse:
        """Handle known HTTP exceptions."""
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log HTTP exception
        logger.warning(
            "HTTP exception raised",
            request_id=request.state.request_id,
            method=request.method,
            url=str(request.url),
            error_code=exception.error_code,
            status_code=exception.status_code,
            duration_ms=duration * 1000,
            message=exception.message
        )
        
        # Return error response
        error_response = exception.to_error_response()
        error_response.request_id = request.state.request_id
        
        return JSONResponse(
            status_code=exception.status_code,
            content=error_response.dict(),
            headers=exception.headers
        )
    
    async def _handle_unexpected_exception(
        self,
        exception: Exception,
        request: Request,
        start_time: datetime
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Create error context
        context = ErrorContext(
            request_id=request.state.request_id,
            endpoint=str(request.url.path),
            method=request.method,
            user_agent=request.headers.get("user-agent"),
            ip_address=self._get_client_ip(request),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add request details if enabled
        if self.log_request_body:
            try:
                body = await request.body()
                context.request_body = body.decode() if body else None
            except Exception:
                context.request_body = "[Unable to read request body]"
        
        if self.log_headers:
            context.headers = dict(request.headers)
        
        # Classify error
        error_type = self.error_classifier.classify_error(exception)
        priority = self.error_classifier.get_priority(error_type, exception)
        
        # Record error for monitoring
        if self.error_monitor:
            self.error_monitor.record_error(error_type, priority, exception, context)
        
        # Attempt recovery
        recovery_result = None
        if self.error_recovery:
            recovery_result = await self.error_recovery.attempt_recovery(
                error_type, exception, context
            )
        
        # Log error with full context
        await self._log_error(exception, context, error_type, priority, duration, recovery_result)
        
        # Return error response
        return await self._create_error_response(exception, context, error_type, recovery_result)
    
    async def _log_error(
        self,
        exception: Exception,
        context: ErrorContext,
        error_type: ErrorType,
        priority: ErrorPriority,
        duration: float,
        recovery_result: Optional[Dict[str, Any]]
    ):
        """Log error with full context."""
        log_data = {
            "request_id": context.request_id,
            "error_type": error_type.value,
            "priority": priority.value,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "endpoint": context.endpoint,
            "method": context.method,
            "duration_ms": duration * 1000,
            "user_agent": context.user_agent,
            "ip_address": context.ip_address,
            "traceback": traceback.format_exc(),
            "recovery_attempted": recovery_result is not None,
            "recovery_result": recovery_result
        }
        
        # Add request details if available
        if context.request_body:
            log_data["request_body"] = context.request_body
        if context.headers:
            log_data["headers"] = context.headers
        
        # Log based on priority
        if priority == ErrorPriority.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif priority == ErrorPriority.HIGH:
            logger.error("High priority error occurred", **log_data)
        elif priority == ErrorPriority.MEDIUM:
            logger.warning("Medium priority error occurred", **log_data)
        else:
            logger.info("Low priority error occurred", **log_data)
    
    async def _create_error_response(
        self,
        exception: Exception,
        context: ErrorContext,
        error_type: ErrorType,
        recovery_result: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Create error response."""
        # Create internal server error
        internal_error = InternalServerError(
            message="An unexpected error occurred",
            request_id=context.request_id
        )
        
        # Add recovery information if available
        if recovery_result:
            internal_error.message = f"Service temporarily degraded: {recovery_result.get('message', 'Using fallback data')}"
        
        # Create error response
        error_response = internal_error.to_error_response()
        
        # Add additional context for debugging
        error_response.details = [
            {
                "field": "error_type",
                "message": f"Error classified as: {error_type.value}",
                "value": error_type.value
            },
            {
                "field": "recovery",
                "message": "Recovery attempted" if recovery_result else "No recovery available",
                "value": recovery_result
            }
        ]
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict(),
            headers={"X-Request-ID": context.request_id}
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# =============================================================================
# Error Monitoring Endpoints
# =============================================================================

class ErrorMonitoringEndpoints:
    """Endpoints for error monitoring and management."""
    
    def __init__(self, error_monitor: ErrorMonitor):
        
    """__init__ function."""
self.error_monitor = error_monitor
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_monitor.get_error_stats()
    
    def clear_error_history(self) -> Any:
        """Clear error history."""
        self.error_monitor.clear_history()
        return {"message": "Error history cleared"}
    
    def get_error_alerts(self) -> List[Dict[str, Any]]:
        """Get current error alerts."""
        alerts = []
        for error_key, count in self.error_monitor.error_counts.items():
            if count > 0:
                error_type, priority = error_key.split("_", 1)
                alerts.append({
                    "error_type": error_type,
                    "priority": priority,
                    "count": count,
                    "threshold": self.error_monitor.alert_thresholds.get(ErrorPriority(priority), 10)
                })
        return alerts

# =============================================================================
# Usage Examples
# =============================================================================

def create_error_handling_middleware(
    enable_recovery: bool = True,
    enable_monitoring: bool = True,
    log_request_body: bool = False,
    log_headers: bool = False
) -> ErrorHandlingMiddleware:
    """Create error handling middleware with configuration."""
    return ErrorHandlingMiddleware(
        app=None,  # Will be set by FastAPI
        enable_recovery=enable_recovery,
        enable_monitoring=enable_monitoring,
        log_request_body=log_request_body,
        log_headers=log_headers
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ErrorType",
    "ErrorPriority",
    "ErrorContext",
    "ErrorClassifier",
    "ErrorMonitor",
    "ErrorRecovery",
    "ErrorHandlingMiddleware",
    "ErrorMonitoringEndpoints",
    "create_error_handling_middleware",
] 