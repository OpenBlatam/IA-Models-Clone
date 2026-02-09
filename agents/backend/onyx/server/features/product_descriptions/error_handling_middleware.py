from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import traceback
import json
import uuid
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uvicorn
from http_exceptions import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Error Handling Middleware
Product Descriptions Feature - Comprehensive Error Handling, Logging, and Monitoring
"""



# Import HTTP exceptions
    ProductDescriptionsHTTPException,
    InternalServerHTTPException,
    create_error_response,
    log_error,
    ErrorCode,
    ErrorSeverity
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
request_start_time_var: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)
error_context_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar('error_context', default=None)

class ErrorType(str, Enum):
    """Types of errors that can occur"""
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT = "RATE_LIMIT"
    GIT_OPERATION = "GIT_OPERATION"
    MODEL_VERSION = "MODEL_VERSION"
    PERFORMANCE = "PERFORMANCE"
    DATABASE = "DATABASE"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    CONFIGURATION = "CONFIGURATION"
    UNEXPECTED = "UNEXPECTED"
    TIMEOUT = "TIMEOUT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_id: str
    error_type: ErrorType
    error_code: str
    message: str
    details: Optional[str]
    severity: ErrorSeverity
    timestamp: datetime
    request_id: Optional[str]
    path: Optional[str]
    method: Optional[str]
    status_code: int
    client_ip: Optional[str]
    user_agent: Optional[str]
    duration_ms: float
    stack_trace: Optional[str]
    context: Optional[Dict[str, Any]]
    correlation_id: Optional[str]
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ErrorStats:
    """Error statistics for monitoring"""
    total_errors: int
    errors_by_type: Dict[str, int]
    errors_by_severity: Dict[str, int]
    errors_by_status_code: Dict[int, int]
    errors_by_path: Dict[str, int]
    recent_errors: List[ErrorRecord]
    error_rate: float  # errors per minute
    avg_response_time: float
    uptime: float

class ErrorMonitor:
    """Error monitoring and alerting system"""
    
    def __init__(self, max_errors: int = 1000, alert_threshold: int = 10):
        
    """__init__ function."""
self.max_errors = max_errors
        self.alert_threshold = alert_threshold
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)
        self.status_code_counts: Dict[int, int] = defaultdict(int)
        self.path_counts: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.slow_requests: deque = deque(maxlen=100)
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        
    def record_error(self, error_record: ErrorRecord) -> None:
        """Record an error for monitoring"""
        self.errors.append(error_record)
        
        # Update counters
        self.error_counts[error_record.error_type.value] += 1
        self.severity_counts[error_record.severity.value] += 1
        self.status_code_counts[error_record.status_code] += 1
        self.path_counts[error_record.path or "unknown"] += 1
        
        # Check for alerts
        self._check_alerts(error_record)
        
        # Update circuit breaker
        self._update_circuit_breaker(error_record)
    
    def record_response_time(self, duration_ms: float) -> None:
        """Record response time for performance monitoring"""
        self.response_times.append(duration_ms)
        
        # Track slow requests
        if duration_ms > 1000:  # 1 second threshold
            self.slow_requests.append({
                "duration_ms": duration_ms,
                "timestamp": datetime.now()
            })
    
    def _check_alerts(self, error_record: ErrorRecord) -> None:
        """Check if alerts should be triggered"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        # Check error rate
        recent_errors = [e for e in self.errors if 
                        current_time - e.timestamp.timestamp() < 60]
        
        if len(recent_errors) >= self.alert_threshold:
            self._trigger_alert("HIGH_ERROR_RATE", {
                "error_count": len(recent_errors),
                "threshold": self.alert_threshold,
                "time_window": "1 minute"
            })
            self.last_alert_time = current_time
        
        # Check critical errors
        if error_record.severity == ErrorSeverity.CRITICAL:
            self._trigger_alert("CRITICAL_ERROR", {
                "error_id": error_record.error_id,
                "message": error_record.message,
                "path": error_record.path,
                "status_code": error_record.status_code
            })
            self.last_alert_time = current_time
        
        # Check for repeated errors
        recent_similar_errors = [e for e in recent_errors 
                               if e.error_type == error_record.error_type 
                               and e.path == error_record.path]
        
        if len(recent_similar_errors) >= 3:
            self._trigger_alert("REPEATED_ERRORS", {
                "error_type": error_record.error_type.value,
                "path": error_record.path,
                "count": len(recent_similar_errors),
                "time_window": "1 minute"
            })
            self.last_alert_time = current_time
    
    def _update_circuit_breaker(self, error_record: ErrorRecord) -> None:
        """Update circuit breaker state"""
        if error_record.status_code >= 500:
            self.circuit_breaker_failures += 1
            
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self._trigger_alert("CIRCUIT_BREAKER_OPEN", {
                    "failure_count": self.circuit_breaker_failures,
                    "threshold": self.circuit_breaker_threshold
                })
        else:
            # Reset circuit breaker on successful requests
            self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
            
            if self.circuit_breaker_failures == 0 and self.circuit_breaker_open:
                self.circuit_breaker_open = False
                self._trigger_alert("CIRCUIT_BREAKER_CLOSED", {
                    "failure_count": self.circuit_breaker_failures
                })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger an alert (placeholder for integration with alerting systems)"""
        alert = {
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "monitor_id": "error_monitor"
        }
        
        logger.warning(f"ALERT: {alert_type} - {json.dumps(alert, indent=2)}")
        
        # Here you would integrate with your alerting system
        # e.g., Slack, PagerDuty, email, etc.
    
    def get_stats(self) -> ErrorStats:
        """Get current error statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate error rate (errors per minute)
        recent_errors = [e for e in self.errors if 
                        current_time - e.timestamp.timestamp() < 60]
        error_rate = len(recent_errors)
        
        # Calculate average response time
        avg_response_time = 0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return ErrorStats(
            total_errors=len(self.errors),
            errors_by_type=dict(self.error_counts),
            errors_by_severity=dict(self.severity_counts),
            errors_by_status_code=dict(self.status_code_counts),
            errors_by_path=dict(self.path_counts),
            recent_errors=list(self.errors)[-10:],  # Last 10 errors
            error_rate=error_rate,
            avg_response_time=avg_response_time,
            uptime=uptime
        )
    
    def clear_old_errors(self, max_age_hours: int = 24) -> int:
        """Clear old error records"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        initial_count = len(self.errors)
        
        # Remove old errors
        self.errors = deque(
            [e for e in self.errors if e.timestamp > cutoff_time],
            maxlen=self.max_errors
        )
        
        cleared_count = initial_count - len(self.errors)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old error records")
        
        return cleared_count

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Comprehensive error handling middleware"""
    
    def __init__(
        self,
        app: ASGIApp,
        monitor: Optional[ErrorMonitor] = None,
        enable_logging: bool = True,
        enable_monitoring: bool = True,
        log_slow_requests: bool = True,
        slow_request_threshold_ms: int = 1000,
        include_traceback: bool = False,
        max_error_details_length: int = 1000
    ):
        
    """__init__ function."""
super().__init__(app)
        self.monitor = monitor or ErrorMonitor()
        self.enable_logging = enable_logging
        self.enable_monitoring = enable_monitoring
        self.log_slow_requests = log_slow_requests
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.include_traceback = include_traceback
        self.max_error_details_length = max_error_details_length
        
        # Error handlers for different exception types
        self.error_handlers: Dict[type, Callable] = {
            ValueError: self._handle_validation_error,
            KeyError: self._handle_key_error,
            TypeError: self._handle_type_error,
            AttributeError: self._handle_attribute_error,
            ImportError: self._handle_import_error,
            OSError: self._handle_os_error,
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            MemoryError: self._handle_memory_error,
            RecursionError: self._handle_recursion_error,
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and handle errors"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Record start time
        start_time = time.time()
        request_start_time_var.set(start_time)
        
        # Set error context
        error_context = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "correlation_id": request.headers.get("X-Correlation-ID")
        }
        error_context_var.set(error_context)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = (time.time() - start_time) * 1000
            
            # Record response time
            if self.enable_monitoring:
                self.monitor.record_response_time(duration)
            
            # Log slow requests
            if self.log_slow_requests and duration > self.slow_request_threshold_ms:
                self._log_slow_request(request, duration, response.status_code)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.2f}ms"
            
            return response
            
        except ProductDescriptionsHTTPException as e:
            # Handle custom HTTP exceptions
            return await self._handle_custom_exception(request, e, start_time)
            
        except Exception as e:
            # Handle unexpected errors
            return await self._handle_unexpected_error(request, e, start_time)
    
    async def _handle_custom_exception(
        self, 
        request: Request, 
        exc: ProductDescriptionsHTTPException, 
        start_time: float
    ) -> JSONResponse:
        """Handle custom HTTP exceptions"""
        duration = (time.time() - start_time) * 1000
        error_context = error_context_var.get() or {}
        
        # Create error record
        error_record = ErrorRecord(
            error_id=str(uuid.uuid4()),
            error_type=self._map_error_type(exc.error_code),
            error_code=exc.error_code.value,
            message=exc.detail.get("message", str(exc)),
            details=exc.detail.get("details"),
            severity=exc.severity,
            timestamp=datetime.now(),
            request_id=error_context.get("request_id"),
            path=error_context.get("path"),
            method=error_context.get("method"),
            status_code=exc.status_code,
            client_ip=error_context.get("client_ip"),
            user_agent=error_context.get("user_agent"),
            duration_ms=duration,
            stack_trace=traceback.format_exc() if self.include_traceback else None,
            context=exc.context.model_dump() if exc.context else None,
            correlation_id=error_context.get("correlation_id")
        )
        
        # Log and monitor
        if self.enable_logging:
            self._log_error(error_record)
        
        if self.enable_monitoring:
            self.monitor.record_error(error_record)
        
        # Return response
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers
        )
    
    async def _handle_unexpected_error(
        self, 
        request: Request, 
        exc: Exception, 
        start_time: float
    ) -> JSONResponse:
        """Handle unexpected errors"""
        duration = (time.time() - start_time) * 1000
        error_context = error_context_var.get() or {}
        
        # Determine error type and details
        error_type, error_details = self._analyze_exception(exc)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=str(uuid.uuid4()),
            error_type=error_type,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR.value,
            message="An unexpected error occurred",
            details=self._truncate_error_details(str(exc)),
            severity=ErrorSeverity.CRITICAL,
            timestamp=datetime.now(),
            request_id=error_context.get("request_id"),
            path=error_context.get("path"),
            method=error_context.get("method"),
            status_code=500,
            client_ip=error_context.get("client_ip"),
            user_agent=error_context.get("user_agent"),
            duration_ms=duration,
            stack_trace=traceback.format_exc() if self.include_traceback else None,
            context={"original_exception": type(exc).__name__},
            correlation_id=error_context.get("correlation_id")
        )
        
        # Log and monitor
        if self.enable_logging:
            self._log_error(error_record)
        
        if self.enable_monitoring:
            self.monitor.record_error(error_record)
        
        # Create error response
        error_response = {
            "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
            "message": "An unexpected error occurred",
            "details": "Internal server error" if not self.include_traceback else str(exc),
            "severity": ErrorSeverity.CRITICAL.value,
            "timestamp": datetime.now().isoformat(),
            "request_id": error_context.get("request_id"),
            "path": error_context.get("path"),
            "method": error_context.get("method"),
            "correlation_id": error_context.get("correlation_id")
        }
        
        # Add traceback in development
        if self.include_traceback:
            error_response["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content=error_response,
            headers={
                "X-Request-ID": error_context.get("request_id", ""),
                "X-Error-ID": error_record.error_id
            }
        )
    
    def _analyze_exception(self, exc: Exception) -> tuple[ErrorType, str]:
        """Analyze exception to determine type and details"""
        # Check for specific error handlers
        for exc_type, handler in self.error_handlers.items():
            if isinstance(exc, exc_type):
                return handler(exc)
        
        # Default analysis
        exc_name = type(exc).__name__
        
        if "timeout" in str(exc).lower():
            return ErrorType.TIMEOUT, f"Timeout error: {exc_name}"
        elif "connection" in str(exc).lower():
            return ErrorType.EXTERNAL_SERVICE, f"Connection error: {exc_name}"
        elif "memory" in str(exc).lower():
            return ErrorType.PERFORMANCE, f"Memory error: {exc_name}"
        elif "database" in str(exc).lower() or "sql" in str(exc).lower():
            return ErrorType.DATABASE, f"Database error: {exc_name}"
        else:
            return ErrorType.UNEXPECTED, f"Unexpected error: {exc_name}"
    
    def _handle_validation_error(self, exc: ValueError) -> tuple[ErrorType, str]:
        return ErrorType.VALIDATION, f"Validation error: {str(exc)}"
    
    def _handle_key_error(self, exc: KeyError) -> tuple[ErrorType, str]:
        return ErrorType.VALIDATION, f"Missing key: {str(exc)}"
    
    def _handle_type_error(self, exc: TypeError) -> tuple[ErrorType, str]:
        return ErrorType.VALIDATION, f"Type error: {str(exc)}"
    
    def _handle_attribute_error(self, exc: AttributeError) -> tuple[ErrorType, str]:
        return ErrorType.VALIDATION, f"Attribute error: {str(exc)}"
    
    def _handle_import_error(self, exc: ImportError) -> tuple[ErrorType, str]:
        return ErrorType.CONFIGURATION, f"Import error: {str(exc)}"
    
    def _handle_os_error(self, exc: OSError) -> tuple[ErrorType, str]:
        return ErrorType.EXTERNAL_SERVICE, f"OS error: {str(exc)}"
    
    def _handle_connection_error(self, exc: ConnectionError) -> tuple[ErrorType, str]:
        return ErrorType.EXTERNAL_SERVICE, f"Connection error: {str(exc)}"
    
    def _handle_timeout_error(self, exc: TimeoutError) -> tuple[ErrorType, str]:
        return ErrorType.TIMEOUT, f"Timeout error: {str(exc)}"
    
    def _handle_memory_error(self, exc: MemoryError) -> tuple[ErrorType, str]:
        return ErrorType.PERFORMANCE, f"Memory error: {str(exc)}"
    
    def _handle_recursion_error(self, exc: RecursionError) -> tuple[ErrorType, str]:
        return ErrorType.PERFORMANCE, f"Recursion error: {str(exc)}"
    
    def _map_error_type(self, error_code: ErrorCode) -> ErrorType:
        """Map error code to error type"""
        mapping = {
            ErrorCode.VALIDATION_ERROR: ErrorType.VALIDATION,
            ErrorCode.INVALID_INPUT: ErrorType.VALIDATION,
            ErrorCode.MISSING_REQUIRED_FIELD: ErrorType.VALIDATION,
            ErrorCode.UNAUTHORIZED: ErrorType.AUTHENTICATION,
            ErrorCode.INVALID_CREDENTIALS: ErrorType.AUTHENTICATION,
            ErrorCode.FORBIDDEN: ErrorType.AUTHORIZATION,
            ErrorCode.RESOURCE_NOT_FOUND: ErrorType.NOT_FOUND,
            ErrorCode.RESOURCE_CONFLICT: ErrorType.CONFLICT,
            ErrorCode.RATE_LIMIT_EXCEEDED: ErrorType.RATE_LIMIT,
            ErrorCode.GIT_OPERATION_ERROR: ErrorType.GIT_OPERATION,
            ErrorCode.MODEL_VERSION_ERROR: ErrorType.MODEL_VERSION,
            ErrorCode.PERFORMANCE_ERROR: ErrorType.PERFORMANCE,
            ErrorCode.DATABASE_ERROR: ErrorType.DATABASE,
            ErrorCode.EXTERNAL_SERVICE_ERROR: ErrorType.EXTERNAL_SERVICE,
            ErrorCode.CONFIGURATION_ERROR: ErrorType.CONFIGURATION,
            ErrorCode.TIMEOUT_ERROR: ErrorType.TIMEOUT,
            ErrorCode.CIRCUIT_BREAKER_OPEN: ErrorType.CIRCUIT_BREAKER,
        }
        return mapping.get(error_code, ErrorType.UNEXPECTED)
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else None
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error with appropriate level"""
        log_data = {
            "error_id": error_record.error_id,
            "error_type": error_record.error_type.value,
            "error_code": error_record.error_code,
            "message": error_record.message,
            "severity": error_record.severity.value,
            "request_id": error_record.request_id,
            "path": error_record.path,
            "method": error_record.method,
            "status_code": error_record.status_code,
            "client_ip": error_record.client_ip,
            "duration_ms": error_record.duration_ms,
            "timestamp": error_record.timestamp.isoformat()
        }
        
        if error_record.stack_trace:
            log_data["stack_trace"] = error_record.stack_trace
        
        if error_record.context:
            log_data["context"] = error_record.context
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {json.dumps(log_data, indent=2)}")
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
    
    async def _log_slow_request(self, request: Request, duration: float, status_code: int) -> None:
        """Log slow requests"""
        log_data = {
            "duration_ms": duration,
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"SLOW REQUEST: {json.dumps(log_data, indent=2)}")
    
    def _truncate_error_details(self, details: str) -> str:
        """Truncate error details to prevent oversized responses"""
        if len(details) <= self.max_error_details_length:
            return details
        return details[:self.max_error_details_length] + "..."
    
    def get_stats(self) -> ErrorStats:
        """Get error statistics"""
        return self.monitor.get_stats()
    
    def clear_old_errors(self, max_age_hours: int = 24) -> int:
        """Clear old error records"""
        return self.monitor.clear_old_errors(max_age_hours)

# Utility functions
async def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_var.get()

async def get_request_duration() -> Optional[float]:
    """Get current request duration"""
    start_time = request_start_time_var.get()
    if start_time:
        return time.time() - start_time
    return None

def get_error_context() -> Optional[Dict[str, Any]]:
    """Get current error context"""
    return error_context_var.get()

def create_error_handling_middleware(
    app: ASGIApp,
    enable_logging: bool = True,
    enable_monitoring: bool = True,
    log_slow_requests: bool = True,
    slow_request_threshold_ms: int = 1000,
    include_traceback: bool = False,
    max_errors: int = 1000,
    alert_threshold: int = 10
) -> ErrorHandlingMiddleware:
    """Create error handling middleware with configuration"""
    monitor = ErrorMonitor(max_errors=max_errors, alert_threshold=alert_threshold)
    
    return ErrorHandlingMiddleware(
        app=app,
        monitor=monitor,
        enable_logging=enable_logging,
        enable_monitoring=enable_monitoring,
        log_slow_requests=log_slow_requests,
        slow_request_threshold_ms=slow_request_threshold_ms,
        include_traceback=include_traceback
    ) 