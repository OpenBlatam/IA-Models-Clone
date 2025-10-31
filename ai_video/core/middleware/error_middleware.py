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
import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import hashlib
import sys
import os
from datetime import datetime, timedelta
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import psutil
import gc
from .http_exceptions import (
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
from typing import Any, List, Dict, Optional
"""
🚀 ERROR MIDDLEWARE - UNEXPECTED ERRORS, LOGGING & MONITORING
============================================================

Comprehensive middleware system for handling unexpected errors:
- Global error handling middleware
- Structured logging middleware
- Error monitoring and alerting
- Performance tracking
- Request/response correlation
- Error recovery strategies
"""




    AIVideoHTTPException,
    SystemError,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
    HTTPExceptionHandler,
    ErrorMonitor
)

logger = logging.getLogger(__name__)

# ============================================================================
# 1. ERROR SEVERITY AND CATEGORIZATION
# ============================================================================

class ErrorType(Enum):
    """Types of errors that can occur."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_SERVICE = "external_service"
    MODEL = "model"
    MEMORY = "memory"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class ErrorAction(Enum):
    """Actions to take for different error types."""
    LOG = "log"
    ALERT = "alert"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    IGNORE = "ignore"

@dataclass
class ErrorInfo:
    """Information about an error."""
    error_type: ErrorType
    severity: ErrorSeverity
    action: ErrorAction
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    alert_threshold: int = 5
    circuit_break_threshold: int = 10

# ============================================================================
# 2. ERROR MONITORING AND ALERTING
# ============================================================================

class ErrorTracker:
    """Track errors for monitoring and alerting."""
    
    def __init__(self) -> Any:
        self.error_counts = {}
        self.error_timestamps = []
        self.circuit_breakers = {}
        self.alert_history = []
        self.max_history = 1000
    
    def record_error(self, error_type: ErrorType, error_info: ErrorInfo, context: Dict[str, Any]):
        """Record an error for monitoring."""
        error_key = f"{error_type.value}:{error_info.severity.value}"
        
        # Update counts
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Record timestamp
        timestamp = time.time()
        self.error_timestamps.append({
            "timestamp": timestamp,
            "error_type": error_type.value,
            "severity": error_info.severity.value,
            "context": context
        })
        
        # Keep only recent history
        if len(self.error_timestamps) > self.max_history:
            self.error_timestamps = self.error_timestamps[-self.max_history:]
        
        # Check circuit breaker
        self._check_circuit_breaker(error_type, error_info)
        
        # Check alert threshold
        self._check_alert_threshold(error_type, error_info)
    
    def _check_circuit_breaker(self, error_type: ErrorType, error_info: ErrorInfo):
        """Check if circuit breaker should be triggered."""
        if error_info.circuit_break_threshold <= 0:
            return
        
        error_key = error_type.value
        recent_errors = [
            e for e in self.error_timestamps[-100:]  # Last 100 errors
            if e["error_type"] == error_type.value
        ]
        
        if len(recent_errors) >= error_info.circuit_break_threshold:
            self.circuit_breakers[error_key] = {
                "triggered_at": time.time(),
                "error_count": len(recent_errors),
                "threshold": error_info.circuit_break_threshold
            }
            
            logger.critical(f"Circuit breaker triggered for {error_type.value}")
    
    def _check_alert_threshold(self, error_type: ErrorType, error_info: ErrorInfo):
        """Check if alert should be sent."""
        if error_info.alert_threshold <= 0:
            return
        
        recent_errors = [
            e for e in self.error_timestamps[-50:]  # Last 50 errors
            if e["error_type"] == error_type.value
        ]
        
        if len(recent_errors) >= error_info.alert_threshold:
            alert = {
                "timestamp": time.time(),
                "error_type": error_type.value,
                "error_count": len(recent_errors),
                "threshold": error_info.alert_threshold,
                "severity": error_info.severity.value
            }
            
            self.alert_history.append(alert)
            
            # Keep only recent alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            logger.warning(f"Alert threshold reached for {error_type.value}: {len(recent_errors)} errors")
    
    def is_circuit_broken(self, error_type: ErrorType) -> bool:
        """Check if circuit breaker is active for error type."""
        return error_type.value in self.circuit_breakers
    
    def get_error_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get error statistics for the last N minutes."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_errors = [
            e for e in self.error_timestamps
            if e["timestamp"] > cutoff_time
        ]
        
        return {
            "total_errors": len(recent_errors),
            "errors_by_type": self._group_errors_by_type(recent_errors),
            "errors_by_severity": self._group_errors_by_severity(recent_errors),
            "error_rate": len(recent_errors) / window_minutes,
            "circuit_breakers": self.circuit_breakers,
            "recent_alerts": self.alert_history[-10:] if self.alert_history else []
        }
    
    def _group_errors_by_type(self, errors: List[Dict]) -> Dict[str, int]:
        """Group errors by type."""
        grouped = {}
        for error in errors:
            error_type = error["error_type"]
            grouped[error_type] = grouped.get(error_type, 0) + 1
        return grouped
    
    def _group_errors_by_severity(self, errors: List[Dict]) -> Dict[str, int]:
        """Group errors by severity."""
        grouped = {}
        for error in errors:
            severity = error["severity"]
            grouped[severity] = grouped.get(severity, 0) + 1
        return grouped

# ============================================================================
# 3. STRUCTURED LOGGING MIDDLEWARE
# ============================================================================

@dataclass
class RequestLog:
    """Log entry for a request."""
    request_id: str
    method: str
    url: str
    client_ip: str
    user_agent: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    user_id: Optional[str] = None
    video_id: Optional[str] = None
    model_name: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    memory_usage: Optional[float] = None

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging of requests and responses."""
    
    def __init__(self, app: ASGIApp, log_level: str = "INFO"):
        
    """__init__ function."""
super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        self.logger = logging.getLogger("request_logger")
        self.logger.setLevel(self.log_level)
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log structured information."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Extract request information
        log_entry = RequestLog(
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            start_time=start_time,
            user_id=request.headers.get("x-user-id"),
            video_id=self._extract_video_id(request),
            model_name=request.query_params.get("model_name"),
            request_size=self._get_request_size(request)
        )
        
        # Log request start
        self._log_request_start(log_entry)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update log entry
            end_time = time.time()
            log_entry.end_time = end_time
            log_entry.duration = end_time - start_time
            log_entry.status_code = response.status_code
            log_entry.response_size = self._get_response_size(response)
            log_entry.memory_usage = self._get_memory_usage()
            
            # Log request completion
            self._log_request_complete(log_entry)
            
            return response
            
        except Exception as exc:
            # Update log entry with error
            end_time = time.time()
            log_entry.end_time = end_time
            log_entry.duration = end_time - start_time
            log_entry.error = str(exc)
            log_entry.error_type = exc.__class__.__name__
            log_entry.memory_usage = self._get_memory_usage()
            
            # Log request error
            self._log_request_error(log_entry)
            
            # Re-raise exception
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _extract_video_id(self, request: Request) -> Optional[str]:
        """Extract video ID from request path or query parameters."""
        # Check path parameters
        if "video_id" in request.path_params:
            return request.path_params["video_id"]
        
        # Check query parameters
        return request.query_params.get("video_id")
    
    async def _get_request_size(self, request: Request) -> Optional[int]:
        """Get request size in bytes."""
        try:
            content_length = request.headers.get("content-length")
            return int(content_length) if content_length else None
        except (ValueError, TypeError):
            return None
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response size in bytes."""
        try:
            content_length = response.headers.get("content-length")
            return int(content_length) if content_length else None
        except (ValueError, TypeError):
            return None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _log_request_start(self, log_entry: RequestLog):
        """Log request start."""
        log_data = {
            "event": "request_start",
            "request_id": log_entry.request_id,
            "method": log_entry.method,
            "url": log_entry.url,
            "client_ip": log_entry.client_ip,
            "user_agent": log_entry.user_agent,
            "timestamp": datetime.fromtimestamp(log_entry.start_time).isoformat(),
            "user_id": log_entry.user_id,
            "video_id": log_entry.video_id,
            "model_name": log_entry.model_name,
            "request_size": log_entry.request_size
        }
        
        self.logger.info(json.dumps(log_data))
    
    def _log_request_complete(self, log_entry: RequestLog):
        """Log request completion."""
        log_data = {
            "event": "request_complete",
            "request_id": log_entry.request_id,
            "method": log_entry.method,
            "url": log_entry.url,
            "duration": log_entry.duration,
            "status_code": log_entry.status_code,
            "timestamp": datetime.fromtimestamp(log_entry.end_time).isoformat(),
            "response_size": log_entry.response_size,
            "memory_usage": log_entry.memory_usage
        }
        
        # Log based on status code
        if log_entry.status_code >= 400:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def _log_request_error(self, log_entry: RequestLog):
        """Log request error."""
        log_data = {
            "event": "request_error",
            "request_id": log_entry.request_id,
            "method": log_entry.method,
            "url": log_entry.url,
            "duration": log_entry.duration,
            "error": log_entry.error,
            "error_type": log_entry.error_type,
            "timestamp": datetime.fromtimestamp(log_entry.end_time).isoformat(),
            "memory_usage": log_entry.memory_usage
        }
        
        self.logger.error(json.dumps(log_data))

# ============================================================================
# 4. ERROR HANDLING MIDDLEWARE
# ============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling unexpected errors."""
    
    def __init__(self, app: ASGIApp, error_tracker: ErrorTracker):
        
    """__init__ function."""
super().__init__(app)
        self.error_tracker = error_tracker
        self.error_handler = HTTPExceptionHandler()
        self.error_monitor = ErrorMonitor()
        
        # Error type mapping
        self.error_type_mapping = {
            RequestValidationError: ErrorType.VALIDATION,
            StarletteHTTPException: ErrorType.SYSTEM,
            AIVideoHTTPException: ErrorType.SYSTEM,
            TimeoutError: ErrorType.TIMEOUT,
            MemoryError: ErrorType.MEMORY,
            ConnectionError: ErrorType.EXTERNAL_SERVICE,
            OSError: ErrorType.SYSTEM,
            ValueError: ErrorType.VALIDATION,
            TypeError: ErrorType.VALIDATION,
            KeyError: ErrorType.VALIDATION,
            IndexError: ErrorType.VALIDATION,
            AttributeError: ErrorType.SYSTEM,
            ImportError: ErrorType.SYSTEM,
            ModuleNotFoundError: ErrorType.SYSTEM,
            PermissionError: ErrorType.AUTHORIZATION,
            FileNotFoundError: ErrorType.NOT_FOUND,
            IsADirectoryError: ErrorType.VALIDATION,
            NotADirectoryError: ErrorType.VALIDATION,
            FileExistsError: ErrorType.VALIDATION,
            BrokenPipeError: ErrorType.EXTERNAL_SERVICE,
            ConnectionResetError: ErrorType.EXTERNAL_SERVICE,
            ConnectionRefusedError: ErrorType.EXTERNAL_SERVICE,
            ConnectionAbortedError: ErrorType.EXTERNAL_SERVICE,
            TimeoutError: ErrorType.TIMEOUT,
            BlockingIOError: ErrorType.SYSTEM,
            ChildProcessError: ErrorType.SYSTEM,
            ProcessLookupError: ErrorType.SYSTEM,
            InterruptedError: ErrorType.SYSTEM,
            RecursionError: ErrorType.MEMORY,
            NotImplementedError: ErrorType.SYSTEM,
            RuntimeError: ErrorType.SYSTEM,
            SystemError: ErrorType.SYSTEM,
            ReferenceError: ErrorType.SYSTEM,
            SyntaxError: ErrorType.SYSTEM,
            IndentationError: ErrorType.SYSTEM,
            TabError: ErrorType.SYSTEM,
            UnicodeError: ErrorType.VALIDATION,
            UnicodeDecodeError: ErrorType.VALIDATION,
            UnicodeEncodeError: ErrorType.VALIDATION,
            UnicodeTranslateError: ErrorType.VALIDATION,
            Warning: ErrorType.SYSTEM,
            UserWarning: ErrorType.SYSTEM,
            DeprecationWarning: ErrorType.SYSTEM,
            PendingDeprecationWarning: ErrorType.SYSTEM,
            SyntaxWarning: ErrorType.SYSTEM,
            RuntimeWarning: ErrorType.SYSTEM,
            FutureWarning: ErrorType.SYSTEM,
            ImportWarning: ErrorType.SYSTEM,
            UnicodeWarning: ErrorType.SYSTEM,
            BytesWarning: ErrorType.SYSTEM,
            ResourceWarning: ErrorType.SYSTEM
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Determine error type
            error_type = self._determine_error_type(exc)
            error_info = self._get_error_info(error_type, exc)
            
            # Create error context
            context = self._create_error_context(request, exc)
            
            # Record error for monitoring
            self.error_tracker.record_error(error_type, error_info, context)
            self.error_monitor.record_error(self._convert_to_ai_video_exception(exc, context))
            
            # Handle error based on type
            return await self._handle_error(exc, error_type, error_info, context)
    
    def _determine_error_type(self, exc: Exception) -> ErrorType:
        """Determine the type of error."""
        exc_type = type(exc)
        
        # Check direct type mapping
        if exc_type in self.error_type_mapping:
            return self.error_type_mapping[exc_type]
        
        # Check for specific error patterns
        error_message = str(exc).lower()
        
        if any(word in error_message for word in ["timeout", "timed out"]):
            return ErrorType.TIMEOUT
        elif any(word in error_message for word in ["memory", "out of memory"]):
            return ErrorType.MEMORY
        elif any(word in error_message for word in ["database", "sql", "connection"]):
            return ErrorType.DATABASE
        elif any(word in error_message for word in ["cache", "redis"]):
            return ErrorType.CACHE
        elif any(word in error_message for word in ["model", "inference", "gpu"]):
            return ErrorType.MODEL
        elif any(word in error_message for word in ["not found", "missing", "404"]):
            return ErrorType.NOT_FOUND
        elif any(word in error_message for word in ["permission", "access denied", "unauthorized"]):
            return ErrorType.AUTHORIZATION
        elif any(word in error_message for word in ["rate limit", "too many requests"]):
            return ErrorType.RATE_LIMIT
        elif any(word in error_message for word in ["validation", "invalid", "bad request"]):
            return ErrorType.VALIDATION
        
        return ErrorType.UNKNOWN
    
    def _get_error_info(self, error_type: ErrorType, exc: Exception) -> ErrorInfo:
        """Get error information based on error type."""
        if error_type == ErrorType.TIMEOUT:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                action=ErrorAction.RETRY,
                max_retries=3,
                retry_delay=2.0,
                alert_threshold=10
            )
        elif error_type == ErrorType.MEMORY:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3,
                circuit_break_threshold=5
            )
        elif error_type == ErrorType.DATABASE:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.RETRY,
                max_retries=5,
                retry_delay=1.0,
                alert_threshold=5,
                circuit_break_threshold=10
            )
        elif error_type == ErrorType.CACHE:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                action=ErrorAction.FALLBACK,
                alert_threshold=10
            )
        elif error_type == ErrorType.MODEL:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3,
                circuit_break_threshold=5
            )
        elif error_type == ErrorType.EXTERNAL_SERVICE:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.RETRY,
                max_retries=3,
                retry_delay=5.0,
                alert_threshold=5,
                circuit_break_threshold=10
            )
        else:
            return ErrorInfo(
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                action=ErrorAction.LOG,
                alert_threshold=20
            )
    
    def _create_error_context(self, request: Request, exc: Exception) -> Dict[str, Any]:
        """Create error context from request and exception."""
        return {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "url": str(request.url),
            "client_ip": self._get_client_ip(request),
            "user_id": request.headers.get("x-user-id"),
            "video_id": self._extract_video_id(request),
            "model_name": request.query_params.get("model_name"),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "timestamp": time.time()
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _extract_video_id(self, request: Request) -> Optional[str]:
        """Extract video ID from request."""
        if "video_id" in request.path_params:
            return request.path_params["video_id"]
        return request.query_params.get("video_id")
    
    def _convert_to_ai_video_exception(self, exc: Exception, context: Dict[str, Any]) -> AIVideoHTTPException:
        """Convert exception to AI Video exception."""
        error_context = ErrorContext(
            user_id=context.get("user_id"),
            request_id=context.get("request_id"),
            video_id=context.get("video_id"),
            model_name=context.get("model_name"),
            operation=context.get("method"),
            additional_data={
                "original_error": str(exc),
                "error_type": exc.__class__.__name__
            }
        )
        
        return SystemError(
            detail=f"Unexpected error: {str(exc)}",
            context=error_context
        )
    
    async def _handle_error(self, exc: Exception, error_type: ErrorType, 
                          error_info: ErrorInfo, context: Dict[str, Any]) -> Response:
        """Handle error based on error type and configuration."""
        
        # Check circuit breaker
        if self.error_tracker.is_circuit_broken(error_type):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "type": "CircuitBreakerError",
                        "message": f"Service temporarily unavailable due to {error_type.value} errors",
                        "category": "system_error",
                        "severity": "high",
                        "status_code": 503,
                        "timestamp": time.time(),
                        "context": context
                    }
                }
            )
        
        # Handle based on action
        if error_info.action == ErrorAction.RETRY and error_info.retry_count < error_info.max_retries:
            # Retry logic would be implemented here
            # For now, just log and continue
            logger.warning(f"Retry attempt {error_info.retry_count + 1} for {error_type.value}")
        
        elif error_info.action == ErrorAction.FALLBACK:
            # Fallback logic would be implemented here
            logger.info(f"Using fallback for {error_type.value}")
        
        # Convert to appropriate HTTP response
        if isinstance(exc, AIVideoHTTPException):
            return self.error_handler.handle_exception(exc)
        elif isinstance(exc, RequestValidationError):
            return self.error_handler.handle_exception(exc)
        elif isinstance(exc, StarletteHTTPException):
            return self.error_handler.handle_exception(exc)
        else:
            # Convert to system error
            system_error = self._convert_to_ai_video_exception(exc, context)
            return self.error_handler.handle_exception(system_error)

# ============================================================================
# 5. PERFORMANCE MONITORING MIDDLEWARE
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    request_id: str
    method: str
    url: str
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_usage: float
    status_code: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
        self.metrics = []
        self.max_metrics = 1000
        self.slow_request_threshold = 5.0  # seconds
        self.high_memory_threshold = 1024  # MB
    
    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Get initial metrics
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            # Create metrics
            metrics = PerformanceMetrics(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                cpu_usage=cpu_before,
                status_code=response.status_code
            )
            
            # Record metrics
            self._record_metrics(metrics)
            
            # Check for performance issues
            self._check_performance_issues(metrics)
            
            return response
            
        except Exception as exc:
            # Calculate metrics for error case
            end_time = time.time()
            duration = end_time - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            metrics = PerformanceMetrics(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                cpu_usage=cpu_before,
                status_code=500,
                error=str(exc)
            )
            
            self._record_metrics(metrics)
            self._check_performance_issues(metrics)
            
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def _check_performance_issues(self, metrics: PerformanceMetrics):
        """Check for performance issues and log warnings."""
        if metrics.duration > self.slow_request_threshold:
            logger.warning(f"Slow request detected: {metrics.duration:.2f}s for {metrics.url}")
        
        if metrics.memory_delta > 100:  # 100MB increase
            logger.warning(f"High memory usage detected: {metrics.memory_delta:.2f}MB increase")
        
        if metrics.memory_after > self.high_memory_threshold:
            logger.warning(f"High memory usage: {metrics.memory_after:.2f}MB")
    
    def get_performance_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance statistics for the last N minutes."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics available"}
        
        durations = [m.duration for m in recent_metrics]
        memory_deltas = [m.memory_delta for m in recent_metrics]
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        
        return {
            "total_requests": len(recent_metrics),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "slow_requests": len([d for d in durations if d > self.slow_request_threshold]),
            "high_memory_requests": len([m for m in memory_deltas if m > 100])
        }

# ============================================================================
# 6. MIDDLEWARE STACK
# ============================================================================

class MiddlewareStack:
    """Stack of middleware for the application."""
    
    def __init__(self) -> Any:
        self.error_tracker = ErrorTracker()
        self.middleware_stack = []
    
    def add_middleware(self, middleware_class: type, **kwargs):
        """Add middleware to the stack."""
        self.middleware_stack.append((middleware_class, kwargs))
    
    def create_middleware_stack(self, app: ASGIApp) -> ASGIApp:
        """Create the middleware stack."""
        # Add default middleware
        if not any(m[0] == StructuredLoggingMiddleware for m in self.middleware_stack):
            self.add_middleware(StructuredLoggingMiddleware, log_level="INFO")
        
        if not any(m[0] == ErrorHandlingMiddleware for m in self.middleware_stack):
            self.add_middleware(ErrorHandlingMiddleware, error_tracker=self.error_tracker)
        
        if not any(m[0] == PerformanceMonitoringMiddleware for m in self.middleware_stack):
            self.add_middleware(PerformanceMonitoringMiddleware)
        
        # Apply middleware in reverse order (last added is innermost)
        for middleware_class, kwargs in reversed(self.middleware_stack):
            app = middleware_class(app, **kwargs)
        
        return app
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_tracker.get_error_stats()

# ============================================================================
# 7. USAGE EXAMPLES
# ============================================================================

def create_app_with_middleware() -> ASGIApp:
    """Create FastAPI app with comprehensive middleware."""
    
    app = FastAPI(title="AI Video API with Middleware")
    
    # Create middleware stack
    middleware_stack = MiddlewareStack()
    
    # Add custom middleware if needed
    # middleware_stack.add_middleware(CustomMiddleware, custom_param="value")
    
    # Apply middleware stack
    app = middleware_stack.create_middleware_stack(app)
    
    # Add routes
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/errors/stats")
    async def get_error_stats():
        
    """get_error_stats function."""
return middleware_stack.get_error_stats()
    
    @app.get("/performance/stats")
    async def get_performance_stats():
        
    """get_performance_stats function."""
# This would need to be implemented to access the performance middleware
        return {"message": "Performance stats endpoint"}
    
    return app

async def example_middleware_usage():
    """Example of using the middleware system."""
    
    # Create app with middleware
    app = create_app_with_middleware()
    
    # Simulate some requests to see middleware in action
    
    client = TestClient(app)
    
    # Normal request
    response = client.get("/health")
    print(f"Health check: {response.status_code}")
    
    # Request that might cause errors
    response = client.get("/nonexistent")
    print(f"Error request: {response.status_code}")
    
    # Get error stats
    response = client.get("/errors/stats")
    print(f"Error stats: {response.json()}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_middleware_usage()) 