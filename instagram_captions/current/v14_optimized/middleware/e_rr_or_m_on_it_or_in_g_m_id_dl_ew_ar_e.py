from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import uuid
import logging
import asyncio
import traceback
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from ..types.exceptions import (
from ..utils.error_handling import ErrorTracker, ErrorType, ErrorSeverity
from typing import Any, List, Dict, Optional
"""
Error Monitoring Middleware for Instagram Captions API v14.0

Comprehensive middleware for:
- Unexpected error handling and recovery
- Structured error logging with context
- Error monitoring and alerting
- Performance impact tracking
- Error categorization and prioritization
"""



    create_error_response, AIGenerationError, CacheError, 
    ModelLoadingError, ServiceUnavailableError
)

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for monitoring and alerting"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    AI_MODEL = "ai_model"
    CACHE = "cache"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorPriority(Enum):
    """Error priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Structured error record for monitoring"""
    timestamp: datetime
    request_id: str
    error_type: str
    error_category: ErrorCategory
    priority: ErrorPriority
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str]
    endpoint: str
    method: str
    client_ip: str
    user_agent: str
    response_time: float
    status_code: int


class ErrorMonitor:
    """Advanced error monitoring and alerting system"""
    
    def __init__(self, max_errors: int = 1000, alert_threshold: int = 10):
        
    """__init__ function."""
self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.priority_counts: Dict[ErrorPriority, int] = defaultdict(int)
        self.endpoint_errors: Dict[str, int] = defaultdict(int)
        self.alert_threshold = alert_threshold
        self.alerts: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        # Performance tracking
        self.total_requests = 0
        self.error_requests = 0
        self.avg_response_time = 0.0
        
        # Alerting configuration
        self.alert_rules = {
            "error_rate_threshold": 0.05,  # 5% error rate
            "consecutive_errors_threshold": 5,
            "critical_error_threshold": 1,
            "response_time_threshold": 5.0  # 5 seconds
        }
    
    def record_error(self, error_record: ErrorRecord):
        """Record an error with full context"""
        # Add to error history
        self.errors.append(error_record)
        
        # Update counters
        self.error_counts[error_record.error_type] += 1
        self.category_counts[error_record.error_category] += 1
        self.priority_counts[error_record.priority] += 1
        self.endpoint_errors[error_record.endpoint] += 1
        
        # Update performance metrics
        self.error_requests += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + error_record.response_time) / 
            self.total_requests
        )
        
        # Check for alerts
        self._check_alerts(error_record)
        
        # Log based on priority
        self._log_error(error_record)
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request for performance tracking"""
        self.total_requests += 1
        if is_error:
            self.error_requests += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / 
            self.total_requests
        )
    
    def _check_alerts(self, error_record: ErrorRecord):
        """Check if error should trigger alerts"""
        current_time = time.time()
        
        # Check error rate
        error_rate = self.error_requests / max(self.total_requests, 1)
        if error_rate > self.alert_rules["error_rate_threshold"]:
            self._create_alert(
                "HIGH_ERROR_RATE",
                f"Error rate is {error_rate:.2%} (threshold: {self.alert_rules['error_rate_threshold']:.2%})",
                error_record
            )
        
        # Check consecutive errors
        recent_errors = [e for e in self.errors if 
                        (current_time - e.timestamp.timestamp()) < 60]  # Last minute
        if len(recent_errors) >= self.alert_rules["consecutive_errors_threshold"]:
            self._create_alert(
                "CONSECUTIVE_ERRORS",
                f"{len(recent_errors)} consecutive errors in the last minute",
                error_record
            )
        
        # Check critical errors
        if error_record.priority == ErrorPriority.CRITICAL:
            self._create_alert(
                "CRITICAL_ERROR",
                f"Critical error: {error_record.message}",
                error_record
            )
        
        # Check response time
        if error_record.response_time > self.alert_rules["response_time_threshold"]:
            self._create_alert(
                "SLOW_RESPONSE",
                f"Slow response time: {error_record.response_time:.2f}s",
                error_record
            )
    
    def _create_alert(self, alert_type: str, message: str, error_record: ErrorRecord):
        """Create and log an alert"""
        alert = {
            "timestamp": datetime.now(timezone.utc),
            "alert_type": alert_type,
            "message": message,
            "error_record": asdict(error_record),
            "severity": "HIGH" if alert_type in ["CRITICAL_ERROR", "HIGH_ERROR_RATE"] else "MEDIUM"
        }
        
        self.alerts.append(alert)
        
        # Log alert
        logger.critical(f"🚨 ALERT [{alert_type}]: {message}")
        logger.critical(f"Error details: {error_record.message}")
        logger.critical(f"Endpoint: {error_record.endpoint}")
        logger.critical(f"Request ID: {error_record.request_id}")
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level based on priority"""
        log_data = {
            "request_id": error_record.request_id,
            "error_type": error_record.error_type,
            "category": error_record.error_category.value,
            "priority": error_record.priority.value,
            "message": error_record.message,
            "endpoint": error_record.endpoint,
            "method": error_record.method,
            "response_time": error_record.response_time,
            "status_code": error_record.status_code,
            "client_ip": error_record.client_ip,
            "user_agent": error_record.user_agent
        }
        
        if error_record.priority == ErrorPriority.CRITICAL:
            logger.critical("CRITICAL ERROR", extra=log_data, exc_info=True)
        elif error_record.priority == ErrorPriority.HIGH:
            logger.error("HIGH PRIORITY ERROR", extra=log_data, exc_info=True)
        elif error_record.priority == ErrorPriority.MEDIUM:
            logger.warning("MEDIUM PRIORITY ERROR", extra=log_data)
        else:
            logger.info("LOW PRIORITY ERROR", extra=log_data)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "total_requests": self.total_requests,
            "error_requests": self.error_requests,
            "error_rate": self.error_requests / max(self.total_requests, 1),
            "avg_response_time": self.avg_response_time,
            "uptime": time.time() - self.start_time,
            "error_counts": dict(self.error_counts),
            "category_counts": {cat.value: count for cat, count in self.category_counts.items()},
            "priority_counts": {pri.value: count for pri, count in self.priority_counts.items()},
            "endpoint_errors": dict(self.endpoint_errors),
            "recent_alerts": len([a for a in self.alerts if 
                                (time.time() - a["timestamp"].timestamp()) < 3600]),  # Last hour
            "total_alerts": len(self.alerts)
        }
    
    def get_recent_errors(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get errors from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_errors = [
            asdict(error) for error in self.errors 
            if error.timestamp.timestamp() > cutoff_time
        ]
        return recent_errors


class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Comprehensive error monitoring middleware"""
    
    def __init__(self, app, enable_detailed_logging: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.error_monitor = ErrorMonitor()
        self.enable_detailed_logging = enable_detailed_logging
        self.error_tracker = ErrorTracker()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not present
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Extract request context
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.perf_counter() - start_time
            
            # Record successful request
            self.error_monitor.record_request(response_time, False)
            
            # Add monitoring headers
            response.headers["X-Error-Rate"] = f"{self.error_monitor.error_requests / max(self.error_monitor.total_requests, 1):.3f}"
            response.headers["X-Avg-Response-Time"] = f"{self.error_monitor.avg_response_time:.3f}s"
            
            return response
            
        except HTTPException as e:
            # HTTP exceptions are handled by FastAPI
            response_time = time.perf_counter() - start_time
            self.error_monitor.record_request(response_time, True)
            
            # Record error for monitoring
            error_record = self._create_error_record(
                error_type=type(e).__name__,
                error_category=self._categorize_error(e),
                priority=self._determine_priority(e),
                message=str(e.detail),
                details={"status_code": e.status_code},
                stack_trace=None,
                endpoint=endpoint,
                method=request.method,
                client_ip=client_ip,
                user_agent=user_agent,
                response_time=response_time,
                status_code=e.status_code,
                request_id=request_id
            )
            
            self.error_monitor.record_error(error_record)
            raise
            
        except Exception as e:
            # Unexpected errors
            response_time = time.perf_counter() - start_time
            self.error_monitor.record_request(response_time, True)
            
            # Create comprehensive error record
            error_record = self._create_error_record(
                error_type=type(e).__name__,
                error_category=self._categorize_error(e),
                priority=self._determine_priority(e),
                message=str(e),
                details={"exception_type": type(e).__name__},
                stack_trace=traceback.format_exc() if self.enable_detailed_logging else None,
                endpoint=endpoint,
                method=request.method,
                client_ip=client_ip,
                user_agent=user_agent,
                response_time=response_time,
                status_code=500,
                request_id=request_id
            )
            
            self.error_monitor.record_error(error_record)
            
            # Return structured error response
            error_response = create_error_response(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                status_code=500,
                details={
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                    "endpoint": endpoint
                },
                request_id=request_id,
                path=str(request.url.path),
                method=request.method
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
                headers={"X-Request-ID": request_id}
            )
    
    def _create_error_record(self, **kwargs) -> ErrorRecord:
        """Create an error record with current timestamp"""
        return ErrorRecord(
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and content"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # HTTP exceptions
        if isinstance(error, HTTPException):
            if error.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif error.status_code == 403:
                return ErrorCategory.AUTHORIZATION
            elif error.status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif error.status_code == 400:
                return ErrorCategory.VALIDATION
            else:
                return ErrorCategory.SYSTEM
        
        # AI-related errors
        if isinstance(error, (AIGenerationError, ModelLoadingError)):
            return ErrorCategory.AI_MODEL
        
        # Cache errors
        if isinstance(error, CacheError):
            return ErrorCategory.CACHE
        
        # Network-related errors
        if any(keyword in error_message for keyword in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK
        
        # External service errors
        if any(keyword in error_message for keyword in ["api", "service", "external"]):
            return ErrorCategory.EXTERNAL_SERVICE
        
        # Database errors
        if any(keyword in error_message for keyword in ["database", "sql", "db"]):
            return ErrorCategory.DATABASE
        
        return ErrorCategory.UNKNOWN
    
    def _determine_priority(self, error: Exception) -> ErrorPriority:
        """Determine error priority based on type and impact"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if isinstance(error, (AIGenerationError, ModelLoadingError)):
            return ErrorPriority.CRITICAL
        
        # High priority errors
        if any(keyword in error_message for keyword in ["memory", "disk", "database", "connection"]):
            return ErrorPriority.HIGH
        
        # Medium priority errors
        if isinstance(error, HTTPException) and error.status_code >= 500:
            return ErrorPriority.HIGH
        elif isinstance(error, HTTPException) and error.status_code >= 400:
            return ErrorPriority.MEDIUM
        
        return ErrorPriority.LOW


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """Middleware for error recovery and graceful degradation"""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.recovery_strategies = {
            "ai_model": self._recover_ai_model,
            "cache": self._recover_cache,
            "database": self._recover_database
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Try to recover based on error type
            recovery_result = await self._attempt_recovery(e, request)
            if recovery_result:
                return recovery_result
            
            # If recovery fails, re-raise the error
            raise
    
    async def _attempt_recovery(self, error: Exception, request: Request) -> Optional[Response]:
        """Attempt to recover from error"""
        error_category = self._categorize_error(error)
        
        if error_category.value in self.recovery_strategies:
            try:
                return await self.recovery_strategies[error_category.value](error, request)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
        
        return None
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for recovery strategies"""
        if isinstance(error, (AIGenerationError, ModelLoadingError)):
            return ErrorCategory.AI_MODEL
        elif isinstance(error, CacheError):
            return ErrorCategory.CACHE
        else:
            return ErrorCategory.SYSTEM
    
    async def _recover_ai_model(self, error: Exception, request: Request) -> Optional[Response]:
        """Recover from AI model errors"""
        # Implement fallback to simpler model or cached responses
        logger.warning("Attempting AI model recovery...")
        return None
    
    async def _recover_cache(self, error: Exception, request: Request) -> Optional[Response]:
        """Recover from cache errors"""
        # Implement fallback to direct processing
        logger.warning("Attempting cache recovery...")
        return None
    
    async def _recover_database(self, error: Exception, request: Request) -> Optional[Response]:
        """Recover from database errors"""
        # Implement fallback to in-memory storage
        logger.warning("Attempting database recovery...")
        return None


# Global error monitor instance
error_monitor = ErrorMonitor()


def get_error_monitor() -> ErrorMonitor:
    """Get the global error monitor instance"""
    return error_monitor


async def log_error_with_context(
    error: Exception,
    request: Request,
    response_time: float,
    additional_context: Optional[Dict[str, Any]] = None
):
    """Utility function to log errors with full context"""
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    endpoint = f"{request.method} {request.url.path}"
    
    error_record = ErrorRecord(
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        error_type=type(error).__name__,
        error_category=ErrorCategory.UNKNOWN,
        priority=ErrorPriority.MEDIUM,
        message=str(error),
        details=additional_context or {},
        stack_trace=traceback.format_exc(),
        endpoint=endpoint,
        method=request.method,
        client_ip=client_ip,
        user_agent=user_agent,
        response_time=response_time,
        status_code=500
    )
    
    error_monitor.record_error(error_record) 