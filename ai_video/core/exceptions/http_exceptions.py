from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
import json
import time
from contextlib import contextmanager
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
    import asyncio
from typing import Any, List, Dict, Optional
"""
🚀 HTTP EXCEPTION SYSTEM - AI VIDEO SPECIFIC ERRORS
===================================================

Comprehensive HTTP exception system for AI Video applications:
- Specific HTTP status codes for different error types
- Detailed error messages and context
- Error categorization and handling
- FastAPI integration with proper error responses
- Error logging and monitoring
"""



logger = logging.getLogger(__name__)

# ============================================================================
# 1. ERROR CATEGORIES AND TYPES
# ============================================================================

class ErrorCategory(Enum):
    """Categories of errors in the AI Video system."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_CONFLICT = "resource_conflict"
    PROCESSING_ERROR = "processing_error"
    MODEL_ERROR = "model_error"
    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for errors."""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    video_id: Optional[str] = None
    model_name: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# 2. BASE HTTP EXCEPTION CLASSES
# ============================================================================

class AIVideoHTTPException(HTTPException):
    """Base HTTP exception for AI Video system."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        
    """__init__ function."""
super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": {
                "type": self.__class__.__name__,
                "message": self.detail,
                "category": self.category.value,
                "severity": self.severity.value,
                "status_code": self.status_code,
                "timestamp": self.timestamp,
                "context": {
                    "user_id": self.context.user_id,
                    "request_id": self.context.request_id,
                    "video_id": self.context.video_id,
                    "model_name": self.context.model_name,
                    "operation": self.context.operation,
                    "additional_data": self.context.additional_data
                }
            }
        }

# ============================================================================
# 3. SPECIFIC HTTP EXCEPTIONS
# ============================================================================

# Validation Errors (400)
class ValidationError(AIVideoHTTPException):
    """Validation error for request data."""
    
    def __init__(
        self,
        detail: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if field:
            detail = f"Validation error for field '{field}': {detail}"
            if value is not None:
                detail += f" (value: {value})"
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context
        )

class InvalidVideoRequestError(ValidationError):
    """Invalid video generation request."""
    
    def __init__(
        self,
        detail: str,
        video_id: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.video_id = video_id
        
        super().__init__(
            detail=f"Invalid video request: {detail}",
            context=context
        )

class InvalidModelRequestError(ValidationError):
    """Invalid model request."""
    
    def __init__(
        self,
        detail: str,
        model_name: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.model_name = model_name
        
        super().__init__(
            detail=f"Invalid model request: {detail}",
            context=context
        )

# Authentication Errors (401)
class AuthenticationError(AIVideoHTTPException):
    """Authentication error."""
    
    def __init__(
        self,
        detail: str = "Authentication required",
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class InvalidTokenError(AuthenticationError):
    """Invalid authentication token."""
    
    def __init__(self, context: Optional[ErrorContext] = None):
        
    """__init__ function."""
super().__init__(
            detail="Invalid or expired authentication token",
            context=context
        )

# Authorization Errors (403)
class AuthorizationError(AIVideoHTTPException):
    """Authorization error."""
    
    def __init__(
        self,
        detail: str = "Access denied",
        resource: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if resource:
            detail = f"Access denied to resource: {resource}"
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class InsufficientPermissionsError(AuthorizationError):
    """Insufficient permissions for operation."""
    
    def __init__(
        self,
        operation: str,
        required_permissions: List[str],
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.operation = operation
        
        super().__init__(
            detail=f"Insufficient permissions for operation '{operation}'. Required: {', '.join(required_permissions)}",
            context=context
        )

# Resource Not Found Errors (404)
class ResourceNotFoundError(AIVideoHTTPException):
    """Resource not found error."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_type} with id '{resource_id}' not found",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.LOW,
            context=context
        )

class VideoNotFoundError(ResourceNotFoundError):
    """Video not found error."""
    
    def __init__(
        self,
        video_id: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.video_id = video_id
        
        super().__init__(
            resource_type="Video",
            resource_id=video_id,
            context=context
        )

class ModelNotFoundError(ResourceNotFoundError):
    """Model not found error."""
    
    def __init__(
        self,
        model_name: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.model_name = model_name
        
        super().__init__(
            resource_type="Model",
            resource_id=model_name,
            context=context
        )

# Resource Conflict Errors (409)
class ResourceConflictError(AIVideoHTTPException):
    """Resource conflict error."""
    
    def __init__(
        self,
        detail: str,
        resource_type: str,
        resource_id: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            category=ErrorCategory.RESOURCE_CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

class VideoAlreadyExistsError(ResourceConflictError):
    """Video already exists error."""
    
    def __init__(
        self,
        video_id: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.video_id = video_id
        
        super().__init__(
            detail=f"Video with id '{video_id}' already exists",
            resource_type="Video",
            resource_id=video_id,
            context=context
        )

# Processing Errors (422)
class ProcessingError(AIVideoHTTPException):
    """Video processing error."""
    
    def __init__(
        self,
        detail: str,
        video_id: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.video_id = video_id
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            category=ErrorCategory.PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class VideoGenerationError(ProcessingError):
    """Video generation failed error."""
    
    def __init__(
        self,
        detail: str,
        video_id: Optional[str] = None,
        model_name: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.video_id = video_id
        context.model_name = model_name
        
        super().__init__(
            detail=f"Video generation failed: {detail}",
            video_id=video_id,
            context=context
        )

class VideoProcessingTimeoutError(ProcessingError):
    """Video processing timeout error."""
    
    def __init__(
        self,
        video_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
detail = "Video processing timed out"
        if timeout_seconds:
            detail += f" after {timeout_seconds} seconds"
        
        super().__init__(
            detail=detail,
            video_id=video_id,
            context=context
        )

# Model Errors (500)
class ModelError(AIVideoHTTPException):
    """AI model error."""
    
    def __init__(
        self,
        detail: str,
        model_name: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.model_name = model_name
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            category=ErrorCategory.MODEL_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class ModelLoadError(ModelError):
    """Model loading error."""
    
    def __init__(
        self,
        model_name: str,
        detail: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            detail=f"Failed to load model '{model_name}': {detail}",
            model_name=model_name,
            context=context
        )

class ModelInferenceError(ModelError):
    """Model inference error."""
    
    def __init__(
        self,
        model_name: str,
        detail: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            detail=f"Model inference failed for '{model_name}': {detail}",
            model_name=model_name,
            context=context
        )

# Database Errors (500)
class DatabaseError(AIVideoHTTPException):
    """Database error."""
    
    def __init__(
        self,
        detail: str,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.operation = operation
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            category=ErrorCategory.DATABASE_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context
        )

class DatabaseConnectionError(DatabaseError):
    """Database connection error."""
    
    def __init__(self, detail: str, context: Optional[ErrorContext] = None):
        
    """__init__ function."""
super().__init__(
            detail=f"Database connection failed: {detail}",
            operation="connection",
            context=context
        )

class DatabaseQueryError(DatabaseError):
    """Database query error."""
    
    def __init__(
        self,
        detail: str,
        query: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if query:
            detail = f"Database query failed: {detail} (query: {query})"
        
        super().__init__(
            detail=detail,
            operation="query",
            context=context
        )

# Cache Errors (500)
class CacheError(AIVideoHTTPException):
    """Cache error."""
    
    def __init__(
        self,
        detail: str,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if context is None:
            context = ErrorContext()
        context.operation = operation
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            category=ErrorCategory.CACHE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

class CacheConnectionError(CacheError):
    """Cache connection error."""
    
    def __init__(self, detail: str, context: Optional[ErrorContext] = None):
        
    """__init__ function."""
super().__init__(
            detail=f"Cache connection failed: {detail}",
            operation="connection",
            context=context
        )

# External Service Errors (502)
class ExternalServiceError(AIVideoHTTPException):
    """External service error."""
    
    def __init__(
        self,
        service_name: str,
        detail: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External service '{service_name}' error: {detail}",
            category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context
        )

# Rate Limit Errors (429)
class RateLimitError(AIVideoHTTPException):
    """Rate limit exceeded error."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
detail = f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"
        if retry_after:
            detail += f". Retry after {retry_after} seconds"
        
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            category=ErrorCategory.RATE_LIMIT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            headers=headers
        )

# System Errors (500)
class SystemError(AIVideoHTTPException):
    """System error."""
    
    def __init__(
        self,
        detail: str,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            context=context
        )

class MemoryError(SystemError):
    """Memory error."""
    
    def __init__(
        self,
        detail: str,
        available_memory: Optional[float] = None,
        required_memory: Optional[float] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if available_memory and required_memory:
            detail = f"{detail} (available: {available_memory}MB, required: {required_memory}MB)"
        
        super().__init__(
            detail=detail,
            context=context
        )

# Timeout Errors (408)
class TimeoutError(AIVideoHTTPException):
    """Request timeout error."""
    
    def __init__(
        self,
        detail: str,
        timeout_seconds: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
if timeout_seconds:
            detail = f"{detail} (timeout: {timeout_seconds}s)"
        
        super().__init__(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=detail,
            category=ErrorCategory.TIMEOUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

# ============================================================================
# 4. ERROR HANDLER AND RESPONSE GENERATOR
# ============================================================================

class HTTPExceptionHandler:
    """Handler for HTTP exceptions with proper response formatting."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
    
    def handle_exception(self, exc: Exception, request: Optional[Request] = None) -> JSONResponse:
        """Handle exception and return proper JSON response."""
        
        # Extract request context
        context = self._extract_request_context(request)
        
        # Handle different exception types
        if isinstance(exc, AIVideoHTTPException):
            return self._handle_ai_video_exception(exc, context)
        elif isinstance(exc, HTTPException):
            return self._handle_fastapi_exception(exc, context)
        elif isinstance(exc, RequestValidationError):
            return self._handle_validation_error(exc, context)
        else:
            return self._handle_unexpected_error(exc, context)
    
    async def _extract_request_context(self, request: Optional[Request]) -> ErrorContext:
        """Extract context from request."""
        context = ErrorContext()
        
        if request:
            context.request_id = request.headers.get("X-Request-ID")
            context.user_id = request.headers.get("X-User-ID")
            
            # Extract video_id from path parameters
            if "video_id" in request.path_params:
                context.video_id = request.path_params["video_id"]
            
            # Extract model_name from query parameters
            if "model_name" in request.query_params:
                context.model_name = request.query_params["model_name"]
        
        return context
    
    def _handle_ai_video_exception(self, exc: AIVideoHTTPException, context: ErrorContext) -> JSONResponse:
        """Handle AI Video specific exceptions."""
        # Merge contexts
        if context.user_id:
            exc.context.user_id = context.user_id
        if context.request_id:
            exc.context.request_id = context.request_id
        if context.video_id:
            exc.context.video_id = context.video_id
        if context.model_name:
            exc.context.model_name = context.model_name
        
        # Log error
        self._log_error(exc)
        
        # Return JSON response
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=exc.headers
        )
    
    async def _handle_fastapi_exception(self, exc: HTTPException, context: ErrorContext) -> JSONResponse:
        """Handle FastAPI exceptions."""
        # Convert to AI Video format
        ai_video_exc = AIVideoHTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            category=ErrorCategory.SYSTEM_ERROR,
            context=context
        )
        
        self._log_error(ai_video_exc)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ai_video_exc.to_dict(),
            headers=exc.headers
        )
    
    def _handle_validation_error(self, exc: RequestValidationError, context: ErrorContext) -> JSONResponse:
        """Handle validation errors."""
        # Extract validation details
        errors = []
        for error in exc.errors():
            errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        # Create validation error
        validation_exc = ValidationError(
            detail="Request validation failed",
            context=context
        )
        validation_exc.context.additional_data["validation_errors"] = errors
        
        self._log_error(validation_exc)
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=validation_exc.to_dict()
        )
    
    def _handle_unexpected_error(self, exc: Exception, context: ErrorContext) -> JSONResponse:
        """Handle unexpected errors."""
        # Create system error
        system_exc = SystemError(
            detail="An unexpected error occurred",
            context=context
        )
        system_exc.context.additional_data["original_error"] = str(exc)
        system_exc.context.additional_data["error_type"] = exc.__class__.__name__
        
        self._log_error(system_exc, original_exc=exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=system_exc.to_dict()
        )
    
    def _log_error(self, exc: AIVideoHTTPException, original_exc: Optional[Exception] = None):
        """Log error with appropriate level."""
        log_data = {
            "error_type": exc.__class__.__name__,
            "message": exc.detail,
            "category": exc.category.value,
            "severity": exc.severity.value,
            "status_code": exc.status_code,
            "context": {
                "user_id": exc.context.user_id,
                "request_id": exc.context.request_id,
                "video_id": exc.context.video_id,
                "model_name": exc.context.model_name,
                "operation": exc.context.operation
            }
        }
        
        if original_exc:
            log_data["original_error"] = str(original_exc)
            log_data["traceback"] = traceback.format_exc()
        
        # Log based on severity
        if exc.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(json.dumps(log_data))
        elif exc.severity == ErrorSeverity.HIGH:
            self.logger.error(json.dumps(log_data))
        elif exc.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))

# ============================================================================
# 5. ERROR MONITORING AND METRICS
# ============================================================================

class ErrorMonitor:
    """Monitor and track errors for analytics."""
    
    def __init__(self) -> Any:
        self.error_counts = {}
        self.error_timestamps = []
        self.logger = logging.getLogger(__name__)
    
    def record_error(self, exc: AIVideoHTTPException):
        """Record error for monitoring."""
        error_key = f"{exc.category.value}:{exc.__class__.__name__}"
        
        # Update counts
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Record timestamp
        self.error_timestamps.append({
            "timestamp": exc.timestamp,
            "category": exc.category.value,
            "severity": exc.severity.value,
            "status_code": exc.status_code
        })
        
        # Keep only last 1000 errors
        if len(self.error_timestamps) > 1000:
            self.error_timestamps = self.error_timestamps[-1000:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts,
            "total_errors": len(self.error_timestamps),
            "recent_errors": self.error_timestamps[-100:] if self.error_timestamps else []
        }
    
    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Calculate error rate in the last N minutes."""
        if not self.error_timestamps:
            return 0.0
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_errors = [
            error for error in self.error_timestamps
            if error["timestamp"] > cutoff_time
        ]
        
        return len(recent_errors) / window_minutes

# ============================================================================
# 6. CONTEXT MANAGERS AND DECORATORS
# ============================================================================

@contextmanager
def error_context(
    operation: str,
    user_id: Optional[str] = None,
    video_id: Optional[str] = None,
    model_name: Optional[str] = None
):
    """Context manager for error handling with context."""
    context = ErrorContext(
        user_id=user_id,
        video_id=video_id,
        model_name=model_name,
        operation=operation
    )
    
    try:
        yield context
    except AIVideoHTTPException as exc:
        # Update context
        exc.context = context
        raise
    except Exception as exc:
        # Convert to system error
        raise SystemError(
            detail=f"Error during {operation}: {str(exc)}",
            context=context
        )

def handle_errors(func) -> Any:
    """Decorator to handle errors in functions."""
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except AIVideoHTTPException:
            # Re-raise AI Video exceptions
            raise
        except Exception as exc:
            # Convert to system error
            raise SystemError(
                detail=f"Unexpected error in {func.__name__}: {str(exc)}"
            )
    
    return wrapper

# ============================================================================
# 7. USAGE EXAMPLES
# ============================================================================

async def example_video_processing_with_errors():
    """Example of video processing with proper error handling."""
    
    # Initialize error handler
    error_handler = HTTPExceptionHandler()
    error_monitor = ErrorMonitor()
    
    try:
        # Simulate video processing
        video_id = "video_123"
        model_name = "stable-diffusion"
        
        with error_context("video_generation", video_id=video_id, model_name=model_name):
            # Validate input
            if not video_id:
                raise InvalidVideoRequestError("Video ID is required", video_id)
            
            # Check if video exists
            if not await video_exists(video_id):
                raise VideoNotFoundError(video_id)
            
            # Load model
            try:
                model = await load_model(model_name)
            except Exception as e:
                raise ModelLoadError(model_name, str(e))
            
            # Generate video
            try:
                result = await generate_video(model, video_id)
            except Exception as e:
                raise VideoGenerationError(str(e), video_id, model_name)
            
            return result
            
    except AIVideoHTTPException as exc:
        # Record error for monitoring
        error_monitor.record_error(exc)
        
        # Handle with proper HTTP response
        return error_handler.handle_exception(exc)

async def example_rate_limiting():
    """Example of rate limiting with proper error handling."""
    
    try:
        # Check rate limit
        if await is_rate_limited("user_123"):
            raise RateLimitError(
                limit=100,
                window_seconds=3600,
                retry_after=60
            )
        
        # Process request
        return await process_request()
        
    except RateLimitError as exc:
        # Return proper rate limit response
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=exc.headers
        )

# Mock functions for examples
async def video_exists(video_id: str) -> bool:
    return False

async def load_model(model_name: str):
    
    """load_model function."""
raise Exception("Model not found")

async def generate_video(model, video_id: str):
    
    """generate_video function."""
raise Exception("Generation failed")

async def is_rate_limited(user_id: str) -> bool:
    return True

async def process_request():
    
    """process_request function."""
return {"status": "success"}

# ============================================================================
# 8. FASTAPI INTEGRATION
# ============================================================================

def setup_error_handlers(app) -> Any:
    """Setup error handlers for FastAPI app."""
    error_handler = HTTPExceptionHandler()
    
    @app.exception_handler(AIVideoHTTPException)
    async def ai_video_exception_handler(request: Request, exc: AIVideoHTTPException):
        
    """ai_video_exception_handler function."""
return error_handler.handle_exception(exc, request)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        
    """http_exception_handler function."""
return error_handler.handle_exception(exc, request)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        
    """validation_exception_handler function."""
return error_handler.handle_exception(exc, request)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        
    """general_exception_handler function."""
return error_handler.handle_exception(exc, request)

if __name__ == "__main__":
    # Example usage
    async def main():
        
    """main function."""
# Test error handling
        try:
            await example_video_processing_with_errors()
        except Exception as e:
            print(f"Error handled: {e}")
        
        # Test rate limiting
        await example_rate_limiting()
    
    asyncio.run(main()) 