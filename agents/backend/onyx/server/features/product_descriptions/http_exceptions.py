from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
import traceback
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
HTTP Exception Handling
Product Descriptions Feature - Specific HTTP Exceptions and Error Models
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorCode(str, Enum):
    """Standard error codes for the application"""
    
    # Validation errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Authentication errors (401)
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Authorization errors (403)
    FORBIDDEN = "FORBIDDEN"
    ACCESS_DENIED = "ACCESS_DENIED"
    RESOURCE_OWNERSHIP_REQUIRED = "RESOURCE_OWNERSHIP_REQUIRED"
    
    # Not found errors (404)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    
    # Conflict errors (409)
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    
    # Rate limiting errors (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    
    # Server errors (500)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # Git operation errors
    GIT_OPERATION_ERROR = "GIT_OPERATION_ERROR"
    GIT_REPOSITORY_NOT_FOUND = "GIT_REPOSITORY_NOT_FOUND"
    GIT_BRANCH_NOT_FOUND = "GIT_BRANCH_NOT_FOUND"
    GIT_COMMIT_ERROR = "GIT_COMMIT_ERROR"
    GIT_MERGE_CONFLICT = "GIT_MERGE_CONFLICT"
    
    # Model versioning errors
    MODEL_VERSION_ERROR = "MODEL_VERSION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    VERSION_ALREADY_EXISTS = "VERSION_ALREADY_EXISTS"
    MODEL_VALIDATION_ERROR = "MODEL_VALIDATION_ERROR"
    
    # Performance errors
    PERFORMANCE_ERROR = "PERFORMANCE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorContext(BaseModel):
    """Additional context for errors"""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    value: Optional[Any] = Field(default=None, description="Value that caused the error")
    expected: Optional[Any] = Field(default=None, description="Expected value or format")
    suggestion: Optional[str] = Field(default=None, description="Suggestion to fix the error")
    documentation_url: Optional[str] = Field(default=None, description="Link to relevant documentation")

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Detailed error description")
    severity: ErrorSeverity = Field(default=ErrorSeverity.MEDIUM, description="Error severity level")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")
    context: Optional[ErrorContext] = Field(default=None, description="Additional error context")
    retry_after: Optional[int] = Field(default=None, description="Retry after seconds (for rate limiting)")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for debugging")

class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details"""
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Field validation errors")

class RateLimitErrorResponse(ErrorResponse):
    """Rate limit error response"""
    limit: Optional[int] = Field(default=None, description="Request limit")
    remaining: Optional[int] = Field(default=None, description="Remaining requests")
    reset_time: Optional[datetime] = Field(default=None, description="Reset time")

class GitErrorResponse(ErrorResponse):
    """Git operation error response"""
    git_command: Optional[str] = Field(default=None, description="Git command that failed")
    git_output: Optional[str] = Field(default=None, description="Git command output")
    repository_path: Optional[str] = Field(default=None, description="Repository path")

class ModelErrorResponse(ErrorResponse):
    """Model versioning error response"""
    model_name: Optional[str] = Field(default=None, description="Model name")
    version: Optional[str] = Field(default=None, description="Model version")
    model_path: Optional[str] = Field(default=None, description="Model file path")

class PerformanceErrorResponse(ErrorResponse):
    """Performance-related error response"""
    operation: Optional[str] = Field(default=None, description="Operation that failed")
    duration: Optional[float] = Field(default=None, description="Operation duration")
    resource: Optional[str] = Field(default=None, description="Resource that caused the error")

# Custom HTTP Exceptions
class ProductDescriptionsHTTPException(HTTPException):
    """Base custom HTTP exception for Product Descriptions Feature"""
    
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        details: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        retry_after: Optional[int] = None,
        correlation_id: Optional[str] = None
    ):
        
    """__init__ function."""
self.error_code = error_code
        self.details = details
        self.severity = severity
        self.context = context
        self.request_id = request_id
        self.path = path
        self.method = method
        self.retry_after = retry_after
        self.correlation_id = correlation_id
        
        # Create error response
        error_response = ErrorResponse(
            error_code=error_code.value,
            message=message,
            details=details,
            severity=severity,
            timestamp=datetime.now(),
            request_id=request_id,
            path=path,
            method=method,
            context=context,
            retry_after=retry_after,
            correlation_id=correlation_id
        )
        
        # Add retry-after header for rate limiting
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status_code,
            detail=error_response.model_dump(),
            headers=headers
        )

# Specific HTTP Exceptions
class ValidationHTTPException(ProductDescriptionsHTTPException):
    """Validation error exception"""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[Any] = None,
        suggestion: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
context = ErrorContext(
            field=field,
            value=value,
            expected=expected,
            suggestion=suggestion
        )
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            severity=ErrorSeverity.LOW,
            context=context,
            **kwargs
        )
        
        if validation_errors:
            self.detail = ValidationErrorResponse(
                error_code=ErrorCode.VALIDATION_ERROR.value,
                message=message,
                severity=ErrorSeverity.LOW,
                context=context,
                validation_errors=validation_errors,
                **{k: v for k, v in kwargs.items() if k in ErrorResponse.__fields__}
            ).model_dump()

class UnauthorizedHTTPException(ProductDescriptionsHTTPException):
    """Unauthorized access exception"""
    
    def __init__(self, message: str = "Unauthorized access", **kwargs):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=ErrorCode.UNAUTHORIZED,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class ForbiddenHTTPException(ProductDescriptionsHTTPException):
    """Forbidden access exception"""
    
    def __init__(self, message: str = "Access forbidden", **kwargs):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=ErrorCode.FORBIDDEN,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class NotFoundHTTPException(ProductDescriptionsHTTPException):
    """Resource not found exception"""
    
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        
    """__init__ function."""
message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class ConflictHTTPException(ProductDescriptionsHTTPException):
    """Resource conflict exception"""
    
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code=ErrorCode.RESOURCE_CONFLICT,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class RateLimitHTTPException(ProductDescriptionsHTTPException):
    """Rate limit exceeded exception"""
    
    def __init__(
        self,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        retry_after: Optional[int] = 60,
        **kwargs
    ):
        
    """__init__ function."""
message = "Rate limit exceeded. Please try again later."
        
        error_response = RateLimitErrorResponse(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED.value,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            **{k: v for k, v in kwargs.items() if k in ErrorResponse.__fields__}
        )
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            **kwargs
        )
        
        self.detail = error_response.model_dump()

class GitOperationHTTPException(ProductDescriptionsHTTPException):
    """Git operation error exception"""
    
    def __init__(
        self,
        message: str,
        git_command: Optional[str] = None,
        git_output: Optional[str] = None,
        repository_path: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
error_response = GitErrorResponse(
            error_code=ErrorCode.GIT_OPERATION_ERROR.value,
            message=message,
            severity=ErrorSeverity.HIGH,
            git_command=git_command,
            git_output=git_output,
            repository_path=repository_path,
            **{k: v for k, v in kwargs.items() if k in ErrorResponse.__fields__}
        )
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.GIT_OPERATION_ERROR,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.detail = error_response.model_dump()

class ModelVersionHTTPException(ProductDescriptionsHTTPException):
    """Model versioning error exception"""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
error_response = ModelErrorResponse(
            error_code=ErrorCode.MODEL_VERSION_ERROR.value,
            message=message,
            severity=ErrorSeverity.HIGH,
            model_name=model_name,
            version=version,
            model_path=model_path,
            **{k: v for k, v in kwargs.items() if k in ErrorResponse.__fields__}
        )
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.MODEL_VERSION_ERROR,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.detail = error_response.model_dump()

class PerformanceHTTPException(ProductDescriptionsHTTPException):
    """Performance-related error exception"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        duration: Optional[float] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
error_response = PerformanceErrorResponse(
            error_code=ErrorCode.PERFORMANCE_ERROR.value,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            operation=operation,
            duration=duration,
            resource=resource,
            **{k: v for k, v in kwargs.items() if k in ErrorResponse.__fields__}
        )
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.PERFORMANCE_ERROR,
            message=message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        
        self.detail = error_response.model_dump()

class InternalServerHTTPException(ProductDescriptionsHTTPException):
    """Internal server error exception"""
    
    def __init__(self, message: str = "Internal server error", **kwargs):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message=message,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )

# Utility functions for creating exceptions
def create_validation_error(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None,
    expected: Optional[Any] = None,
    suggestion: Optional[str] = None,
    validation_errors: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ValidationHTTPException:
    """Create a validation error exception"""
    return ValidationHTTPException(
        message=message,
        field=field,
        value=value,
        expected=expected,
        suggestion=suggestion,
        validation_errors=validation_errors,
        **kwargs
    )

def create_not_found_error(
    resource_type: str,
    resource_id: str,
    **kwargs
) -> NotFoundHTTPException:
    """Create a not found error exception"""
    return NotFoundHTTPException(
        resource_type=resource_type,
        resource_id=resource_id,
        **kwargs
    )

def create_git_error(
    message: str,
    git_command: Optional[str] = None,
    git_output: Optional[str] = None,
    repository_path: Optional[str] = None,
    **kwargs
) -> GitOperationHTTPException:
    """Create a git operation error exception"""
    return GitOperationHTTPException(
        message=message,
        git_command=git_command,
        git_output=git_output,
        repository_path=repository_path,
        **kwargs
    )

def create_model_error(
    message: str,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> ModelVersionHTTPException:
    """Create a model versioning error exception"""
    return ModelVersionHTTPException(
        message=message,
        model_name=model_name,
        version=version,
        model_path=model_path,
        **kwargs
    )

def create_rate_limit_error(
    limit: Optional[int] = None,
    remaining: Optional[int] = None,
    reset_time: Optional[datetime] = None,
    retry_after: Optional[int] = 60,
    **kwargs
) -> RateLimitHTTPException:
    """Create a rate limit error exception"""
    return RateLimitHTTPException(
        limit=limit,
        remaining=remaining,
        reset_time=reset_time,
        retry_after=retry_after,
        **kwargs
    )

# Error logging utility
def log_error(
    exception: ProductDescriptionsHTTPException,
    additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """Log error with context"""
    log_data = {
        "error_code": exception.error_code.value,
        "message": exception.detail.get("message", str(exception)),
        "status_code": exception.status_code,
        "severity": exception.severity.value,
        "request_id": exception.request_id,
        "path": exception.path,
        "method": exception.method,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_context:
        log_data.update(additional_context)
    
    if exception.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"Critical error: {log_data}")
    elif exception.severity == ErrorSeverity.HIGH:
        logger.error(f"High severity error: {log_data}")
    elif exception.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"Medium severity error: {log_data}")
    else:
        logger.info(f"Low severity error: {log_data}")

# Error response utility
def create_error_response(
    exception: ProductDescriptionsHTTPException,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = exception.detail.copy()
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response 