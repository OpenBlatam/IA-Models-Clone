from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Instagram Captions API v14.0 - HTTP Exceptions
Comprehensive exception handling with specific HTTP status codes and structured responses
"""


logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")


class APIErrorResponse(BaseModel):
    """Standardized API error response"""
    error: bool = Field(default=True, description="Error flag")
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")
    status_code: int = Field(description="HTTP status code")


# =============================================================================
# VALIDATION ERRORS (4xx)
# =============================================================================

class ValidationError(HTTPException):
    """Base class for validation errors (400 Bad Request)"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_detail.model_dump()
        )


class ContentValidationError(ValidationError):
    """Content validation error"""
    
    def __init__(
        self,
        message: str = "Content validation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="CONTENT_VALIDATION_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class StyleValidationError(ValidationError):
    """Style validation error"""
    
    def __init__(
        self,
        message: str = "Invalid caption style",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="STYLE_VALIDATION_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class HashtagCountError(ValidationError):
    """Hashtag count validation error"""
    
    def __init__(
        self,
        message: str = "Invalid hashtag count",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="HASHTAG_COUNT_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class BatchSizeError(ValidationError):
    """Batch size validation error"""
    
    def __init__(
        self,
        message: str = "Invalid batch size",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="BATCH_SIZE_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class UnauthorizedError(HTTPException):
    """Unauthorized access error (401)"""
    
    def __init__(
        self,
        message: str = "Unauthorized access",
        error_code: str = "UNAUTHORIZED",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_detail.model_dump()
        )


class ForbiddenError(HTTPException):
    """Forbidden access error (403)"""
    
    def __init__(
        self,
        message: str = "Access forbidden",
        error_code: str = "FORBIDDEN",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=error_detail.model_dump()
        )


class NotFoundError(HTTPException):
    """Resource not found error (404)"""
    
    def __init__(
        self,
        message: str = "Resource not found",
        error_code: str = "NOT_FOUND",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_detail.model_dump()
        )


class MethodNotAllowedError(HTTPException):
    """Method not allowed error (405)"""
    
    def __init__(
        self,
        message: str = "Method not allowed",
        error_code: str = "METHOD_NOT_ALLOWED",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=error_detail.model_dump()
        )


class RequestTimeoutError(HTTPException):
    """Request timeout error (408)"""
    
    def __init__(
        self,
        message: str = "Request timeout",
        error_code: str = "REQUEST_TIMEOUT",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=error_detail.model_dump()
        )


class ConflictError(HTTPException):
    """Conflict error (409)"""
    
    def __init__(
        self,
        message: str = "Resource conflict",
        error_code: str = "CONFLICT",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_detail.model_dump()
        )


class TooManyRequestsError(HTTPException):
    """Rate limit exceeded error (429)"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_EXCEEDED",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
if details is None:
            details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_detail.model_dump()
        )


# =============================================================================
# SERVER ERRORS (5xx)
# =============================================================================

class InternalServerError(HTTPException):
    """Internal server error (500)"""
    
    def __init__(
        self,
        message: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail.model_dump()
        )


class AIGenerationError(InternalServerError):
    """AI generation error"""
    
    def __init__(
        self,
        message: str = "AI caption generation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="AI_GENERATION_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class CacheError(InternalServerError):
    """Cache operation error"""
    
    def __init__(
        self,
        message: str = "Cache operation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class ModelLoadingError(InternalServerError):
    """AI model loading error"""
    
    def __init__(
        self,
        message: str = "AI model loading failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="MODEL_LOADING_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class DatabaseError(InternalServerError):
    """Database operation error"""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class ExternalServiceError(InternalServerError):
    """External service error"""
    
    def __init__(
        self,
        message: str = "External service error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )


class NotImplementedError(HTTPException):
    """Not implemented error (501)"""
    
    def __init__(
        self,
        message: str = "Feature not implemented",
        error_code: str = "NOT_IMPLEMENTED",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=error_detail.model_dump()
        )


class ServiceUnavailableError(HTTPException):
    """Service unavailable error (503)"""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        error_code: str = "SERVICE_UNAVAILABLE",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        
    """__init__ function."""
if details is None:
            details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail.model_dump()
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> APIErrorResponse:
    """Create standardized error response"""
    return APIErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        path=path,
        method=method,
        status_code=status_code
    )


def handle_validation_error(
    field: str,
    value: Any,
    message: str,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> ValidationError:
    """Create validation error with field details"""
    details = {
        "field": field,
        "value": str(value),
        "error_type": "validation"
    }
    
    return ValidationError(
        message=message,
        details=details,
        request_id=request_id,
        path=path,
        method=method
    )


def handle_rate_limit_error(
    limit: int,
    window: int,
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> TooManyRequestsError:
    """Create rate limit error with details"""
    details = {
        "rate_limit": limit,
        "window_seconds": window,
        "error_type": "rate_limit"
    }
    
    return TooManyRequestsError(
        message=f"Rate limit exceeded: {limit} requests per {window} seconds",
        details=details,
        retry_after=retry_after,
        request_id=request_id,
        path=path,
        method=method
    )


def handle_ai_error(
    operation: str,
    error: Exception,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> AIGenerationError:
    """Create AI generation error with operation details"""
    details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    return AIGenerationError(
        message=f"AI {operation} failed: {str(error)}",
        details=details,
        request_id=request_id,
        path=path,
        method=method
    )


def handle_cache_error(
    operation: str,
    error: Exception,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> CacheError:
    """Create cache error with operation details"""
    details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    return CacheError(
        message=f"Cache {operation} failed: {str(error)}",
        details=details,
        request_id=request_id,
        path=path,
        method=method
    )


# =============================================================================
# ERROR CODES MAPPING
# =============================================================================

ERROR_CODES = {
    # Validation Errors (4xx)
    "VALIDATION_ERROR": ValidationError,
    "CONTENT_VALIDATION_ERROR": ContentValidationError,
    "STYLE_VALIDATION_ERROR": StyleValidationError,
    "HASHTAG_COUNT_ERROR": HashtagCountError,
    "BATCH_SIZE_ERROR": BatchSizeError,
    "UNAUTHORIZED": UnauthorizedError,
    "FORBIDDEN": ForbiddenError,
    "NOT_FOUND": NotFoundError,
    "METHOD_NOT_ALLOWED": MethodNotAllowedError,
    "REQUEST_TIMEOUT": RequestTimeoutError,
    "CONFLICT": ConflictError,
    "RATE_LIMIT_EXCEEDED": TooManyRequestsError,
    
    # Server Errors (5xx)
    "INTERNAL_ERROR": InternalServerError,
    "AI_GENERATION_ERROR": AIGenerationError,
    "CACHE_ERROR": CacheError,
    "MODEL_LOADING_ERROR": ModelLoadingError,
    "DATABASE_ERROR": DatabaseError,
    "EXTERNAL_SERVICE_ERROR": ExternalServiceError,
    "NOT_IMPLEMENTED": NotImplementedError,
    "SERVICE_UNAVAILABLE": ServiceUnavailableError,
}


def get_exception_class(error_code: str) -> type:
    """Get exception class by error code"""
    return ERROR_CODES.get(error_code, InternalServerError)


def create_exception_from_code(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> HTTPException:
    """Create exception from error code"""
    exception_class = get_exception_class(error_code)
    
    if issubclass(exception_class, ValidationError):
        return exception_class(
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
    elif issubclass(exception_class, InternalServerError):
        return exception_class(
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
    else:
        return exception_class(
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        ) 