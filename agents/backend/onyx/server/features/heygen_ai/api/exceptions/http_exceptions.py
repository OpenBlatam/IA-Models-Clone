from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import (
from datetime import datetime, timezone
from enum import Enum
import structlog
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
HTTP Exceptions for HeyGen AI API
Comprehensive exception handling with specific HTTP status codes and structured responses.
"""

    Dict, List, Any, Optional, Union, Type, ClassVar
)

logger = structlog.get_logger()

# =============================================================================
# Error Categories
# =============================================================================

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_ERROR = "internal_error"
    BUSINESS_LOGIC = "business_logic"
    RESOURCE_CONFLICT = "resource_conflict"
    UNSUPPORTED_MEDIA = "unsupported_media"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    TIMEOUT = "timeout"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    value: Optional[Any] = Field(None, description="Value that caused the error")
    suggestion: Optional[str] = Field(None, description="Suggestion to fix the error")

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error_code: str = Field(..., description="Unique error code")
    message: str = Field(..., description="Human-readable error message")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    documentation_url: Optional[str] = Field(None, description="Link to documentation")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")

# =============================================================================
# Base HTTP Exception Classes
# =============================================================================

class BaseHTTPException(HTTPException):
    """Base class for all HTTP exceptions."""
    
    error_code: ClassVar[str]
    category: ClassVar[ErrorCategory]
    severity: ClassVar[ErrorSeverity]
    documentation_url: ClassVar[Optional[str]] = None
    
    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        
    """__init__ function."""
self.message = message or self.get_default_message()
        self.details = details
        self.request_id = request_id
        self.retry_after = retry_after
        
        # Set headers
        headers = kwargs.get("headers", {})
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        if request_id:
            headers["X-Request-ID"] = request_id
        
        super().__init__(
            status_code=self.status_code,
            detail=self.to_error_response().dict(),
            headers=headers
        )
    
    def get_default_message(self) -> str:
        """Get default error message."""
        return "An error occurred"
    
    def to_error_response(self) -> ErrorResponse:
        """Convert exception to error response."""
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            category=self.category,
            severity=self.severity,
            timestamp=datetime.now(timezone.utc),
            request_id=self.request_id,
            details=self.details,
            documentation_url=self.documentation_url,
            retry_after=self.retry_after
        )

# =============================================================================
# Validation Errors (400)
# =============================================================================

class ValidationError(BaseHTTPException):
    """Validation error (400)."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "VALIDATION_ERROR"
    category = ErrorCategory.VALIDATION
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Request validation failed"

class InvalidInputError(ValidationError):
    """Invalid input error."""
    error_code = "INVALID_INPUT"
    
    def get_default_message(self) -> str:
        return "Invalid input provided"

class MissingRequiredFieldError(ValidationError):
    """Missing required field error."""
    error_code = "MISSING_REQUIRED_FIELD"
    
    def get_default_message(self) -> str:
        return "Required field is missing"

class InvalidFormatError(ValidationError):
    """Invalid format error."""
    error_code = "INVALID_FORMAT"
    
    def get_default_message(self) -> str:
        return "Invalid format provided"

class InvalidVideoFormatError(ValidationError):
    """Invalid video format error."""
    error_code = "INVALID_VIDEO_FORMAT"
    
    def get_default_message(self) -> str:
        return "Invalid video format. Supported formats: mp4, mov, avi"

class InvalidScriptError(ValidationError):
    """Invalid script error."""
    error_code = "INVALID_SCRIPT"
    
    def get_default_message(self) -> str:
        return "Invalid script provided. Script must be non-empty and within length limits"

# =============================================================================
# Authentication Errors (401)
# =============================================================================

class AuthenticationError(BaseHTTPException):
    """Authentication error (401)."""
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_ERROR"
    category = ErrorCategory.AUTHENTICATION
    severity = ErrorSeverity.HIGH
    
    def get_default_message(self) -> str:
        return "Authentication required"

class InvalidCredentialsError(AuthenticationError):
    """Invalid credentials error."""
    error_code = "INVALID_CREDENTIALS"
    
    def get_default_message(self) -> str:
        return "Invalid credentials provided"

class ExpiredTokenError(AuthenticationError):
    """Expired token error."""
    error_code = "EXPIRED_TOKEN"
    
    def get_default_message(self) -> str:
        return "Authentication token has expired"

class MissingTokenError(AuthenticationError):
    """Missing token error."""
    error_code = "MISSING_TOKEN"
    
    def get_default_message(self) -> str:
        return "Authentication token is required"

# =============================================================================
# Authorization Errors (403)
# =============================================================================

class AuthorizationError(BaseHTTPException):
    """Authorization error (403)."""
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "AUTHORIZATION_ERROR"
    category = ErrorCategory.AUTHORIZATION
    severity = ErrorSeverity.HIGH
    
    def get_default_message(self) -> str:
        return "Access denied"

class InsufficientPermissionsError(AuthorizationError):
    """Insufficient permissions error."""
    error_code = "INSUFFICIENT_PERMISSIONS"
    
    def get_default_message(self) -> str:
        return "Insufficient permissions to perform this action"

class ResourceAccessDeniedError(AuthorizationError):
    """Resource access denied error."""
    error_code = "RESOURCE_ACCESS_DENIED"
    
    def get_default_message(self) -> str:
        return "Access to this resource is denied"

class SubscriptionRequiredError(AuthorizationError):
    """Subscription required error."""
    error_code = "SUBSCRIPTION_REQUIRED"
    
    def get_default_message(self) -> str:
        return "Active subscription required for this feature"

# =============================================================================
# Not Found Errors (404)
# =============================================================================

class NotFoundError(BaseHTTPException):
    """Not found error (404)."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"
    category = ErrorCategory.NOT_FOUND
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Resource not found"

class UserNotFoundError(NotFoundError):
    """User not found error."""
    error_code = "USER_NOT_FOUND"
    
    def get_default_message(self) -> str:
        return "User not found"

class VideoNotFoundError(NotFoundError):
    """Video not found error."""
    error_code = "VIDEO_NOT_FOUND"
    
    def get_default_message(self) -> str:
        return "Video not found"

class TemplateNotFoundError(NotFoundError):
    """Template not found error."""
    error_code = "TEMPLATE_NOT_FOUND"
    
    def get_default_message(self) -> str:
        return "Template not found"

class ProjectNotFoundError(NotFoundError):
    """Project not found error."""
    error_code = "PROJECT_NOT_FOUND"
    
    def get_default_message(self) -> str:
        return "Project not found"

# =============================================================================
# Rate Limit Errors (429)
# =============================================================================

class RateLimitError(BaseHTTPException):
    """Rate limit error (429)."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"
    category = ErrorCategory.RATE_LIMIT
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Rate limit exceeded"

class APIRateLimitError(RateLimitError):
    """API rate limit error."""
    error_code = "API_RATE_LIMIT_EXCEEDED"
    
    def get_default_message(self) -> str:
        return "API rate limit exceeded. Please try again later"

class VideoCreationRateLimitError(RateLimitError):
    """Video creation rate limit error."""
    error_code = "VIDEO_CREATION_RATE_LIMIT"
    
    def get_default_message(self) -> str:
        return "Video creation rate limit exceeded. Please wait before creating more videos"

# =============================================================================
# External Service Errors (502, 503, 504)
# =============================================================================

class ExternalServiceError(BaseHTTPException):
    """External service error."""
    error_code = "EXTERNAL_SERVICE_ERROR"
    category = ErrorCategory.EXTERNAL_SERVICE
    severity = ErrorSeverity.HIGH

class HeyGenAPIError(ExternalServiceError):
    """HeyGen API error."""
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "HEYGEN_API_ERROR"
    
    def get_default_message(self) -> str:
        return "HeyGen AI service is temporarily unavailable"

class VideoProcessingError(ExternalServiceError):
    """Video processing error."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "VIDEO_PROCESSING_ERROR"
    
    def get_default_message(self) -> str:
        return "Video processing service is temporarily unavailable"

class ExternalAPITimeoutError(ExternalServiceError):
    """External API timeout error."""
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "EXTERNAL_API_TIMEOUT"
    
    def get_default_message(self) -> str:
        return "External service request timed out"

# =============================================================================
# Resource Conflict Errors (409)
# =============================================================================

class ResourceConflictError(BaseHTTPException):
    """Resource conflict error (409)."""
    status_code = status.HTTP_409_CONFLICT
    error_code = "RESOURCE_CONFLICT"
    category = ErrorCategory.RESOURCE_CONFLICT
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Resource conflict occurred"

class DuplicateResourceError(ResourceConflictError):
    """Duplicate resource error."""
    error_code = "DUPLICATE_RESOURCE"
    
    def get_default_message(self) -> str:
        return "Resource already exists"

class VideoAlreadyProcessingError(ResourceConflictError):
    """Video already processing error."""
    error_code = "VIDEO_ALREADY_PROCESSING"
    
    def get_default_message(self) -> str:
        return "Video is already being processed"

# =============================================================================
# Payload Too Large Errors (413)
# =============================================================================

class PayloadTooLargeError(BaseHTTPException):
    """Payload too large error (413)."""
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "PAYLOAD_TOO_LARGE"
    category = ErrorCategory.PAYLOAD_TOO_LARGE
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Request payload is too large"

class VideoFileTooLargeError(PayloadTooLargeError):
    """Video file too large error."""
    error_code = "VIDEO_FILE_TOO_LARGE"
    
    def get_default_message(self) -> str:
        return "Video file size exceeds the maximum allowed limit"

class ScriptTooLongError(PayloadTooLargeError):
    """Script too long error."""
    error_code = "SCRIPT_TOO_LONG"
    
    def get_default_message(self) -> str:
        return "Script length exceeds the maximum allowed limit"

# =============================================================================
# Unsupported Media Errors (415)
# =============================================================================

class UnsupportedMediaError(BaseHTTPException):
    """Unsupported media error (415)."""
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    error_code = "UNSUPPORTED_MEDIA"
    category = ErrorCategory.UNSUPPORTED_MEDIA
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Unsupported media type"

class UnsupportedVideoFormatError(UnsupportedMediaError):
    """Unsupported video format error."""
    error_code = "UNSUPPORTED_VIDEO_FORMAT"
    
    def get_default_message(self) -> str:
        return "Unsupported video format. Please use mp4, mov, or avi"

# =============================================================================
# Business Logic Errors (422)
# =============================================================================

class BusinessLogicError(BaseHTTPException):
    """Business logic error (422)."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "BUSINESS_LOGIC_ERROR"
    category = ErrorCategory.BUSINESS_LOGIC
    severity = ErrorSeverity.MEDIUM
    
    def get_default_message(self) -> str:
        return "Business logic validation failed"

class VideoDurationLimitError(BusinessLogicError):
    """Video duration limit error."""
    error_code = "VIDEO_DURATION_LIMIT"
    
    def get_default_message(self) -> str:
        return "Video duration exceeds the maximum allowed limit"

class InvalidVideoTemplateError(BusinessLogicError):
    """Invalid video template error."""
    error_code = "INVALID_VIDEO_TEMPLATE"
    
    def get_default_message(self) -> str:
        return "Invalid video template selected"

class ScriptContentError(BusinessLogicError):
    """Script content error."""
    error_code = "SCRIPT_CONTENT_ERROR"
    
    def get_default_message(self) -> str:
        return "Script content violates content guidelines"

# =============================================================================
# Internal Server Errors (500)
# =============================================================================

class InternalServerError(BaseHTTPException):
    """Internal server error (500)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_SERVER_ERROR"
    category = ErrorCategory.INTERNAL_ERROR
    severity = ErrorSeverity.CRITICAL
    
    def get_default_message(self) -> str:
        return "Internal server error occurred"

class DatabaseError(InternalServerError):
    """Database error."""
    error_code = "DATABASE_ERROR"
    
    def get_default_message(self) -> str:
        return "Database operation failed"

class CacheError(InternalServerError):
    """Cache error."""
    error_code = "CACHE_ERROR"
    
    def get_default_message(self) -> str:
        return "Cache operation failed"

class ConfigurationError(InternalServerError):
    """Configuration error."""
    error_code = "CONFIGURATION_ERROR"
    
    def get_default_message(self) -> str:
        return "Configuration error occurred"

# =============================================================================
# Exception Factory
# =============================================================================

class ExceptionFactory:
    """Factory for creating HTTP exceptions."""
    
    @staticmethod
    def create_validation_error(
        field: str,
        message: str,
        value: Any = None,
        suggestion: str = None
    ) -> ValidationError:
        """Create a validation error."""
        detail = ErrorDetail(
            field=field,
            message=message,
            value=value,
            suggestion=suggestion
        )
        return ValidationError(details=[detail])
    
    @staticmethod
    def create_rate_limit_error(
        retry_after: int,
        message: str = None
    ) -> RateLimitError:
        """Create a rate limit error."""
        return RateLimitError(
            message=message,
            retry_after=retry_after
        )
    
    @staticmethod
    def create_external_service_error(
        service_name: str,
        error_message: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY
    ) -> ExternalServiceError:
        """Create an external service error."""
        return ExternalServiceError(
            status_code=status_code,
            message=f"{service_name} error: {error_message}"
        )

# =============================================================================
# Exception Handler Utilities
# =============================================================================

def log_exception(exception: BaseHTTPException, request_id: str = None):
    """Log exception with structured logging."""
    logger.error(
        "HTTP exception raised",
        error_code=exception.error_code,
        message=exception.message,
        category=exception.category.value,
        severity=exception.severity.value,
        status_code=exception.status_code,
        request_id=request_id,
        details=exception.details
    )

def create_error_response(
    exception: BaseHTTPException,
    request_id: str = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    if request_id:
        exception.request_id = request_id
    
    return exception.to_error_response().dict()

# =============================================================================
# Error Code Registry
# =============================================================================

ERROR_CODE_REGISTRY = {
    # Validation Errors
    "VALIDATION_ERROR": ValidationError,
    "INVALID_INPUT": InvalidInputError,
    "MISSING_REQUIRED_FIELD": MissingRequiredFieldError,
    "INVALID_FORMAT": InvalidFormatError,
    "INVALID_VIDEO_FORMAT": InvalidVideoFormatError,
    "INVALID_SCRIPT": InvalidScriptError,
    
    # Authentication Errors
    "AUTHENTICATION_ERROR": AuthenticationError,
    "INVALID_CREDENTIALS": InvalidCredentialsError,
    "EXPIRED_TOKEN": ExpiredTokenError,
    "MISSING_TOKEN": MissingTokenError,
    
    # Authorization Errors
    "AUTHORIZATION_ERROR": AuthorizationError,
    "INSUFFICIENT_PERMISSIONS": InsufficientPermissionsError,
    "RESOURCE_ACCESS_DENIED": ResourceAccessDeniedError,
    "SUBSCRIPTION_REQUIRED": SubscriptionRequiredError,
    
    # Not Found Errors
    "NOT_FOUND": NotFoundError,
    "USER_NOT_FOUND": UserNotFoundError,
    "VIDEO_NOT_FOUND": VideoNotFoundError,
    "TEMPLATE_NOT_FOUND": TemplateNotFoundError,
    "PROJECT_NOT_FOUND": ProjectNotFoundError,
    
    # Rate Limit Errors
    "RATE_LIMIT_EXCEEDED": RateLimitError,
    "API_RATE_LIMIT_EXCEEDED": APIRateLimitError,
    "VIDEO_CREATION_RATE_LIMIT": VideoCreationRateLimitError,
    
    # External Service Errors
    "EXTERNAL_SERVICE_ERROR": ExternalServiceError,
    "HEYGEN_API_ERROR": HeyGenAPIError,
    "VIDEO_PROCESSING_ERROR": VideoProcessingError,
    "EXTERNAL_API_TIMEOUT": ExternalAPITimeoutError,
    
    # Resource Conflict Errors
    "RESOURCE_CONFLICT": ResourceConflictError,
    "DUPLICATE_RESOURCE": DuplicateResourceError,
    "VIDEO_ALREADY_PROCESSING": VideoAlreadyProcessingError,
    
    # Payload Too Large Errors
    "PAYLOAD_TOO_LARGE": PayloadTooLargeError,
    "VIDEO_FILE_TOO_LARGE": VideoFileTooLargeError,
    "SCRIPT_TOO_LONG": ScriptTooLongError,
    
    # Unsupported Media Errors
    "UNSUPPORTED_MEDIA": UnsupportedMediaError,
    "UNSUPPORTED_VIDEO_FORMAT": UnsupportedVideoFormatError,
    
    # Business Logic Errors
    "BUSINESS_LOGIC_ERROR": BusinessLogicError,
    "VIDEO_DURATION_LIMIT": VideoDurationLimitError,
    "INVALID_VIDEO_TEMPLATE": InvalidVideoTemplateError,
    "SCRIPT_CONTENT_ERROR": ScriptContentError,
    
    # Internal Server Errors
    "INTERNAL_SERVER_ERROR": InternalServerError,
    "DATABASE_ERROR": DatabaseError,
    "CACHE_ERROR": CacheError,
    "CONFIGURATION_ERROR": ConfigurationError,
}

def get_exception_class(error_code: str) -> Type[BaseHTTPException]:
    """Get exception class by error code."""
    return ERROR_CODE_REGISTRY.get(error_code, BaseHTTPException)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorDetail",
    "ErrorResponse",
    "BaseHTTPException",
    "ValidationError",
    "InvalidInputError",
    "MissingRequiredFieldError",
    "InvalidFormatError",
    "InvalidVideoFormatError",
    "InvalidScriptError",
    "AuthenticationError",
    "InvalidCredentialsError",
    "ExpiredTokenError",
    "MissingTokenError",
    "AuthorizationError",
    "InsufficientPermissionsError",
    "ResourceAccessDeniedError",
    "SubscriptionRequiredError",
    "NotFoundError",
    "UserNotFoundError",
    "VideoNotFoundError",
    "TemplateNotFoundError",
    "ProjectNotFoundError",
    "RateLimitError",
    "APIRateLimitError",
    "VideoCreationRateLimitError",
    "ExternalServiceError",
    "HeyGenAPIError",
    "VideoProcessingError",
    "ExternalAPITimeoutError",
    "ResourceConflictError",
    "DuplicateResourceError",
    "VideoAlreadyProcessingError",
    "PayloadTooLargeError",
    "VideoFileTooLargeError",
    "ScriptTooLongError",
    "UnsupportedMediaError",
    "UnsupportedVideoFormatError",
    "BusinessLogicError",
    "VideoDurationLimitError",
    "InvalidVideoTemplateError",
    "ScriptContentError",
    "InternalServerError",
    "DatabaseError",
    "CacheError",
    "ConfigurationError",
    "ExceptionFactory",
    "log_exception",
    "create_error_response",
    "ERROR_CODE_REGISTRY",
    "get_exception_class",
] 