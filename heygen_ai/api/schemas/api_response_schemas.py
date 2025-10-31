from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Dict, List, Any, Optional, Union, Generic, TypeVar
from datetime import datetime, timezone
from enum import Enum
from pydantic import (
import uuid
from .base_schemas import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
API Response Schemas for HeyGen AI API
Consistent response formatting and error handling for all endpoints.
"""

    BaseModel, Field, validator, root_validator, 
    ConfigDict, computed_field, model_validator
)

    BaseResponse, ErrorResponse, DataResponse, PaginatedDataResponse,
    ValidationError, ValidationResponse
)

# =============================================================================
# API Response Types
# =============================================================================

class APIResponseType(str, Enum):
    """API response type enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"

class APIErrorCode(str, Enum):
    """API error code enumeration."""
    # General errors
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # Authentication errors
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Authorization errors
    FORBIDDEN = "FORBIDDEN"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    RESOURCE_ACCESS_DENIED = "RESOURCE_ACCESS_DENIED"
    
    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    HEYGEN_API_ERROR = "HEYGEN_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"

# =============================================================================
# Standard API Response Models
# =============================================================================

class StandardAPIResponse(BaseModel):
    """Standard API response model."""
    success: bool = Field(
        description="Whether the request was successful"
    )
    message: Optional[str] = Field(
        default=None,
        description="Response message"
    )
    data: Optional[Any] = Field(
        default=None,
        description="Response data"
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error information"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class SuccessAPIResponse(StandardAPIResponse):
    """Success API response model."""
    success: bool = Field(
        default=True,
        description="Success status"
    )
    data: Any = Field(
        description="Response data"
    )

class ErrorAPIResponse(StandardAPIResponse):
    """Error API response model."""
    success: bool = Field(
        default=False,
        description="Error status"
    )
    error: Dict[str, Any] = Field(
        description="Error information"
    )

# =============================================================================
# Detailed Error Response Models
# =============================================================================

class APIError(BaseModel):
    """API error model."""
    code: APIErrorCode = Field(
        description="Error code"
    )
    message: str = Field(
        description="Error message"
    )
    details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Error details"
    )
    field_errors: Optional[List[ValidationError]] = Field(
        default=None,
        description="Field-specific validation errors"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions for resolving the error"
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="Link to relevant documentation"
    )
    retry_after: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying (for rate limits)"
    )
    
    model_config = ConfigDict(
        use_enum_values=True
    )

class DetailedErrorResponse(ErrorAPIResponse):
    """Detailed error response model."""
    error: APIError = Field(
        description="Detailed error information"
    )

# =============================================================================
# Pagination Response Models
# =============================================================================

class PaginationInfo(BaseModel):
    """Pagination information model."""
    page: int = Field(
        description="Current page number"
    )
    per_page: int = Field(
        description="Items per page"
    )
    total: int = Field(
        description="Total number of items"
    )
    total_pages: int = Field(
        description="Total number of pages"
    )
    has_next: bool = Field(
        description="Whether there is a next page"
    )
    has_prev: bool = Field(
        description="Whether there is a previous page"
    )
    next_page: Optional[int] = Field(
        default=None,
        description="Next page number"
    )
    prev_page: Optional[int] = Field(
        default=None,
        description="Previous page number"
    )
    
    @computed_field
    @property
    def offset(self) -> int:
        """Calculate offset for pagination."""
        return (self.page - 1) * self.per_page

class PaginatedAPIResponse(SuccessAPIResponse):
    """Paginated API response model."""
    data: List[Any] = Field(
        description="List of items"
    )
    pagination: PaginationInfo = Field(
        description="Pagination information"
    )

# =============================================================================
# Status Response Models
# =============================================================================

class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

class ServiceInfo(BaseModel):
    """Service information model."""
    name: str = Field(
        description="Service name"
    )
    status: ServiceStatus = Field(
        description="Service status"
    )
    version: str = Field(
        description="Service version"
    )
    uptime: Optional[float] = Field(
        default=None,
        description="Service uptime in seconds"
    )
    response_time: Optional[float] = Field(
        default=None,
        description="Service response time in milliseconds"
    )
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last health check timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if unhealthy"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class HealthCheckResponse(SuccessAPIResponse):
    """Health check response model."""
    data: Dict[str, ServiceInfo] = Field(
        description="Service health information"
    )
    
    @computed_field
    @property
    def overall_status(self) -> ServiceStatus:
        """Calculate overall health status."""
        if not self.data:
            return ServiceStatus.UNHEALTHY
        
        statuses = [service.status for service in self.data.values()]
        
        if ServiceStatus.UNHEALTHY in statuses:
            return ServiceStatus.UNHEALTHY
        elif ServiceStatus.DEGRADED in statuses:
            return ServiceStatus.DEGRADED
        elif ServiceStatus.MAINTENANCE in statuses:
            return ServiceStatus.MAINTENANCE
        elif all(status == ServiceStatus.HEALTHY for status in statuses):
            return ServiceStatus.HEALTHY
        else:
            return ServiceStatus.DEGRADED

# =============================================================================
# Rate Limiting Response Models
# =============================================================================

class RateLimitInfo(BaseModel):
    """Rate limit information model."""
    limit: int = Field(
        description="Rate limit (requests per window)"
    )
    remaining: int = Field(
        description="Remaining requests in current window"
    )
    reset_time: datetime = Field(
        description="Time when rate limit resets"
    )
    window_size: int = Field(
        description="Rate limit window size in seconds"
    )
    retry_after: int = Field(
        description="Seconds to wait before retrying"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class RateLimitResponse(DetailedErrorResponse):
    """Rate limit error response model."""
    error: APIError = Field(
        description="Rate limit error information"
    )
    rate_limit_info: RateLimitInfo = Field(
        description="Rate limit information"
    )

# =============================================================================
# File Upload Response Models
# =============================================================================

class FileUploadInfo(BaseModel):
    """File upload information model."""
    file_id: str = Field(
        description="Uploaded file ID"
    )
    filename: str = Field(
        description="Original filename"
    )
    content_type: str = Field(
        description="File content type"
    )
    size: int = Field(
        description="File size in bytes"
    )
    url: Optional[str] = Field(
        default=None,
        description="File URL"
    )
    hash: Optional[str] = Field(
        default=None,
        description="File hash for integrity"
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Upload timestamp"
    )
    
    @computed_field
    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return round(self.size / (1024 * 1024), 2)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class FileUploadAPIResponse(SuccessAPIResponse):
    """File upload API response model."""
    data: FileUploadInfo = Field(
        description="File upload information"
    )

# =============================================================================
# Webhook Response Models
# =============================================================================

class WebhookEventInfo(BaseModel):
    """Webhook event information model."""
    event_id: str = Field(
        description="Event ID"
    )
    event_type: str = Field(
        description="Event type"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    data: Dict[str, Any] = Field(
        description="Event data"
    )
    source: Optional[str] = Field(
        default=None,
        description="Event source"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class WebhookResponse(SuccessAPIResponse):
    """Webhook response model."""
    data: WebhookEventInfo = Field(
        description="Webhook event information"
    )

# =============================================================================
# Batch Operation Response Models
# =============================================================================

class BatchOperationResult(BaseModel):
    """Batch operation result model."""
    operation_id: str = Field(
        description="Batch operation ID"
    )
    total_items: int = Field(
        description="Total number of items"
    )
    successful: int = Field(
        description="Number of successful operations"
    )
    failed: int = Field(
        description="Number of failed operations"
    )
    skipped: int = Field(
        default=0,
        description="Number of skipped operations"
    )
    results: List[Dict[str, Any]] = Field(
        default=[],
        description="Individual operation results"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Operation start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Operation completion time"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_items == 0:
            return 0.0
        return round((self.successful / self.total_items) * 100, 2)
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.completed_at is not None
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[int]:
        """Get operation duration in seconds."""
        if self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class BatchOperationResponse(SuccessAPIResponse):
    """Batch operation response model."""
    data: BatchOperationResult = Field(
        description="Batch operation result"
    )

# =============================================================================
# Utility Response Models
# =============================================================================

class CountResponse(SuccessAPIResponse):
    """Count response model."""
    data: int = Field(
        description="Count value"
    )

class BooleanResponse(SuccessAPIResponse):
    """Boolean response model."""
    data: bool = Field(
        description="Boolean value"
    )

class StringResponse(SuccessAPIResponse):
    """String response model."""
    data: str = Field(
        description="String value"
    )

class EmptyResponse(SuccessAPIResponse):
    """Empty response model."""
    data: None = Field(
        default=None,
        description="No data"
    )

class MessageResponse(SuccessAPIResponse):
    """Message response model."""
    data: str = Field(
        description="Response message"
    )

# =============================================================================
# Response Factory Functions
# =============================================================================

def create_success_response(
    data: Any,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> SuccessAPIResponse:
    """Create a success response."""
    return SuccessAPIResponse(
        success=True,
        data=data,
        message=message,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc)
    )

def create_error_response(
    code: APIErrorCode,
    message: str,
    details: Optional[List[Dict[str, Any]]] = None,
    field_errors: Optional[List[ValidationError]] = None,
    suggestions: Optional[List[str]] = None,
    documentation_url: Optional[str] = None,
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None
) -> DetailedErrorResponse:
    """Create an error response."""
    error = APIError(
        code=code,
        message=message,
        details=details,
        field_errors=field_errors,
        suggestions=suggestions,
        documentation_url=documentation_url,
        retry_after=retry_after
    )
    
    return DetailedErrorResponse(
        success=False,
        error=error,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc)
    )

def create_validation_error_response(
    field_errors: List[ValidationError],
    message: str = "Validation failed",
    request_id: Optional[str] = None
) -> DetailedErrorResponse:
    """Create a validation error response."""
    return create_error_response(
        code=APIErrorCode.VALIDATION_ERROR,
        message=message,
        field_errors=field_errors,
        suggestions=[
            "Check the field_errors array for specific validation issues",
            "Ensure all required fields are provided",
            "Verify data types and formats"
        ],
        request_id=request_id
    )

def create_not_found_response(
    resource_type: str,
    resource_id: str,
    request_id: Optional[str] = None
) -> DetailedErrorResponse:
    """Create a not found error response."""
    return create_error_response(
        code=APIErrorCode.RESOURCE_NOT_FOUND,
        message=f"{resource_type} with ID '{resource_id}' not found",
        suggestions=[
            f"Verify the {resource_type} ID is correct",
            f"Check if the {resource_type} has been deleted",
            "Ensure you have access to this resource"
        ],
        request_id=request_id
    )

def create_unauthorized_response(
    message: str = "Authentication required",
    request_id: Optional[str] = None
) -> DetailedErrorResponse:
    """Create an unauthorized error response."""
    return create_error_response(
        code=APIErrorCode.UNAUTHORIZED,
        message=message,
        suggestions=[
            "Provide valid authentication credentials",
            "Check if your token is valid and not expired",
            "Ensure you're using the correct authentication method"
        ],
        request_id=request_id
    )

def create_forbidden_response(
    message: str = "Access denied",
    request_id: Optional[str] = None
) -> DetailedErrorResponse:
    """Create a forbidden error response."""
    return create_error_response(
        code=APIErrorCode.FORBIDDEN,
        message=message,
        suggestions=[
            "Check your permissions for this resource",
            "Contact your administrator for access",
            "Verify your subscription level"
        ],
        request_id=request_id
    )

def create_rate_limit_response(
    rate_limit_info: RateLimitInfo,
    request_id: Optional[str] = None
) -> RateLimitResponse:
    """Create a rate limit error response."""
    error = APIError(
        code=APIErrorCode.RATE_LIMIT_EXCEEDED,
        message="Rate limit exceeded",
        suggestions=[
            "Wait before making additional requests",
            "Check your API usage limits",
            "Consider upgrading your subscription"
        ],
        retry_after=rate_limit_info.retry_after
    )
    
    return RateLimitResponse(
        success=False,
        error=error,
        rate_limit_info=rate_limit_info,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc)
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "APIResponseType",
    "APIErrorCode",
    "ServiceStatus",
    
    # Standard Response Models
    "StandardAPIResponse",
    "SuccessAPIResponse",
    "ErrorAPIResponse",
    
    # Error Response Models
    "APIError",
    "DetailedErrorResponse",
    
    # Pagination Response Models
    "PaginationInfo",
    "PaginatedAPIResponse",
    
    # Status Response Models
    "ServiceInfo",
    "HealthCheckResponse",
    
    # Rate Limiting Response Models
    "RateLimitInfo",
    "RateLimitResponse",
    
    # File Upload Response Models
    "FileUploadInfo",
    "FileUploadAPIResponse",
    
    # Webhook Response Models
    "WebhookEventInfo",
    "WebhookResponse",
    
    # Batch Operation Response Models
    "BatchOperationResult",
    "BatchOperationResponse",
    
    # Utility Response Models
    "CountResponse",
    "BooleanResponse",
    "StringResponse",
    "EmptyResponse",
    "MessageResponse",
    
    # Factory Functions
    "create_success_response",
    "create_error_response",
    "create_validation_error_response",
    "create_not_found_response",
    "create_unauthorized_response",
    "create_forbidden_response",
    "create_rate_limit_response",
] 