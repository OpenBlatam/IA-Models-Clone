from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, List, Any, Optional, Union, Generic, TypeVar
from datetime import datetime, timezone
from enum import Enum
from pydantic import (
from pydantic.generics import GenericModel
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Base Pydantic Schemas for HeyGen AI API
Consistent input/output validation and response schemas.
"""

    BaseModel, Field, validator, root_validator, 
    ConfigDict, computed_field, model_validator
)

# =============================================================================
# Base Response Models
# =============================================================================

class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status"
    )
    message: Optional[str] = Field(
        default=None,
        description="Response message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class SuccessResponse(BaseResponse):
    """Success response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Success status"
    )

class ErrorResponse(BaseResponse):
    """Error response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Error status"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for categorization"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Error type classification"
    )
    details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Detailed error information"
    )

class PaginatedResponse(BaseResponse):
    """Paginated response model."""
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
    
    @computed_field
    @property
    def offset(self) -> int:
        """Calculate offset for pagination."""
        return (self.page - 1) * self.per_page

# =============================================================================
# Generic Response Models
# =============================================================================

T = TypeVar('T')

class DataResponse(GenericModel, Generic[T]):
    """Generic data response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status"
    )
    data: T = Field(
        description="Response data"
    )
    message: Optional[str] = Field(
        default=None,
        description="Response message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class PaginatedDataResponse(GenericModel, Generic[T]):
    """Generic paginated data response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status"
    )
    data: List[T] = Field(
        description="Response data list"
    )
    pagination: PaginatedResponse = Field(
        description="Pagination information"
    )
    message: Optional[str] = Field(
        default=None,
        description="Response message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

# =============================================================================
# Base Request Models
# =============================================================================

class BaseRequest(BaseModel):
    """Base request model for all API endpoints."""
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request ID for tracking"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class PaginationRequest(BaseRequest):
    """Base pagination request model."""
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-based)"
    )
    per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page (max 100)"
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Field to sort by"
    )
    sort_order: Optional[str] = Field(
        default="asc",
        regex="^(asc|desc)$",
        description="Sort order (asc or desc)"
    )
    
    @validator('page')
    def validate_page(cls, v) -> bool:
        """Validate page number."""
        if v < 1:
            raise ValueError('Page must be greater than 0')
        return v
    
    @validator('per_page')
    def validate_per_page(cls, v) -> bool:
        """Validate per_page number."""
        if v < 1 or v > 100:
            raise ValueError('Per page must be between 1 and 100')
        return v

class SearchRequest(PaginationRequest):
    """Base search request model."""
    query: Optional[str] = Field(
        default=None,
        description="Search query"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Search filters"
    )
    
    @validator('query')
    def validate_query(cls, v) -> bool:
        """Validate search query."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v

# =============================================================================
# Common Field Models
# =============================================================================

class IDField(BaseModel):
    """ID field model."""
    id: str = Field(
        description="Unique identifier"
    )
    
    @validator('id')
    def validate_id(cls, v) -> bool:
        """Validate ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError('ID cannot be empty')
        return v.strip()

class TimestampFields(BaseModel):
    """Timestamp fields model."""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class StatusFields(BaseModel):
    """Status fields model."""
    is_active: bool = Field(
        default=True,
        description="Active status"
    )
    is_deleted: bool = Field(
        default=False,
        description="Soft delete status"
    )

class MetadataFields(BaseModel):
    """Metadata fields model."""
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for categorization"
    )
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        """Validate tags."""
        if v is not None:
            # Remove empty tags and duplicates
            tags = [tag.strip() for tag in v if tag and tag.strip()]
            return list(set(tags))
        return v

# =============================================================================
# Validation Models
# =============================================================================

class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(
        description="Field name with error"
    )
    message: str = Field(
        description="Error message"
    )
    value: Optional[Any] = Field(
        default=None,
        description="Invalid value"
    )
    code: Optional[str] = Field(
        default=None,
        description="Error code"
    )

class ValidationResponse(ErrorResponse):
    """Validation error response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Error status"
    )
    error_code: str = Field(
        default="VALIDATION_ERROR",
        description="Validation error code"
    )
    error_type: str = Field(
        default="validation",
        description="Validation error type"
    )
    details: List[ValidationError] = Field(
        description="Validation error details"
    )

# =============================================================================
# Health and Status Models
# =============================================================================

class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ServiceHealth(BaseModel):
    """Service health model."""
    name: str = Field(
        description="Service name"
    )
    status: ServiceStatus = Field(
        description="Service status"
    )
    response_time_ms: Optional[float] = Field(
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

class HealthResponse(DataResponse[Dict[str, ServiceHealth]]):
    """Health check response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Overall health status"
    )
    data: Dict[str, ServiceHealth] = Field(
        description="Service health information"
    )
    
    @computed_field
    @property
    def overall_status(self) -> ServiceStatus:
        """Calculate overall health status."""
        if not self.data:
            return ServiceStatus.UNKNOWN
        
        statuses = [service.status for service in self.data.values()]
        
        if ServiceStatus.UNHEALTHY in statuses:
            return ServiceStatus.UNHEALTHY
        elif ServiceStatus.DEGRADED in statuses:
            return ServiceStatus.DEGRADED
        elif all(status == ServiceStatus.HEALTHY for status in statuses):
            return ServiceStatus.HEALTHY
        else:
            return ServiceStatus.UNKNOWN

# =============================================================================
# Rate Limiting Models
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
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class RateLimitResponse(ErrorResponse):
    """Rate limit error response model."""
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Error status"
    )
    error_code: str = Field(
        default="RATE_LIMIT_EXCEEDED",
        description="Rate limit error code"
    )
    error_type: str = Field(
        default="rate_limit",
        description="Rate limit error type"
    )
    rate_limit_info: RateLimitInfo = Field(
        description="Rate limit information"
    )
    retry_after: int = Field(
        description="Seconds to wait before retrying"
    )

# =============================================================================
# File Upload Models
# =============================================================================

class FileInfo(BaseModel):
    """File information model."""
    filename: str = Field(
        description="Original filename"
    )
    content_type: str = Field(
        description="File content type"
    )
    size: int = Field(
        description="File size in bytes"
    )
    hash: Optional[str] = Field(
        default=None,
        description="File hash for integrity"
    )
    
    @validator('size')
    def validate_size(cls, v) -> bool:
        """Validate file size."""
        if v <= 0:
            raise ValueError('File size must be positive')
        return v

class FileUploadRequest(BaseRequest):
    """File upload request model."""
    file_info: FileInfo = Field(
        description="File information"
    )
    purpose: Optional[str] = Field(
        default=None,
        description="Upload purpose"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

class FileUploadResponse(DataResponse[str]):
    """File upload response model."""
    data: str = Field(
        description="Uploaded file ID or URL"
    )
    file_info: FileInfo = Field(
        description="File information"
    )

# =============================================================================
# Webhook Models
# =============================================================================

class WebhookEvent(BaseModel):
    """Webhook event model."""
    event_type: str = Field(
        description="Event type"
    )
    event_id: str = Field(
        description="Unique event ID"
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

class WebhookRequest(BaseRequest):
    """Webhook request model."""
    event: WebhookEvent = Field(
        description="Webhook event"
    )
    signature: Optional[str] = Field(
        default=None,
        description="Request signature for verification"
    )

# =============================================================================
# Utility Models
# =============================================================================

class EmptyResponse(BaseResponse):
    """Empty response model for endpoints that don't return data."""
    pass

class MessageResponse(BaseResponse):
    """Simple message response model."""
    message: str = Field(
        description="Response message"
    )

class CountResponse(DataResponse[int]):
    """Count response model."""
    data: int = Field(
        description="Count value"
    )

class BooleanResponse(DataResponse[bool]):
    """Boolean response model."""
    data: bool = Field(
        description="Boolean value"
    )

class StringResponse(DataResponse[str]):
    """String response model."""
    data: str = Field(
        description="String value"
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Base Response Models
    "ResponseStatus",
    "BaseResponse",
    "SuccessResponse",
    "ErrorResponse",
    "PaginatedResponse",
    
    # Generic Response Models
    "DataResponse",
    "PaginatedDataResponse",
    
    # Base Request Models
    "BaseRequest",
    "PaginationRequest",
    "SearchRequest",
    
    # Common Field Models
    "IDField",
    "TimestampFields",
    "StatusFields",
    "MetadataFields",
    
    # Validation Models
    "ValidationError",
    "ValidationResponse",
    
    # Health and Status Models
    "ServiceStatus",
    "ServiceHealth",
    "HealthResponse",
    
    # Rate Limiting Models
    "RateLimitInfo",
    "RateLimitResponse",
    
    # File Upload Models
    "FileInfo",
    "FileUploadRequest",
    "FileUploadResponse",
    
    # Webhook Models
    "WebhookEvent",
    "WebhookRequest",
    
    # Utility Models
    "EmptyResponse",
    "MessageResponse",
    "CountResponse",
    "BooleanResponse",
    "StringResponse",
] 