from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from fastapi import status
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
📋 HTTP Response Models
======================

Standardized HTTP response models for consistent API responses.
Provides structured success and error response formats.
"""



T = TypeVar('T')


class ResponseStatus(Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL_SUCCESS = "partial_success"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PaginationInfo(BaseModel):
    """Pagination information for list responses"""
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values) -> Any:
        """Calculate total pages based on total items and per page"""
        if 'total_items' in values and 'per_page' in values:
            return (values['total_items'] + values['per_page'] - 1) // values['per_page']
        return v


class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(..., description="Unique error code")
    message: str = Field(..., description="Technical error message")
    user_friendly_message: str = Field(..., description="User-friendly error message")
    category: str = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    operation: Optional[str] = Field(None, description="Operation that failed")
    resource_type: Optional[str] = Field(None, description="Type of resource involved")
    resource_id: Optional[str] = Field(None, description="ID of resource involved")
    validation_errors: Optional[List[str]] = Field(None, description="Validation error details")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")
    help_url: Optional[str] = Field(None, description="Help URL for this error")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional error data")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response model"""
    success: bool = Field(True, description="Always true for success responses")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="Response status")
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = Field(False, description="Always false for error responses")
    status: ResponseStatus = Field(ResponseStatus.ERROR, description="Response status")
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PartialSuccessResponse(BaseModel, Generic[T]):
    """Response model for partial success scenarios"""
    success: bool = Field(True, description="True for partial success")
    status: ResponseStatus = Field(ResponseStatus.PARTIAL_SUCCESS, description="Response status")
    data: T = Field(..., description="Successful data")
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors")
    message: Optional[str] = Field(None, description="Partial success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ListResponse(BaseModel, Generic[T]):
    """Standard list response model"""
    success: bool = Field(True, description="Always true for success responses")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="Response status")
    data: List[T] = Field(..., description="List of items")
    total_count: int = Field(..., description="Total number of items")
    message: Optional[str] = Field(None, description="Success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    success: bool = Field(True, description="Health check status")
    status: str = Field(..., description="Health status (healthy, unhealthy, degraded)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: Optional[str] = Field(None, description="Application version")
    uptime: Optional[float] = Field(None, description="Application uptime in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual health checks")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """Metrics response model"""
    success: bool = Field(True, description="Metrics retrieval status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    metrics: Dict[str, Any] = Field(..., description="Application metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchResponse(BaseModel, Generic[T]):
    """Batch operation response model"""
    success: bool = Field(True, description="Batch operation status")
    status: ResponseStatus = Field(..., description="Response status")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Number of successful items")
    failed_items: int = Field(..., description="Number of failed items")
    data: List[T] = Field(default_factory=list, description="Successful results")
    errors: List[ErrorDetail] = Field(default_factory=list, description="Error details for failed items")
    message: Optional[str] = Field(None, description="Batch operation message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResponseFactory:
    """
    Factory for creating standardized HTTP responses.
    """
    
    @staticmethod
    def success_response(
        data: Any,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        pagination: Optional[PaginationInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SuccessResponse:
        """Create a success response"""
        return SuccessResponse(
            data=data,
            message=message,
            request_id=request_id,
            pagination=pagination,
            metadata=metadata or {}
        )
    
    @staticmethod
    def list_response(
        data: List[Any],
        total_count: int,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        pagination: Optional[PaginationInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ListResponse:
        """Create a list response"""
        return ListResponse(
            data=data,
            total_count=total_count,
            message=message,
            request_id=request_id,
            pagination=pagination,
            metadata=metadata or {}
        )
    
    @staticmethod
    def error_response(
        error_detail: ErrorDetail,
        request_id: Optional[str] = None
    ) -> ErrorResponse:
        """Create an error response"""
        return ErrorResponse(
            error=error_detail,
            request_id=request_id
        )
    
    @staticmethod
    def partial_success_response(
        data: Any,
        errors: List[ErrorDetail],
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        pagination: Optional[PaginationInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PartialSuccessResponse:
        """Create a partial success response"""
        return PartialSuccessResponse(
            data=data,
            errors=errors,
            message=message,
            request_id=request_id,
            pagination=pagination,
            metadata=metadata or {}
        )
    
    @staticmethod
    def batch_response(
        total_items: int,
        successful_items: int,
        failed_items: int,
        data: List[Any],
        errors: List[ErrorDetail],
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchResponse:
        """Create a batch response"""
        status = ResponseStatus.SUCCESS if failed_items == 0 else ResponseStatus.PARTIAL_SUCCESS
        
        return BatchResponse(
            status=status,
            total_items=total_items,
            successful_items=successful_items,
            failed_items=failed_items,
            data=data,
            errors=errors,
            message=message,
            request_id=request_id,
            metadata=metadata or {}
        )
    
    @staticmethod
    def health_check_response(
        status: str,
        version: Optional[str] = None,
        uptime: Optional[float] = None,
        checks: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> HealthCheckResponse:
        """Create a health check response"""
        return HealthCheckResponse(
            status=status,
            version=version,
            uptime=uptime,
            checks=checks or {}
        )
    
    @staticmethod
    def metrics_response(
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MetricsResponse:
        """Create a metrics response"""
        return MetricsResponse(
            metrics=metrics,
            metadata=metadata or {}
        )


class HTTPStatusCodes:
    """
    Constants for HTTP status codes with descriptions.
    """
    
    # Success responses
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client error responses
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    CONFLICT = 409
    GONE = 410
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # Server error responses
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    
    # Status code descriptions
    DESCRIPTIONS = {
        200: "OK",
        201: "Created",
        202: "Accepted",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        409: "Conflict",
        410: "Gone",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout"
    }


# Convenience functions for creating responses
def create_success_response(
    data: Any,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> SuccessResponse:
    """Create a success response"""
    return ResponseFactory.success_response(data, message, request_id, **kwargs)


def create_error_response(
    error_code: str,
    message: str,
    user_friendly_message: Optional[str] = None,
    category: str = "general",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    request_id: Optional[str] = None,
    **kwargs
) -> ErrorResponse:
    """Create an error response"""
    error_detail = ErrorDetail(
        error_code=error_code,
        message=message,
        user_friendly_message=user_friendly_message or message,
        category=category,
        severity=severity,
        request_id=request_id,
        **kwargs
    )
    return ResponseFactory.error_response(error_detail, request_id)


def create_list_response(
    data: List[Any],
    total_count: int,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> ListResponse:
    """Create a list response"""
    return ResponseFactory.list_response(data, total_count, message, request_id, **kwargs)


def create_batch_response(
    total_items: int,
    successful_items: int,
    failed_items: int,
    data: List[Any],
    errors: List[ErrorDetail],
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> BatchResponse:
    """Create a batch response"""
    return ResponseFactory.batch_response(
        total_items, successful_items, failed_items,
        data, errors, message, request_id, **kwargs
    )


# Example usage
def example_usage():
    """Example of how to use the response models"""
    
    # Success response
    user_data = {"id": "123", "name": "John Doe", "email": "john@example.com"}
    success_response = create_success_response(
        data=user_data,
        message="User retrieved successfully",
        request_id="req-123"
    )
    print("Success Response:", success_response.dict())
    
    # Error response
    error_response = create_error_response(
        error_code="USER_NOT_FOUND",
        message="User with ID 123 not found in database",
        user_friendly_message="User not found",
        category="resource_not_found",
        severity=ErrorSeverity.MEDIUM,
        request_id="req-123"
    )
    print("Error Response:", error_response.dict())
    
    # List response
    users = [
        {"id": "1", "name": "John"},
        {"id": "2", "name": "Jane"}
    ]
    list_response = create_list_response(
        data=users,
        total_count=2,
        message="Users retrieved successfully",
        request_id="req-123"
    )
    print("List Response:", list_response.dict())
    
    # Batch response
    batch_errors = [
        ErrorDetail(
            error_code="VALIDATION_ERROR",
            message="Invalid email format",
            user_friendly_message="Please provide a valid email address",
            category="validation",
            severity=ErrorSeverity.MEDIUM
        )
    ]
    batch_response = create_batch_response(
        total_items=3,
        successful_items=2,
        failed_items=1,
        data=users,
        errors=batch_errors,
        message="Batch operation completed with some errors",
        request_id="req-123"
    )
    print("Batch Response:", batch_response.dict())


match __name__:
    case "__main__":
    example_usage() 