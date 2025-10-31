"""
PDF Variantes API - Centralized Exception Handling
Custom exceptions with proper error codes and messages
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException


class BaseAPIException(HTTPException):
    """Base exception for all API exceptions"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or self.__class__.__name__
        self.metadata = metadata or {}


class ValidationError(BaseAPIException):
    """Validation error - 422"""
    def __init__(self, detail: str, field: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        if field:
            metadata["field"] = field
        super().__init__(
            status_code=422,
            detail=detail,
            error_code="VALIDATION_ERROR",
            metadata=metadata
        )


class NotFoundError(BaseAPIException):
    """Resource not found - 404"""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=404,
            detail=f"{resource_type} with id '{resource_id}' not found",
            error_code="NOT_FOUND",
            metadata={"resource_type": resource_type, "resource_id": resource_id}
        )


class ConflictError(BaseAPIException):
    """Resource conflict - 409"""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=409,
            detail=detail,
            error_code="CONFLICT",
            metadata=metadata or {}
        )


class UnauthorizedError(BaseAPIException):
    """Unauthorized access - 401"""
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=401,
            detail=detail,
            error_code="UNAUTHORIZED"
        )


class ForbiddenError(BaseAPIException):
    """Forbidden access - 403"""
    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            status_code=403,
            detail=detail,
            error_code="FORBIDDEN"
        )


class RateLimitError(BaseAPIException):
    """Rate limit exceeded - 429"""
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        metadata = {}
        if retry_after:
            metadata["retry_after"] = retry_after
        super().__init__(
            status_code=429,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            metadata=metadata
        )


class ServiceUnavailableError(BaseAPIException):
    """Service unavailable - 503"""
    def __init__(self, detail: str = "Service temporarily unavailable", service: Optional[str] = None):
        metadata = {}
        if service:
            metadata["service"] = service
        super().__init__(
            status_code=503,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
            metadata=metadata
        )


class InternalServerError(BaseAPIException):
    """Internal server error - 500"""
    def __init__(self, detail: str = "An internal error occurred", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="INTERNAL_ERROR",
            metadata=metadata or {}
        )


def format_error_response(exception: BaseAPIException) -> Dict[str, Any]:
    """Format exception for API response"""
    response = {
        "error": {
            "message": exception.detail,
            "code": exception.error_code,
            "status_code": exception.status_code,
        }
    }
    
    if exception.metadata:
        response["error"]["metadata"] = exception.metadata
    
    return response






