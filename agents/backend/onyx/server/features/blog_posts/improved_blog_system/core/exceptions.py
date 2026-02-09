"""
Custom exceptions for the blog system
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class BlogSystemException(HTTPException):
    """Base exception for blog system with additional context."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.context = context or {}


class ValidationError(BlogSystemException):
    """Validation error for invalid input data."""
    
    def __init__(
        self,
        detail: str,
        error_code: str = "VALIDATION_ERROR",
        field_errors: Optional[Dict[str, str]] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
            context={"field_errors": field_errors or {}}
        )


class NotFoundError(BlogSystemException):
    """Resource not found error."""
    
    def __init__(
        self,
        resource: str,
        resource_id: Any,
        error_code: str = "NOT_FOUND"
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id '{resource_id}' not found",
            error_code=error_code,
            context={"resource": resource, "resource_id": resource_id}
        )


class ConflictError(BlogSystemException):
    """Resource conflict error (e.g., duplicate creation)."""
    
    def __init__(
        self,
        detail: str,
        error_code: str = "CONFLICT",
        conflicting_field: Optional[str] = None
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            context={"conflicting_field": conflicting_field}
        )


class AuthenticationError(BlogSystemException):
    """Authentication error."""
    
    def __init__(
        self,
        detail: str = "Authentication failed",
        error_code: str = "AUTHENTICATION_ERROR"
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(BlogSystemException):
    """Authorization error."""
    
    def __init__(
        self,
        detail: str = "Insufficient permissions",
        error_code: str = "AUTHORIZATION_ERROR"
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code
        )


class DatabaseError(BlogSystemException):
    """Database operation error."""
    
    def __init__(
        self,
        detail: str,
        error_code: str = "DATABASE_ERROR",
        operation: Optional[str] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            context={"operation": operation}
        )


class ExternalServiceError(BlogSystemException):
    """External service error (e.g., AI service, file storage)."""
    
    def __init__(
        self,
        detail: str,
        service_name: str,
        error_code: str = "EXTERNAL_SERVICE_ERROR"
    ):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
            error_code=error_code,
            context={"service_name": service_name}
        )


class RateLimitError(BlogSystemException):
    """Rate limit exceeded error."""
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_EXCEEDED",
        retry_after: Optional[int] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code=error_code,
            headers=headers,
            context={"retry_after": retry_after}
        )


class FileUploadError(BlogSystemException):
    """File upload error."""
    
    def __init__(
        self,
        detail: str,
        error_code: str = "FILE_UPLOAD_ERROR",
        file_name: Optional[str] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            context={"file_name": file_name}
        )






