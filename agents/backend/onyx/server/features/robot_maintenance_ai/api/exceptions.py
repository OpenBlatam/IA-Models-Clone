"""
Custom exceptions for API error handling.
Centralized exception definitions for consistent error responses.
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any


class MaintenanceAPIException(HTTPException):
    """Base exception for API errors."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class ValidationError(MaintenanceAPIException):
    """Exception for validation errors."""
    
    def __init__(self, detail: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code
        )


class NotFoundError(MaintenanceAPIException):
    """Exception for resource not found errors."""
    
    def __init__(self, resource: str, resource_id: str, error_code: str = "NOT_FOUND"):
        detail = f"{resource} '{resource_id}' not found"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code=error_code
        )


class RateLimitError(MaintenanceAPIException):
    """Exception for rate limit exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
            error_code="RATE_LIMIT_EXCEEDED"
        )


class ServiceUnavailableError(MaintenanceAPIException):
    """Exception for service unavailable errors."""
    
    def __init__(self, detail: str = "Service temporarily unavailable", error_code: str = "SERVICE_UNAVAILABLE"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code
        )


class TimeoutError(MaintenanceAPIException):
    """Exception for timeout errors."""
    
    def __init__(self, detail: str = "Request timeout", error_code: str = "TIMEOUT"):
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=detail,
            error_code=error_code
        )


class InternalServerError(MaintenanceAPIException):
    """Exception for internal server errors."""
    
    def __init__(self, detail: str = "Internal server error", error_code: str = "INTERNAL_ERROR"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code
        )






