"""
Custom exceptions for Logistics AI Platform

This module provides custom exception classes for the logistics platform,
following REST API best practices with proper HTTP status codes and
structured error responses.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class LogisticsException(HTTPException):
    """
    Base exception for logistics platform
    
    All custom exceptions inherit from this class to ensure
    consistent error handling and response formatting.
    """
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logistics exception
        
        Args:
            status_code: HTTP status code
            detail: Error message
            error_code: Machine-readable error code
            headers: Optional HTTP headers
        """
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or "LOGISTICS_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for JSON response
        
        Returns:
            Dictionary with error details
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail,
                "status_code": self.status_code
            }
        }


class NotFoundError(LogisticsException):
    """
    Resource not found exception
    
    Raised when a requested resource does not exist.
    """
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        additional_info: Optional[str] = None
    ):
        """
        Initialize not found error
        
        Args:
            resource_type: Type of resource (e.g., "Container", "Shipment")
            resource_id: Identifier of the resource
            additional_info: Optional additional information
        """
        detail = f"{resource_type} with ID '{resource_id}' not found"
        if additional_info:
            detail += f": {additional_info}"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="NOT_FOUND"
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ValidationError(LogisticsException):
    """
    Validation error exception
    
    Raised when input validation fails.
    """
    
    def __init__(
        self,
        detail: str,
        field: Optional[str] = None,
        errors: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error
        
        Args:
            detail: Error message
            field: Field name that failed validation
            errors: Optional dictionary of field-specific errors
        """
        if field:
            detail = f"Validation error in field '{field}': {detail}"
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )
        self.field = field
        self.errors = errors or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Override to include field-specific errors"""
        result = super().to_dict()
        if self.field:
            result["error"]["field"] = self.field
        if self.errors:
            result["error"]["errors"] = self.errors
        return result


class BusinessLogicError(LogisticsException):
    """
    Business logic error exception
    
    Raised when business rules are violated.
    """
    
    def __init__(
        self,
        detail: str,
        error_code: Optional[str] = None
    ):
        """
        Initialize business logic error
        
        Args:
            detail: Error message
            error_code: Optional specific error code
        """
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code or "BUSINESS_LOGIC_ERROR"
        )


class ConflictError(LogisticsException):
    """
    Resource conflict exception
    
    Raised when a resource conflict occurs (e.g., duplicate creation).
    """
    
    def __init__(
        self,
        detail: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ):
        """
        Initialize conflict error
        
        Args:
            detail: Error message
            resource_type: Type of conflicting resource
            resource_id: Identifier of conflicting resource
        """
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT"
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class RateLimitError(LogisticsException):
    """
    Rate limit exceeded exception
    
    Raised when rate limit is exceeded.
    """
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        """
        Initialize rate limit error
        
        Args:
            detail: Error message
            retry_after: Optional seconds until retry is allowed
        """
        headers = None
        if retry_after is not None:
            headers = {"Retry-After": str(retry_after)}
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers=headers
        )
        self.retry_after = retry_after


class UnauthorizedError(LogisticsException):
    """
    Unauthorized access exception
    
    Raised when authentication or authorization fails.
    """
    
    def __init__(self, detail: str = "Unauthorized access"):
        """
        Initialize unauthorized error
        
        Args:
            detail: Error message
        """
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED"
        )


class ForbiddenError(LogisticsException):
    """
    Forbidden access exception
    
    Raised when user lacks permission for the requested action.
    """
    
    def __init__(self, detail: str = "Forbidden: insufficient permissions"):
        """
        Initialize forbidden error
        
        Args:
            detail: Error message
        """
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN"
        )


class ServiceUnavailableError(LogisticsException):
    """
    Service unavailable exception
    
    Raised when a required service is temporarily unavailable.
    """
    
    def __init__(
        self,
        detail: str = "Service temporarily unavailable",
        retry_after: Optional[int] = None
    ):
        """
        Initialize service unavailable error
        
        Args:
            detail: Error message
            retry_after: Optional seconds until service may be available
        """
        headers = None
        if retry_after is not None:
            headers = {"Retry-After": str(retry_after)}
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
            headers=headers
        )
        self.retry_after = retry_after

