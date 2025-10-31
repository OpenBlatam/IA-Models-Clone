"""
Custom Exceptions for Copywriting API
===================================

Comprehensive error handling with specific exception types.
"""

from typing import Optional, Dict, Any
from uuid import UUID


class CopywritingException(Exception):
    """Base exception for copywriting operations"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "COPYWRITING_ERROR",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[UUID] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.request_id = request_id
        super().__init__(self.message)


class ValidationError(CopywritingException):
    """Raised when input validation fails"""
    
    def __init__(
        self,
        message: str = "Invalid input data",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[UUID] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            request_id=request_id
        )


class ContentGenerationError(CopywritingException):
    """Raised when content generation fails"""
    
    def __init__(
        self,
        message: str = "Failed to generate content",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[UUID] = None
    ):
        super().__init__(
            message=message,
            error_code="CONTENT_GENERATION_ERROR",
            details=details,
            request_id=request_id
        )


class RateLimitExceededError(CopywritingException):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        request_id: Optional[UUID] = None
    ):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            request_id=request_id
        )


class ResourceNotFoundError(CopywritingException):
    """Raised when a requested resource is not found"""
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[UUID] = None
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
            
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=details,
            request_id=request_id
        )


class ExternalServiceError(CopywritingException):
    """Raised when external service fails"""
    
    def __init__(
        self,
        message: str = "External service error",
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        request_id: Optional[UUID] = None
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code
            
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
            request_id=request_id
        )


class DatabaseError(CopywritingException):
    """Raised when database operations fail"""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        request_id: Optional[UUID] = None
    ):
        details = {"operation": operation} if operation else {}
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
            request_id=request_id
        )


class ConfigurationError(CopywritingException):
    """Raised when configuration is invalid"""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        request_id: Optional[UUID] = None
    ):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            request_id=request_id
        )


class AuthenticationError(CopywritingException):
    """Raised when authentication fails"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        request_id: Optional[UUID] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            request_id=request_id
        )


class AuthorizationError(CopywritingException):
    """Raised when authorization fails"""
    
    def __init__(
        self,
        message: str = "Authorization failed",
        required_permission: Optional[str] = None,
        request_id: Optional[UUID] = None
    ):
        details = {"required_permission": required_permission} if required_permission else {}
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=details,
            request_id=request_id
        )































