from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from .error_system import (
from typing import Any, List, Dict, Optional
import asyncio
"""
🚨 HTTPException System
=======================

Comprehensive HTTPException system for modeling expected errors as specific HTTP responses.
Integrates with the existing error system and provides consistent error handling across the application.
"""



    OnyxBaseError, ErrorContext, ErrorCategory, ErrorSeverity,
    ValidationError, AuthenticationError, AuthorizationError,
    DatabaseError, CacheError, NetworkError, ExternalServiceError,
    ResourceNotFoundError, RateLimitError, TimeoutError,
    SerializationError, BusinessLogicError, SystemError
)

logger = logging.getLogger(__name__)


class HTTPErrorType(Enum):
    """HTTP error types for categorization"""
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"


@dataclass
class HTTPErrorDetail:
    """Detailed error information for HTTP responses"""
    error_code: str
    message: str
    user_friendly_message: str
    category: str
    severity: str
    timestamp: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    validation_errors: Optional[List[str]] = None
    retry_after: Optional[int] = None
    help_url: Optional[str] = None


class HTTPErrorResponse(BaseModel):
    """Standardized HTTP error response model"""
    success: bool = False
    error: HTTPErrorDetail
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OnyxHTTPException(HTTPException):
    """
    Enhanced HTTPException that integrates with the Onyx error system
    and provides detailed error information.
    """
    
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        user_friendly_message: Optional[str] = None,
        category: str = "general",
        severity: str = "medium",
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        retry_after: Optional[int] = None,
        help_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        
    """__init__ function."""
# Create error detail
        error_detail = HTTPErrorDetail(
            error_code=error_code,
            message=message,
            user_friendly_message=user_friendly_message or message,
            category=category,
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            user_id=user_id,
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            additional_data=additional_data or {},
            validation_errors=validation_errors,
            retry_after=retry_after,
            help_url=help_url
        )
        
        # Create response model
        response_model = HTTPErrorResponse(
            error=error_detail,
            request_id=request_id
        )
        
        # Set headers
        final_headers = headers or {}
        if retry_after:
            final_headers["Retry-After"] = str(retry_after)
        if help_url:
            final_headers["X-Help-URL"] = help_url
        
        super().__init__(
            status_code=status_code,
            detail=response_model.dict()
        )
    
    @classmethod
    def from_onyx_error(cls, onyx_error: OnyxBaseError, status_code: int) -> "OnyxHTTPException":
        """Create HTTPException from Onyx error"""
        return cls(
            status_code=status_code,
            error_code=onyx_error.error_code,
            message=onyx_error.message,
            user_friendly_message=onyx_error.user_friendly_message,
            category=onyx_error.category.value,
            severity=onyx_error.severity.value,
            request_id=onyx_error.context.request_id,
            user_id=onyx_error.context.user_id,
            operation=onyx_error.context.operation,
            resource_type=onyx_error.context.resource_type,
            resource_id=onyx_error.context.resource_id,
            additional_data=onyx_error.context.additional_data
        )


class HTTPExceptionFactory:
    """
    Factory for creating specific HTTP exceptions with proper status codes
    and detailed error information.
    """
    
    @staticmethod
    async def bad_request(
        message: str,
        error_code: str = "BAD_REQUEST",
        field: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 400 Bad Request exception"""
        additional_data = kwargs.get("additional_data", {})
        if field:
            additional_data["field"] = field
        
        return OnyxHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            message=message,
            category="validation",
            severity="medium",
            request_id=request_id,
            validation_errors=validation_errors,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    def unauthorized(
        message: str = "Authentication required",
        error_code: str = "UNAUTHORIZED",
        auth_method: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 401 Unauthorized exception"""
        additional_data = kwargs.get("additional_data", {})
        if auth_method:
            additional_data["auth_method"] = auth_method
        
        return OnyxHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=error_code,
            message=message,
            category="authentication",
            severity="high",
            request_id=request_id,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    def forbidden(
        message: str = "Access denied",
        error_code: str = "FORBIDDEN",
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 403 Forbidden exception"""
        additional_data = kwargs.get("additional_data", {})
        if required_permission:
            additional_data["required_permission"] = required_permission
        if user_permissions:
            additional_data["user_permissions"] = user_permissions
        
        return OnyxHTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=error_code,
            message=message,
            category="authorization",
            severity="high",
            request_id=request_id,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    def not_found(
        message: str,
        error_code: str = "NOT_FOUND",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 404 Not Found exception"""
        return OnyxHTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=error_code,
            message=message,
            category="resource_not_found",
            severity="medium",
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )
    
    @staticmethod
    def method_not_allowed(
        message: str = "Method not allowed",
        error_code: str = "METHOD_NOT_ALLOWED",
        allowed_methods: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 405 Method Not Allowed exception"""
        additional_data = kwargs.get("additional_data", {})
        if allowed_methods:
            additional_data["allowed_methods"] = allowed_methods
        
        return OnyxHTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            error_code=error_code,
            message=message,
            category="client_error",
            severity="medium",
            request_id=request_id,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    def conflict(
        message: str,
        error_code: str = "CONFLICT",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 409 Conflict exception"""
        return OnyxHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            error_code=error_code,
            message=message,
            category="business_logic",
            severity="medium",
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )
    
    @staticmethod
    def unprocessable_entity(
        message: str,
        error_code: str = "UNPROCESSABLE_ENTITY",
        validation_errors: Optional[List[str]] = None,
        field: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 422 Unprocessable Entity exception"""
        additional_data = kwargs.get("additional_data", {})
        if field:
            additional_data["field"] = field
        
        return OnyxHTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=error_code,
            message=message,
            category="validation",
            severity="medium",
            request_id=request_id,
            validation_errors=validation_errors,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    async def too_many_requests(
        message: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_EXCEEDED",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 429 Too Many Requests exception"""
        additional_data = kwargs.get("additional_data", {})
        if limit:
            additional_data["limit"] = limit
        if window:
            additional_data["window"] = window
        
        return OnyxHTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=error_code,
            message=message,
            category="rate_limit",
            severity="medium",
            request_id=request_id,
            retry_after=retry_after,
            additional_data=additional_data,
            **kwargs
        )
    
    @staticmethod
    def internal_server_error(
        message: str = "Internal server error",
        error_code: str = "INTERNAL_SERVER_ERROR",
        operation: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 500 Internal Server Error exception"""
        return OnyxHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            message=message,
            category="system",
            severity="critical",
            request_id=request_id,
            operation=operation,
            **kwargs
        )
    
    @staticmethod
    def service_unavailable(
        message: str = "Service temporarily unavailable",
        error_code: str = "SERVICE_UNAVAILABLE",
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 503 Service Unavailable exception"""
        return OnyxHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=error_code,
            message=message,
            category="system",
            severity="high",
            request_id=request_id,
            retry_after=retry_after,
            **kwargs
        )
    
    @staticmethod
    def gateway_timeout(
        message: str = "Gateway timeout",
        error_code: str = "GATEWAY_TIMEOUT",
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> OnyxHTTPException:
        """Create 504 Gateway Timeout exception"""
        additional_data = kwargs.get("additional_data", {})
        if timeout_duration:
            additional_data["timeout_duration"] = timeout_duration
        
        return OnyxHTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=error_code,
            message=message,
            category="timeout",
            severity="high",
            request_id=request_id,
            operation=operation,
            additional_data=additional_data,
            **kwargs
        )


class HTTPExceptionMapper:
    """
    Maps Onyx errors to appropriate HTTP exceptions with correct status codes.
    """
    
    # Mapping of Onyx error categories to HTTP status codes
    CATEGORY_TO_STATUS_CODE = {
        ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,
        ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,
        ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,
        ErrorCategory.RESOURCE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ErrorCategory.RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorCategory.TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
        ErrorCategory.BUSINESS_LOGIC: status.HTTP_409_CONFLICT,
        ErrorCategory.DATABASE: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.CACHE: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.NETWORK: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.EXTERNAL_SERVICE: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.SERIALIZATION: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.CONFIGURATION: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    
    # Mapping of specific error types to HTTP status codes
    ERROR_TYPE_TO_STATUS_CODE = {
        ValidationError: status.HTTP_400_BAD_REQUEST,
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
        TimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
        BusinessLogicError: status.HTTP_409_CONFLICT,
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        CacheError: status.HTTP_503_SERVICE_UNAVAILABLE,
        NetworkError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ExternalServiceError: status.HTTP_503_SERVICE_UNAVAILABLE,
        SerializationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        SystemError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    
    @classmethod
    async def map_onyx_error_to_http_exception(cls, onyx_error: OnyxBaseError) -> OnyxHTTPException:
        """Map Onyx error to appropriate HTTP exception"""
        # Try to get status code from error type first
        error_type = type(onyx_error)
        if error_type in cls.ERROR_TYPE_TO_STATUS_CODE:
            status_code = cls.ERROR_TYPE_TO_STATUS_CODE[error_type]
        else:
            # Fall back to category mapping
            status_code = cls.CATEGORY_TO_STATUS_CODE.get(
                onyx_error.category, 
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Create HTTP exception
        return OnyxHTTPException.from_onyx_error(onyx_error, status_code)
    
    @classmethod
    async def map_exception_to_http_exception(cls, exception: Exception) -> OnyxHTTPException:
        """Map any exception to appropriate HTTP exception"""
        if isinstance(exception, OnyxBaseError):
            return cls.map_onyx_error_to_http_exception(exception)
        
        # Handle common Python exceptions
        if isinstance(exception, ValueError):
            return HTTPExceptionFactory.bad_request(
                message=str(exception),
                error_code="VALUE_ERROR"
            )
        elif isinstance(exception, TypeError):
            return HTTPExceptionFactory.bad_request(
                message=str(exception),
                error_code="TYPE_ERROR"
            )
        elif isinstance(exception, KeyError):
            return HTTPExceptionFactory.bad_request(
                message=f"Missing required field: {exception}",
                error_code="MISSING_FIELD"
            )
        elif isinstance(exception, IndexError):
            return HTTPExceptionFactory.bad_request(
                message=str(exception),
                error_code="INDEX_ERROR"
            )
        elif isinstance(exception, AttributeError):
            return HTTPExceptionFactory.bad_request(
                message=str(exception),
                error_code="ATTRIBUTE_ERROR"
            )
        else:
            # Default to internal server error
            return HTTPExceptionFactory.internal_server_error(
                message="An unexpected error occurred",
                error_code="UNEXPECTED_ERROR"
            )


class HTTPExceptionHandler:
    """
    Centralized handler for HTTP exceptions with logging and monitoring.
    """
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.mapper = HTTPExceptionMapper()
    
    def handle_onyx_error(self, onyx_error: OnyxBaseError) -> OnyxHTTPException:
        """Handle Onyx error and convert to HTTP exception"""
        http_exception = self.mapper.map_onyx_error_to_http_exception(onyx_error)
        
        # Log the error
        self._log_http_exception(http_exception)
        
        return http_exception
    
    def handle_exception(self, exception: Exception) -> OnyxHTTPException:
        """Handle any exception and convert to HTTP exception"""
        http_exception = self.mapper.map_exception_to_http_exception(exception)
        
        # Log the error
        self._log_http_exception(http_exception)
        
        return http_exception
    
    def _log_http_exception(self, http_exception: OnyxHTTPException):
        """Log HTTP exception with appropriate level"""
        error_detail = http_exception.detail.get("error", {})
        severity = error_detail.get("severity", "medium")
        error_code = error_detail.get("error_code", "UNKNOWN")
        message = error_detail.get("message", "Unknown error")
        
        log_message = f"[{error_code}] {message}"
        
        if severity == "critical":
            self.logger.critical(log_message, exc_info=True)
        elif severity == "high":
            self.logger.error(log_message, exc_info=True)
        elif severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)


# Convenience functions for common HTTP exceptions
async def raise_bad_request(
    message: str,
    error_code: str = "BAD_REQUEST",
    field: Optional[str] = None,
    validation_errors: Optional[List[str]] = None,
    **kwargs
) -> None:
    """Raise 400 Bad Request exception"""
    raise HTTPExceptionFactory.bad_request(
        message=message,
        error_code=error_code,
        field=field,
        validation_errors=validation_errors,
        **kwargs
    )


def raise_unauthorized(
    message: str = "Authentication required",
    error_code: str = "UNAUTHORIZED",
    **kwargs
) -> None:
    """Raise 401 Unauthorized exception"""
    raise HTTPExceptionFactory.unauthorized(
        message=message,
        error_code=error_code,
        **kwargs
    )


def raise_forbidden(
    message: str = "Access denied",
    error_code: str = "FORBIDDEN",
    **kwargs
) -> None:
    """Raise 403 Forbidden exception"""
    raise HTTPExceptionFactory.forbidden(
        message=message,
        error_code=error_code,
        **kwargs
    )


def raise_not_found(
    message: str,
    error_code: str = "NOT_FOUND",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    **kwargs
) -> None:
    """Raise 404 Not Found exception"""
    raise HTTPExceptionFactory.not_found(
        message=message,
        error_code=error_code,
        resource_type=resource_type,
        resource_id=resource_id,
        **kwargs
    )


def raise_conflict(
    message: str,
    error_code: str = "CONFLICT",
    **kwargs
) -> None:
    """Raise 409 Conflict exception"""
    raise HTTPExceptionFactory.conflict(
        message=message,
        error_code=error_code,
        **kwargs
    )


def raise_unprocessable_entity(
    message: str,
    error_code: str = "UNPROCESSABLE_ENTITY",
    validation_errors: Optional[List[str]] = None,
    **kwargs
) -> None:
    """Raise 422 Unprocessable Entity exception"""
    raise HTTPExceptionFactory.unprocessable_entity(
        message=message,
        error_code=error_code,
        validation_errors=validation_errors,
        **kwargs
    )


async def raise_too_many_requests(
    message: str = "Rate limit exceeded",
    error_code: str = "RATE_LIMIT_EXCEEDED",
    retry_after: Optional[int] = None,
    **kwargs
) -> None:
    """Raise 429 Too Many Requests exception"""
    raise HTTPExceptionFactory.too_many_requests(
        message=message,
        error_code=error_code,
        retry_after=retry_after,
        **kwargs
    )


def raise_internal_server_error(
    message: str = "Internal server error",
    error_code: str = "INTERNAL_SERVER_ERROR",
    **kwargs
) -> None:
    """Raise 500 Internal Server Error exception"""
    raise HTTPExceptionFactory.internal_server_error(
        message=message,
        error_code=error_code,
        **kwargs
    )


def raise_service_unavailable(
    message: str = "Service temporarily unavailable",
    error_code: str = "SERVICE_UNAVAILABLE",
    retry_after: Optional[int] = None,
    **kwargs
) -> None:
    """Raise 503 Service Unavailable exception"""
    raise HTTPExceptionFactory.service_unavailable(
        message=message,
        error_code=error_code,
        retry_after=retry_after,
        **kwargs
    )


# Global handler instance
http_exception_handler = HTTPExceptionHandler()


# Example usage and testing
def example_usage():
    """Example of how to use the HTTP exception system"""
    
    # Using factory methods
    try:
        raise HTTPExceptionFactory.bad_request(
            message="Invalid input data",
            error_code="INVALID_INPUT",
            field="email",
            validation_errors=["Email format is invalid"]
        )
    except OnyxHTTPException as e:
        print(f"Bad Request: {e.status_code} - {e.detail}")
    
    # Using convenience functions
    try:
        raise_not_found(
            message="User not found",
            resource_type="user",
            resource_id="123"
        )
    except OnyxHTTPException as e:
        print(f"Not Found: {e.status_code} - {e.detail}")
    
    # Using Onyx error mapping
    try:
        onyx_error = ValidationError(
            message="Invalid email format",
            field="email",
            value="invalid-email"
        )
        http_exception = http_exception_handler.handle_onyx_error(onyx_error)
        raise http_exception
    except OnyxHTTPException as e:
        print(f"Mapped Error: {e.status_code} - {e.detail}")


match __name__:
    case "__main__":
    example_usage() 