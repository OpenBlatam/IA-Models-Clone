"""
Exception classes for Enhanced Blaze AI.

This module provides custom exception classes with proper error handling,
context information, and standardized error responses.
"""

import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)


class BlazeAIError(Exception):
    """
    Base exception class for all Blaze AI errors.
    
    This class provides a foundation for all custom exceptions
    with standardized error handling and context information.
    """
    
    def __init__(
        self,
        message: str,
        detail: str = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: int = 500,
        context: Optional[ErrorContext] = None,
        retryable: bool = False,
        error_code: Optional[str] = None
    ):
        """
        Initialize BlazeAIError.
        
        Args:
            message: Human-readable error message
            detail: Detailed error description
            category: Error category for classification
            severity: Error severity level
            status_code: HTTP status code for API responses
            context: Additional context information
            retryable: Whether the error is retryable
            error_code: Unique error code for identification
        """
        super().__init__(message)
        self.message = message
        self.detail = detail or message
        self.category = category
        self.severity = severity
        self.status_code = status_code
        self.context = context or ErrorContext()
        self.retryable = retryable
        self.error_code = error_code or f"{category.value.upper()}_{status_code}"
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "detail": self.detail,
            "category": self.category.value,
            "severity": self.severity.value,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "timestamp": self.timestamp,
            "context": {
                "request_id": self.context.request_id,
                "user_id": self.context.user_id,
                "ip_address": self.context.ip_address,
                "endpoint": self.context.endpoint,
                "method": self.context.method,
                "user_agent": self.context.user_agent,
                "additional_data": self.context.additional_data
            }
        }
    
    def add_context(self, **kwargs) -> 'BlazeAIError':
        """Add context information to the error."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.additional_data[key] = value
        return self
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.__class__.__name__}: {self.message} (Code: {self.error_code})"
    
    def __repr__(self) -> str:
        """Detailed representation of the error."""
        return (f"{self.__class__.__name__}("
                f"message='{self.message}', "
                f"category={self.category.value}, "
                f"severity={self.severity.value}, "
                f"status_code={self.status_code})")


class AuthenticationError(BlazeAIError):
    """Authentication-related errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        detail: str = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            status_code=401,
            context=context,
            retryable=False
        )


class AuthorizationError(BlazeAIError):
    """Authorization-related errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        detail: str = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            status_code=403,
            context=context,
            retryable=False
        )


class ValidationError(BlazeAIError):
    """Data validation errors."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        detail: str = None,
        field_errors: Optional[Dict[str, List[str]]] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            status_code=400,
            context=context,
            retryable=False
        )
        self.field_errors = field_errors or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation error to dictionary."""
        error_dict = super().to_dict()
        if self.field_errors:
            error_dict["field_errors"] = self.field_errors
        return error_dict


class RateLimitExceededError(BlazeAIError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        detail: str = None,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            status_code=429,
            context=context,
            retryable=True
        )
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rate limit error to dictionary."""
        error_dict = super().to_dict()
        if self.retry_after:
            error_dict["retry_after"] = self.retry_after
        return error_dict


class ServiceUnavailableError(BlazeAIError):
    """Service unavailable errors."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        detail: str = None,
        service_name: Optional[str] = None,
        estimated_recovery: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            status_code=503,
            context=context,
            retryable=True
        )
        self.service_name = service_name
        self.estimated_recovery = estimated_recovery
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert service unavailable error to dictionary."""
        error_dict = super().to_dict()
        if self.service_name:
            error_dict["service_name"] = self.service_name
        if self.estimated_recovery:
            error_dict["estimated_recovery"] = self.estimated_recovery
        return error_dict


class ExternalServiceError(BlazeAIError):
    """External service errors."""
    
    def __init__(
        self,
        message: str = "External service error",
        detail: str = None,
        service_name: str = None,
        original_error: Optional[Exception] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            status_code=502,
            context=context,
            retryable=True
        )
        self.service_name = service_name
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert external service error to dictionary."""
        error_dict = super().to_dict()
        if self.service_name:
            error_dict["service_name"] = self.service_name
        if self.original_error:
            error_dict["original_error"] = str(self.original_error)
        return error_dict


class ConfigurationError(BlazeAIError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        detail: str = None,
        config_key: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            context=context,
            retryable=False
        )
        self.config_key = config_key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration error to dictionary."""
        error_dict = super().to_dict()
        if self.config_key:
            error_dict["config_key"] = self.config_key
        return error_dict


class SecurityError(BlazeAIError):
    """Security-related errors."""
    
    def __init__(
        self,
        message: str = "Security violation detected",
        detail: str = None,
        threat_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            status_code=403,
            context=context,
            retryable=False
        )
        self.threat_type = threat_type
        self.ip_address = ip_address
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security error to dictionary."""
        error_dict = super().to_dict()
        if self.threat_type:
            error_dict["threat_type"] = self.threat_type
        if self.ip_address:
            error_dict["ip_address"] = self.ip_address
        return error_dict


class DatabaseError(BlazeAIError):
    """Database-related errors."""
    
    def __init__(
        self,
        message: str = "Database error",
        detail: str = None,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            status_code=500,
            context=context,
            retryable=True
        )
        self.operation = operation
        self.table = table
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert database error to dictionary."""
        error_dict = super().to_dict()
        if self.operation:
            error_dict["operation"] = self.operation
        if self.table:
            error_dict["table"] = self.table
        return error_dict


class CacheError(BlazeAIError):
    """Cache-related errors."""
    
    def __init__(
        self,
        message: str = "Cache error",
        detail: str = None,
        operation: Optional[str] = None,
        key: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.MEDIUM,
            status_code=500,
            context=context,
            retryable=True
        )
        self.operation = operation
        self.key = key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache error to dictionary."""
        error_dict = super().to_dict()
        if self.operation:
            error_dict["operation"] = self.operation
        if self.key:
            error_dict["key"] = self.key
        return error_dict


class NetworkError(BlazeAIError):
    """Network-related errors."""
    
    def __init__(
        self,
        message: str = "Network error",
        detail: str = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            detail=detail,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            status_code=500,
            context=context,
            retryable=True
        )
        self.endpoint = endpoint
        self.timeout = timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert network error to dictionary."""
        error_dict = super().to_dict()
        if self.endpoint:
            error_dict["endpoint"] = self.endpoint
        if self.timeout:
            error_dict["timeout"] = self.timeout
        return error_dict


def create_error_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    user_agent: Optional[str] = None,
    **additional_data
) -> ErrorContext:
    """
    Create error context with the provided information.
    
    Args:
        request_id: Unique request identifier
        user_id: User identifier
        ip_address: Client IP address
        endpoint: API endpoint
        method: HTTP method
        user_agent: User agent string
        **additional_data: Additional context data
    
    Returns:
        Configured ErrorContext instance
    """
    context = ErrorContext(
        request_id=request_id,
        user_id=user_id,
        ip_address=ip_address,
        endpoint=endpoint,
        method=method,
        user_agent=user_agent
    )
    
    # Add additional data
    context.additional_data.update(additional_data)
    
    return context


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        error: Exception to check
    
    Returns:
        True if the error is retryable
    """
    if isinstance(error, BlazeAIError):
        return error.retryable
    
    # Check for common retryable errors
    retryable_error_types = (
        ConnectionError,
        TimeoutError,
        OSError,
        NetworkError
    )
    
    return isinstance(error, retryable_error_types)


def get_error_category(error: Exception) -> ErrorCategory:
    """
    Get the error category for an exception.
    
    Args:
        error: Exception to categorize
    
    Returns:
        Error category
    """
    if isinstance(error, BlazeAIError):
        return error.category
    
    # Map built-in exceptions to categories
    if isinstance(error, (ValueError, TypeError)):
        return ErrorCategory.VALIDATION
    elif isinstance(error, (ConnectionError, TimeoutError)):
        return ErrorCategory.NETWORK
    elif isinstance(error, OSError):
        return ErrorCategory.INTERNAL_ERROR
    else:
        return ErrorCategory.UNKNOWN


def get_error_severity(error: Exception) -> ErrorSeverity:
    """
    Get the error severity for an exception.
    
    Args:
        error: Exception to evaluate
    
    Returns:
        Error severity
    """
    if isinstance(error, BlazeAIError):
        return error.severity
    
    # Map built-in exceptions to severity levels
    if isinstance(error, (ValueError, TypeError)):
        return ErrorSeverity.MEDIUM
    elif isinstance(error, (ConnectionError, TimeoutError)):
        return ErrorSeverity.MEDIUM
    elif isinstance(error, OSError):
        return ErrorSeverity.HIGH
    else:
        return ErrorSeverity.MEDIUM
