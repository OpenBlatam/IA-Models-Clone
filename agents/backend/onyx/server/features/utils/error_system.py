from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import traceback
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID
from .error_system import error_factory, ErrorContext, handle_errors, ErrorCategory
from typing import Any, List, Dict, Optional
import asyncio
"""
Error System - Onyx Integration
Comprehensive error handling system with custom error types and error factories.
"""

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERIALIZATION = "serialization"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    """Context information for errors."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_data: Dict[str, Any] = field(default_factory=dict)

class OnyxBaseError(Exception):
    """Base exception for all Onyx errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
        user_friendly_message: Optional[str] = None
    ):
        
    """__init__ function."""
self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.user_friendly_message = user_friendly_message or message
        self.timestamp = datetime.utcnow()
        
        # Log the error
        self._log_error()
        
        # Call parent constructor
        super().__init__(self.user_friendly_message)
    
    def _log_error(self) -> None:
        """Log the error with appropriate level based on severity."""
        log_message = f"[{self.error_code}] {self.message}"
        
        if self.context.user_id:
            log_message += f" (User: {self.context.user_id})"
        if self.context.operation:
            log_message += f" (Operation: {self.context.operation})"
        
        if self.original_exception:
            log_message += f" (Original: {str(self.original_exception)})"
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=True)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=True)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_friendly_message": self.user_friendly_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "operation": self.context.operation,
                "resource_type": self.context.resource_type,
                "resource_id": self.context.resource_id,
                "additional_data": self.context.additional_data
            }
        }

# Validation Errors
class ValidationError(OnyxBaseError):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        validation_errors: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.field = field
        self.value = value
        self.validation_errors = validation_errors or []
        
        error_code = f"VALIDATION_{field.upper()}" if field else "VALIDATION_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_friendly_message=f"Validation error: {message}"
        )

# Authentication Errors
class AuthenticationError(OnyxBaseError):
    """Exception for authentication errors."""
    
    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.auth_method = auth_method
        
        error_code = f"AUTH_{auth_method.upper()}" if auth_method else "AUTH_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_friendly_message="Authentication failed. Please log in again."
        )

# Authorization Errors
class AuthorizationError(OnyxBaseError):
    """Exception for authorization errors."""
    
    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.required_permission = required_permission
        self.user_permissions = user_permissions or []
        
        error_code = f"AUTHZ_{required_permission.upper()}" if required_permission else "AUTHZ_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            user_friendly_message="You don't have permission to perform this action."
        )

# Database Errors
class DatabaseError(OnyxBaseError):
    """Exception for database errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        constraint: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.operation = operation
        self.table = table
        self.constraint = constraint
        
        error_code = f"DB_{operation.upper()}" if operation else "DB_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=original_exception,
            user_friendly_message="Database operation failed. Please try again later."
        )

# Cache Errors
class CacheError(OnyxBaseError):
    """Exception for cache errors."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.cache_key = cache_key
        self.operation = operation
        
        error_code = f"CACHE_{operation.upper()}" if operation else "CACHE_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=original_exception,
            user_friendly_message="Cache operation failed. Please try again."
        )

# Network Errors
class NetworkError(OnyxBaseError):
    """Exception for network errors."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: Optional[float] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.url = url
        self.status_code = status_code
        self.timeout = timeout
        
        error_code = f"NETWORK_{status_code}" if status_code else "NETWORK_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=original_exception,
            user_friendly_message="Network connection failed. Please check your connection and try again."
        )

# External Service Errors
class ExternalServiceError(OnyxBaseError):
    """Exception for external service errors."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        endpoint: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.service_name = service_name
        self.endpoint = endpoint
        self.response_data = response_data or {}
        
        error_code = f"EXTERNAL_{service_name.upper()}"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=original_exception,
            user_friendly_message=f"Service {service_name} is temporarily unavailable. Please try again later."
        )

# Resource Not Found Errors
class ResourceNotFoundError(OnyxBaseError):
    """Exception for resource not found errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.resource_type = resource_type
        self.resource_id = resource_id
        
        error_code = f"NOT_FOUND_{resource_type.upper()}"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_friendly_message=f"{resource_type.title()} not found."
        )

# Rate Limit Errors
class RateLimitError(OnyxBaseError):
    """Exception for rate limit errors."""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.limit = limit
        self.window = window
        self.retry_after = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_friendly_message="Rate limit exceeded. Please wait before trying again."
        )

# Timeout Errors
class TimeoutError(OnyxBaseError):
    """Exception for timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.timeout_duration = timeout_duration
        self.operation = operation
        
        error_code = f"TIMEOUT_{operation.upper()}" if operation else "TIMEOUT_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=original_exception,
            user_friendly_message="Operation timed out. Please try again."
        )

# Serialization Errors
class SerializationError(OnyxBaseError):
    """Exception for serialization errors."""
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        format_type: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.data_type = data_type
        self.format_type = format_type
        
        error_code = f"SERIALIZATION_{data_type.upper()}" if data_type else "SERIALIZATION_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SERIALIZATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=original_exception,
            user_friendly_message="Data processing failed. Please check your input and try again."
        )

# Business Logic Errors
class BusinessLogicError(OnyxBaseError):
    """Exception for business logic errors."""
    
    def __init__(
        self,
        message: str,
        business_rule: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        
    """__init__ function."""
self.business_rule = business_rule
        
        error_code = f"BUSINESS_{business_rule.upper()}" if business_rule else "BUSINESS_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_friendly_message=message
        )

# System Errors
class SystemError(OnyxBaseError):
    """Exception for system errors."""
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        
    """__init__ function."""
self.component = component
        
        error_code = f"SYSTEM_{component.upper()}" if component else "SYSTEM_GENERAL"
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            original_exception=original_exception,
            user_friendly_message="System error occurred. Please contact support if the problem persists."
        )

class ErrorFactory:
    """Factory for creating consistent errors."""
    
    @staticmethod
    def create_validation_error(
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        validation_errors: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None
    ) -> ValidationError:
        """Create a validation error."""
        return ValidationError(message, field, value, validation_errors, context)
    
    @staticmethod
    def create_authentication_error(
        message: str,
        auth_method: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ) -> AuthenticationError:
        """Create an authentication error."""
        return AuthenticationError(message, auth_method, context)
    
    @staticmethod
    def create_authorization_error(
        message: str,
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None
    ) -> AuthorizationError:
        """Create an authorization error."""
        return AuthorizationError(message, required_permission, user_permissions, context)
    
    @staticmethod
    def create_database_error(
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        constraint: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> DatabaseError:
        """Create a database error."""
        return DatabaseError(message, operation, table, constraint, context, original_exception)
    
    @staticmethod
    def create_cache_error(
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> CacheError:
        """Create a cache error."""
        return CacheError(message, cache_key, operation, context, original_exception)
    
    @staticmethod
    def create_network_error(
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: Optional[float] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> NetworkError:
        """Create a network error."""
        return NetworkError(message, url, status_code, timeout, context, original_exception)
    
    @staticmethod
    def create_external_service_error(
        message: str,
        service_name: str,
        endpoint: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> ExternalServiceError:
        """Create an external service error."""
        return ExternalServiceError(message, service_name, endpoint, response_data, context, original_exception)
    
    @staticmethod
    def create_resource_not_found_error(
        message: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ) -> ResourceNotFoundError:
        """Create a resource not found error."""
        return ResourceNotFoundError(message, resource_type, resource_id, context)
    
    @staticmethod
    def create_rate_limit_error(
        message: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ) -> RateLimitError:
        """Create a rate limit error."""
        return RateLimitError(message, limit, window, retry_after, context)
    
    @staticmethod
    def create_timeout_error(
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> TimeoutError:
        """Create a timeout error."""
        return TimeoutError(message, timeout_duration, operation, context, original_exception)
    
    @staticmethod
    def create_serialization_error(
        message: str,
        data_type: Optional[str] = None,
        format_type: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> SerializationError:
        """Create a serialization error."""
        return SerializationError(message, data_type, format_type, context, original_exception)
    
    @staticmethod
    def create_business_logic_error(
        message: str,
        business_rule: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ) -> BusinessLogicError:
        """Create a business logic error."""
        return BusinessLogicError(message, business_rule, context)
    
    @staticmethod
    def create_system_error(
        message: str,
        component: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ) -> SystemError:
        """Create a system error."""
        return SystemError(message, component, context, original_exception)

# Global error factory instance
error_factory = ErrorFactory()

# Error handling decorator
def handle_errors(
    error_category: ErrorCategory,
    operation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator for consistent error handling."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except OnyxBaseError:
                # Re-raise Onyx errors as they're already properly formatted
                raise
            except Exception as e:
                # Create context from function call
                context = ErrorContext(
                    operation=operation or func.__name__,
                    additional_data={
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                
                # Create appropriate error based on category
                if error_category == ErrorCategory.VALIDATION:
                    raise error_factory.create_validation_error(
                        f"Validation failed in {func.__name__}: {str(e)}",
                        context=context,
                        original_exception=e
                    )
                elif error_category == ErrorCategory.DATABASE:
                    raise error_factory.create_database_error(
                        f"Database operation failed in {func.__name__}: {str(e)}",
                        operation=operation or func.__name__,
                        context=context,
                        original_exception=e
                    )
                elif error_category == ErrorCategory.CACHE:
                    raise error_factory.create_cache_error(
                        f"Cache operation failed in {func.__name__}: {str(e)}",
                        operation=operation or func.__name__,
                        context=context,
                        original_exception=e
                    )
                elif error_category == ErrorCategory.NETWORK:
                    raise error_factory.create_network_error(
                        f"Network operation failed in {func.__name__}: {str(e)}",
                        context=context,
                        original_exception=e
                    )
                elif error_category == ErrorCategory.EXTERNAL_SERVICE:
                    raise error_factory.create_external_service_error(
                        f"External service operation failed in {func.__name__}: {str(e)}",
                        service_name="unknown",
                        context=context,
                        original_exception=e
                    )
                else:
                    raise error_factory.create_system_error(
                        f"System error in {func.__name__}: {str(e)}",
                        component=func.__module__,
                        context=context,
                        original_exception=e
                    )
        return wrapper
    return decorator

# Example usage:
"""

# Using error factory
def create_user(user_data: dict, user_id: str):
    
    """create_user function."""
try:
        # User creation logic
        if not user_data.get('email'):
            context = ErrorContext(user_id=user_id, operation="create_user")
            raise error_factory.create_validation_error(
                "Email is required",
                field="email",
                context=context
            )
        
        # Database operation
        user = database.create_user(user_data)
        return user
        
    except DatabaseError as e:
        # Re-raise as it's already properly formatted
        raise
    except Exception as e:
        # Create system error for unexpected exceptions
        context = ErrorContext(user_id=user_id, operation="create_user")
        raise error_factory.create_system_error(
            f"Failed to create user: {str(e)}",
            component="user_service",
            context=context,
            original_exception=e
        )

# Using decorator
@handle_errors(ErrorCategory.DATABASE, operation="get_user")
def get_user(user_id: str):
    
    """get_user function."""
# Database operation that might fail
    return database.get_user(user_id)

# Error handling in API
def user_api_get(user_id: str):
    
    """user_api_get function."""
try:
        user = get_user(user_id)
        if not user:
            context = ErrorContext(user_id=user_id, operation="get_user")
            raise error_factory.create_resource_not_found_error(
                "User not found",
                resource_type="user",
                resource_id=user_id,
                context=context
            )
        return user
    except OnyxBaseError as e:
        # Return structured error response
        return {"error": e.to_dict()}
""" 