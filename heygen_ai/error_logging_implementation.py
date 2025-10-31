from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import logging
import traceback
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
from contextlib import contextmanager
import functools
        from logging.handlers import RotatingFileHandler
from typing import Any, List, Dict, Optional
import asyncio
"""
Error Logging and User-Friendly Error Messages Implementation
===========================================================

This module demonstrates:
- Structured error logging with different levels
- User-friendly error messages
- Error hierarchies and custom exceptions
- Production-ready error handling
- Context-aware error reporting
- Error tracking and monitoring
"""



# ============================================================================
# Error Types and Enums
# ============================================================================

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    NETWORK = "network"
    EXTERNAL_API = "external_api"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorContext(Enum):
    """Error context for better understanding"""
    USER_INPUT = "user_input"
    DATABASE_OPERATION = "database_operation"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    MODEL_TRAINING = "model_training"
    DATA_PROCESSING = "data_processing"
    CONFIGURATION_LOADING = "configuration_loading"
    AUTHENTICATION_PROCESS = "authentication_process"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ErrorDetails:
    """Structured error details"""
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    message: str
    user_message: str
    technical_details: Optional[str] = None
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    message: str
    error_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# ============================================================================
# Custom Exceptions
# ============================================================================

class BaseAppException(Exception):
    """Base exception class for the application"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: ErrorContext = ErrorContext.USER_INPUT,
        error_id: Optional[str] = None,
        technical_details: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.user_message = user_message
        self.severity = severity
        self.category = category
        self.context = context
        self.error_id = error_id or str(uuid.uuid4())
        self.technical_details = technical_details
        self.additional_data = additional_data or {}
        self.timestamp = datetime.now().isoformat()


class ValidationError(BaseAppException):
    """Validation error exception"""
    
    def __init__(self, message: str, user_message: str, field: Optional[str] = None):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            context=ErrorContext.USER_INPUT,
            additional_data={"field": field} if field else {}
        )


class AuthenticationError(BaseAppException):
    """Authentication error exception"""
    
    def __init__(self, message: str, user_message: str = "Authentication failed"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            context=ErrorContext.AUTHENTICATION_PROCESS
        )


class AuthorizationError(BaseAppException):
    """Authorization error exception"""
    
    def __init__(self, message: str, user_message: str = "Access denied"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHORIZATION,
            context=ErrorContext.USER_INPUT
        )


class DatabaseError(BaseAppException):
    """Database error exception"""
    
    def __init__(self, message: str, user_message: str = "Database operation failed"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            context=ErrorContext.DATABASE_OPERATION
        )


class ExternalAPIError(BaseAppException):
    """External API error exception"""
    
    def __init__(self, message: str, user_message: str = "External service temporarily unavailable"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_API,
            context=ErrorContext.API_CALL
        )


class ConfigurationError(BaseAppException):
    """Configuration error exception"""
    
    def __init__(self, message: str, user_message: str = "System configuration error"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=ErrorContext.CONFIGURATION_LOADING
        )


class ModelTrainingError(BaseAppException):
    """Model training error exception"""
    
    def __init__(self, message: str, user_message: str = "Model training failed"):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.MODEL_TRAINING
        )


# ============================================================================
# Error Logger Implementation
# ============================================================================

class ErrorLogger:
    """Comprehensive error logging system"""
    
    def __init__(
        self,
        log_file: str = "logs/errors.log",
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        
    """__init__ function."""
self.log_file = log_file
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("ErrorLogger")
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()  # Remove existing handlers
        
        # Add handlers
        if self.enable_console:
            self._setup_console_handler()
        
        if self.enable_file:
            self._setup_file_handler()
        
        if self.enable_json:
            self._setup_json_handler()
    
    def _setup_console_handler(self) -> Any:
        """Setup console handler for development"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self) -> Any:
        """Setup file handler with rotation"""
        
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self) -> Any:
        """Setup JSON handler for structured logging"""
        json_handler = logging.FileHandler(self.log_file.replace('.log', '_json.log'))
        json_handler.setLevel(self.log_level)
        
        class JSONFormatter(logging.Formatter):
            def format(self, record) -> Any:
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'error_id'):
                    log_entry['error_id'] = record.error_id
                if hasattr(record, 'user_id'):
                    log_entry['user_id'] = record.user_id
                if hasattr(record, 'session_id'):
                    log_entry['session_id'] = record.session_id
                if hasattr(record, 'request_id'):
                    log_entry['request_id'] = record.request_id
                
                return json.dumps(log_entry)
        
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
    
    def log_error(
        self,
        error: Union[Exception, BaseAppException],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an error with structured information"""
        
        # Generate error ID if not present
        if isinstance(error, BaseAppException):
            error_id = error.error_id
            severity = error.severity.value
            category = error.category.value
            context = error.context.value
            message = error.message
            user_message = error.user_message
            technical_details = error.technical_details
            additional_data = error.additional_data
        else:
            error_id = str(uuid.uuid4())
            severity = ErrorSeverity.MEDIUM.value
            category = ErrorCategory.UNKNOWN.value
            context = ErrorContext.USER_INPUT.value
            message = str(error)
            user_message = "An unexpected error occurred"
            technical_details = traceback.format_exc()
            additional_data = {}
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            severity=ErrorSeverity(severity),
            category=ErrorCategory(category),
            context=ErrorContext(context),
            message=message,
            user_message=user_message,
            technical_details=technical_details,
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            additional_data={**additional_data, **(additional_context or {})}
        )
        
        # Log with appropriate level
        log_level = self._get_log_level(severity)
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            name="ErrorLogger",
            level=log_level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=(type(error), error, error.__traceback__)
        )
        
        # Add extra fields to record
        record.error_id = error_id
        record.user_id = user_id
        record.session_id = session_id
        record.request_id = request_id
        
        self.logger.handle(record)
        
        # Log structured error details
        self.logger.info(f"Error Details: {json.dumps(asdict(error_details), indent=2)}")
        
        return error_id
    
    def _get_log_level(self, severity: str) -> int:
        """Map severity to logging level"""
        severity_map = {
            ErrorSeverity.LOW.value: logging.INFO,
            ErrorSeverity.MEDIUM.value: logging.WARNING,
            ErrorSeverity.HIGH.value: logging.ERROR,
            ErrorSeverity.CRITICAL.value: logging.CRITICAL
        }
        return severity_map.get(severity, logging.ERROR)
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def log_critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)


# ============================================================================
# Error Handler Implementation
# ============================================================================

class ErrorHandler:
    """Error handler for managing and responding to errors"""
    
    def __init__(self, logger: ErrorLogger):
        
    """__init__ function."""
self.logger = logger
        self.error_counts = {}
        self.error_thresholds = {
            ErrorSeverity.LOW: 100,
            ErrorSeverity.MEDIUM: 50,
            ErrorSeverity.HIGH: 10,
            ErrorSeverity.CRITICAL: 1
        }
    
    def handle_error(
        self,
        error: Union[Exception, BaseAppException],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle an error and return user-friendly response"""
        
        # Log the error
        error_id = self.logger.log_error(
            error=error,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            additional_context=context
        )
        
        # Update error counts
        self._update_error_counts(error)
        
        # Check if threshold exceeded
        if self._check_threshold_exceeded(error):
            self._handle_threshold_exceeded(error)
        
        # Return user-friendly response
        return self._create_user_response(error, error_id)
    
    def _update_error_counts(self, error: Union[Exception, BaseAppException]):
        """Update error count tracking"""
        if isinstance(error, BaseAppException):
            key = f"{error.category.value}_{error.severity.value}"
        else:
            key = f"unknown_{ErrorSeverity.MEDIUM.value}"
        
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def _check_threshold_exceeded(self, error: Union[Exception, BaseAppException]) -> bool:
        """Check if error threshold is exceeded"""
        if isinstance(error, BaseAppException):
            threshold = self.error_thresholds[error.severity]
            key = f"{error.category.value}_{error.severity.value}"
            count = self.error_counts.get(key, 0)
            return count >= threshold
        return False
    
    def _handle_threshold_exceeded(self, error: Union[Exception, BaseAppException]):
        """Handle threshold exceeded events"""
        if isinstance(error, BaseAppException):
            self.logger.log_critical(
                f"Error threshold exceeded for {error.category.value} - {error.severity.value}",
                error_id=error.error_id
            )
    
    def _create_user_response(self, error: Union[Exception, BaseAppException], error_id: str) -> Dict[str, Any]:
        """Create user-friendly error response"""
        
        if isinstance(error, BaseAppException):
            return {
                "success": False,
                "error": {
                    "id": error_id,
                    "message": error.user_message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp
                }
            }
        else:
            return {
                "success": False,
                "error": {
                    "id": error_id,
                    "message": "An unexpected error occurred. Please try again later.",
                    "category": ErrorCategory.UNKNOWN.value,
                    "severity": ErrorSeverity.MEDIUM.value,
                    "timestamp": datetime.now().isoformat()
                }
            }


# ============================================================================
# Error Context Manager
# ============================================================================

class ErrorContextManager:
    """Context manager for error handling with automatic logging"""
    
    def __init__(self, error_handler: ErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.user_id = None
        self.session_id = None
        self.request_id = None
        self.context = {}
    
    def set_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """Set context for error handling"""
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.context.update(kwargs)
        return self
    
    @contextmanager
    def error_context(self) -> Any:
        """Context manager for automatic error handling"""
        try:
            yield
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                user_id=self.user_id,
                session_id=self.session_id,
                request_id=self.request_id,
                context=self.context
            )
            raise


# ============================================================================
# Error Decorators
# ============================================================================

def handle_errors(error_handler: ErrorHandler):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(error=e)
                raise
        return wrapper
    return decorator


def log_errors(logger: ErrorLogger):
    """Decorator for automatic error logging"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log_error(error=e)
                raise
        return wrapper
    return decorator


# ============================================================================
# User-Friendly Error Messages
# ============================================================================

class UserFriendlyMessages:
    """Collection of user-friendly error messages"""
    
    # Validation errors
    VALIDATION_MESSAGES = {
        "required_field": "This field is required",
        "invalid_email": "Please enter a valid email address",
        "password_too_short": "Password must be at least 8 characters long",
        "password_too_weak": "Password must contain at least one uppercase letter, one lowercase letter, and one number",
        "invalid_phone": "Please enter a valid phone number",
        "invalid_date": "Please enter a valid date",
        "file_too_large": "File size exceeds the maximum limit",
        "invalid_file_type": "File type not supported",
        "invalid_url": "Please enter a valid URL"
    }
    
    # Authentication errors
    AUTHENTICATION_MESSAGES = {
        "invalid_credentials": "Invalid email or password",
        "account_locked": "Your account has been temporarily locked due to multiple failed login attempts",
        "session_expired": "Your session has expired. Please log in again",
        "token_invalid": "Your authentication token is invalid or expired",
        "account_disabled": "Your account has been disabled. Please contact support"
    }
    
    # Authorization errors
    AUTHORIZATION_MESSAGES = {
        "insufficient_permissions": "You don't have permission to perform this action",
        "admin_required": "This action requires administrator privileges",
        "resource_not_found": "The requested resource was not found",
        "access_denied": "Access denied to this resource"
    }
    
    # Database errors
    DATABASE_MESSAGES = {
        "connection_failed": "Unable to connect to the database. Please try again later",
        "query_failed": "Database operation failed. Please try again",
        "duplicate_entry": "This record already exists",
        "constraint_violation": "The operation violates a database constraint",
        "transaction_failed": "Database transaction failed. Please try again"
    }
    
    # Network errors
    NETWORK_MESSAGES = {
        "connection_timeout": "Connection timed out. Please check your internet connection",
        "server_unavailable": "The server is temporarily unavailable. Please try again later",
        "request_timeout": "Request timed out. Please try again",
        "network_error": "Network error occurred. Please check your connection"
    }
    
    # External API errors
    EXTERNAL_API_MESSAGES = {
        "service_unavailable": "External service is temporarily unavailable",
        "rate_limit_exceeded": "Too many requests. Please try again later",
        "invalid_response": "Received invalid response from external service",
        "authentication_failed": "External service authentication failed"
    }
    
    # System errors
    SYSTEM_MESSAGES = {
        "internal_error": "An internal error occurred. Please try again later",
        "configuration_error": "System configuration error. Please contact support",
        "maintenance_mode": "System is under maintenance. Please try again later",
        "resource_exhausted": "System resources are temporarily unavailable"
    }
    
    @classmethod
    def get_message(cls, category: str, error_type: str, default: str = None) -> str:
        """Get user-friendly message by category and type"""
        category_messages = getattr(cls, f"{category.upper()}_MESSAGES", {})
        return category_messages.get(error_type, default or "An unexpected error occurred")


# ============================================================================
# Error Reporting and Monitoring
# ============================================================================

class ErrorReporter:
    """Error reporting and monitoring system"""
    
    def __init__(self, logger: ErrorLogger):
        
    """__init__ function."""
self.logger = logger
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recent_errors": []
        }
    
    def report_error(self, error: Union[Exception, BaseAppException]):
        """Report an error for monitoring"""
        self.error_stats["total_errors"] += 1
        
        if isinstance(error, BaseAppException):
            category = error.category.value
            severity = error.severity.value
            
            # Update category stats
            self.error_stats["errors_by_category"][category] = \
                self.error_stats["errors_by_category"].get(category, 0) + 1
            
            # Update severity stats
            self.error_stats["errors_by_severity"][severity] = \
                self.error_stats["errors_by_severity"].get(severity, 0) + 1
            
            # Add to recent errors
            error_info = {
                "id": error.error_id,
                "category": category,
                "severity": severity,
                "message": error.message,
                "timestamp": error.timestamp
            }
        else:
            error_info = {
                "id": str(uuid.uuid4()),
                "category": ErrorCategory.UNKNOWN.value,
                "severity": ErrorSeverity.MEDIUM.value,
                "message": str(error),
                "timestamp": datetime.now().isoformat()
            }
        
        self.error_stats["recent_errors"].append(error_info)
        
        # Keep only last 100 errors
        if len(self.error_stats["recent_errors"]) > 100:
            self.error_stats["recent_errors"] = self.error_stats["recent_errors"][-100:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return self.error_stats.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "top_categories": sorted(
                self.error_stats["errors_by_category"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "severity_distribution": self.error_stats["errors_by_severity"],
            "recent_error_count": len(self.error_stats["recent_errors"])
        }


# ============================================================================
# Example Usage Functions
# ============================================================================

def demonstrate_error_logging():
    """Demonstrate error logging functionality"""
    
    # Initialize error logging system
    logger = ErrorLogger(
        log_file="logs/errors.log",
        log_level="INFO",
        enable_console=True,
        enable_file=True,
        enable_json=True
    )
    
    error_handler = ErrorHandler(logger)
    error_reporter = ErrorReporter(logger)
    
    # Example 1: Validation Error
    try:
        raise ValidationError(
            message="Email validation failed: invalid format",
            user_message="Please enter a valid email address",
            field="email"
        )
    except ValidationError as e:
        response = error_handler.handle_error(e, user_id="user123")
        error_reporter.report_error(e)
        print(f"Validation Error Response: {response}")
    
    # Example 2: Database Error
    try:
        raise DatabaseError(
            message="Connection to database failed: timeout",
            user_message="Database connection failed. Please try again later."
        )
    except DatabaseError as e:
        response = error_handler.handle_error(e, user_id="user123", session_id="session456")
        error_reporter.report_error(e)
        print(f"Database Error Response: {response}")
    
    # Example 3: External API Error
    try:
        raise ExternalAPIError(
            message="Payment gateway API returned 503 error",
            user_message="Payment service is temporarily unavailable. Please try again in a few minutes."
        )
    except ExternalAPIError as e:
        response = error_handler.handle_error(e, user_id="user123", request_id="req789")
        error_reporter.report_error(e)
        print(f"External API Error Response: {response}")
    
    # Example 4: Generic Exception
    try:
        result = 1 / 0
    except Exception as e:
        response = error_handler.handle_error(e, user_id="user123")
        error_reporter.report_error(e)
        print(f"Generic Error Response: {response}")
    
    # Print error statistics
    print(f"\nError Statistics: {error_reporter.get_error_summary()}")


def demonstrate_error_context_manager():
    """Demonstrate error context manager usage"""
    
    logger = ErrorLogger()
    error_handler = ErrorHandler(logger)
    context_manager = ErrorContextManager(error_handler)
    
    # Set context
    context_manager.set_context(
        user_id="user123",
        session_id="session456",
        request_id="req789",
        operation="data_processing"
    )
    
    # Use context manager
    with context_manager.error_context():
        # This will be automatically logged if an error occurs
        result = 1 / 0


@handle_errors(ErrorHandler(ErrorLogger()))
def demonstrate_error_decorator():
    """Demonstrate error decorator usage"""
    # This function will automatically handle errors
    result = 1 / 0
    return result


def demonstrate_user_friendly_messages():
    """Demonstrate user-friendly error messages"""
    
    # Get messages by category and type
    validation_msg = UserFriendlyMessages.get_message("validation", "invalid_email")
    auth_msg = UserFriendlyMessages.get_message("authentication", "invalid_credentials")
    db_msg = UserFriendlyMessages.get_message("database", "connection_failed")
    
    print(f"Validation Message: {validation_msg}")
    print(f"Authentication Message: {auth_msg}")
    print(f"Database Message: {db_msg}")


if __name__ == "__main__":
    # Run demonstrations
    print("Error Logging and User-Friendly Error Messages Demonstrations")
    print("=" * 80)
    
    demonstrate_error_logging()
    demonstrate_user_friendly_messages()
    
    # Note: These will raise exceptions for demonstration
    # demonstrate_error_context_manager()
    # demonstrate_error_decorator() 