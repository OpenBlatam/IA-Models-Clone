from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Dict, Any
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Exception handling for OS Content UGC Video Generator
Centralized exception classes and error handling
"""


logger = structlog.get_logger("os_content.exceptions")

class OSContentException(Exception):
    """Base exception for OS Content system"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        
        # Log the exception
        logger.error(
            f"OSContentException: {message}",
            error_code=self.error_code,
            details=self.details
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }

class ValidationError(OSContentException):
    """Exception for validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        
    """__init__ function."""
super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})

class ProcessingError(OSContentException):
    """Exception for processing errors"""
    
    def __init__(self, message: str, stage: Optional[str] = None, retryable: bool = False):
        
    """__init__ function."""
super().__init__(message, "PROCESSING_ERROR", {"stage": stage, "retryable": retryable})

class CacheError(OSContentException):
    """Exception for cache errors"""
    
    def __init__(self, message: str, cache_level: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "CACHE_ERROR", {"cache_level": cache_level})

class FileError(OSContentException):
    """Exception for file operations"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "FILE_ERROR", {"file_path": file_path, "operation": operation})

class NetworkError(OSContentException):
    """Exception for network operations"""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        
    """__init__ function."""
super().__init__(message, "NETWORK_ERROR", {"url": url, "status_code": status_code})

class ConfigurationError(OSContentException):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "CONFIGURATION_ERROR", {"config_key": config_key})

class ResourceError(OSContentException):
    """Exception for resource exhaustion"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, current_usage: Optional[float] = None):
        
    """__init__ function."""
super().__init__(message, "RESOURCE_ERROR", {"resource_type": resource_type, "current_usage": current_usage})

class TimeoutError(OSContentException):
    """Exception for timeout errors"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, operation: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "TIMEOUT_ERROR", {"timeout_duration": timeout_duration, "operation": operation})

class SecurityError(OSContentException):
    """Exception for security violations"""
    
    def __init__(self, message: str, violation_type: Optional[str] = None, client_ip: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "SECURITY_ERROR", {"violation_type": violation_type, "client_ip": client_ip})

class DatabaseError(OSContentException):
    """Exception for database operations"""
    
    def __init__(self, message: str, operation: Optional[str] = None, table: Optional[str] = None):
        
    """__init__ function."""
super().__init__(message, "DATABASE_ERROR", {"operation": operation, "table": table})

def handle_exception(func) -> Any:
    """Decorator for exception handling"""
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except OSContentException:
            # Re-raise OSContent exceptions as they're already handled
            raise
        except Exception as e:
            # Convert other exceptions to OSContentException
            logger.error(f"Unhandled exception in {func.__name__}: {str(e)}")
            raise OSContentException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {"function": func.__name__, "exception_type": type(e).__name__}
            )
    return wrapper

def handle_async_exception(func) -> Any:
    """Decorator for async exception handling"""
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except OSContentException:
            # Re-raise OSContent exceptions as they're already handled
            raise
        except Exception as e:
            # Convert other exceptions to OSContentException
            logger.error(f"Unhandled async exception in {func.__name__}: {str(e)}")
            raise OSContentException(
                f"Unexpected async error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {"function": func.__name__, "exception_type": type(e).__name__}
            )
    return wrapper

def create_error_response(exception: OSContentException) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": exception.to_dict(),
        "timestamp": None  # Will be set by the API layer
    }

def is_retryable_error(exception: OSContentException) -> bool:
    """Check if an error is retryable"""
    if isinstance(exception, ProcessingError):
        return exception.details.get("retryable", False)
    elif isinstance(exception, (NetworkError, TimeoutError)):
        return True
    elif isinstance(exception, (ResourceError, DatabaseError)):
        return False  # Resource and database errors are not retryable
    else:
        return False 