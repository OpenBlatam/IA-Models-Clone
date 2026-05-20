from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from functools import wraps
from contextlib import asynccontextmanager
from .exceptions import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Error Handling and Validation Module

Comprehensive error handling and validation system for the AI Video system
with early returns, proper error categorization, and robust validation.
"""


# Import existing exceptions
    AIVideoError, ConfigurationError, ValidationError, WorkflowError,
    SecurityError, PerformanceError, ResourceError, DependencyError
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    operation: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "additional_data": self.additional_data
        }


@dataclass
class ValidationResult:
    """Result of validation operation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_errors: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_error(self, error: str, field: Optional[str] = None) -> None:
        """Add validation error."""
        self.is_valid = False
        self.errors.append(error)
        if field:
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "field_errors": self.field_errors
        }


class ErrorHandler:
    """Centralized error handling with early returns and proper categorization."""
    
    def __init__(self) -> Any:
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_callbacks: List[Callable] = []
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = strategy
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback for monitoring."""
        self.error_callbacks.append(callback)
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> AIVideoError:
        """Handle error with early returns and proper categorization."""
        # Early return for already handled errors
        if isinstance(error, AIVideoError):
            self._log_error(error, context)
            self._notify_callbacks(error, context)
            return error
        
        # Early return for system errors
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
        
        # Convert to AIVideoError
        ai_error = self._convert_to_ai_error(error, context)
        self._log_error(ai_error, context)
        self._notify_callbacks(ai_error, context)
        
        return ai_error
    
    def _convert_to_ai_error(self, error: Exception, context: Optional[ErrorContext] = None) -> AIVideoError:
        """Convert exception to AIVideoError with proper categorization."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Early returns for specific error types
        if isinstance(error, ValueError):
            return ValidationError(error_message, context=context)
        
        if isinstance(error, KeyError):
            return ConfigurationError(f"Missing configuration key: {error_message}", context=context)
        
        if isinstance(error, FileNotFoundError):
            return ConfigurationError(f"File not found: {error_message}", context=context)
        
        if isinstance(error, PermissionError):
            return SecurityError(f"Permission denied: {error_message}", context=context)
        
        if isinstance(error, MemoryError):
            return ResourceError(f"Memory error: {error_message}", context=context)
        
        if isinstance(error, TimeoutError):
            return PerformanceError(f"Operation timed out: {error_message}", context=context)
        
        # Default case - generic error
        return AIVideoError(error_message, error_code="UNKNOWN_ERROR", context=context)
    
    def _log_error(self, error: AIVideoError, context: Optional[ErrorContext] = None) -> None:
        """Log error with context."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        log_data = {
            "error_type": error_type,
            "message": error.message,
            "error_code": error.error_code,
            "details": error.details,
            "count": self.error_counts[error_type]
        }
        
        if context:
            log_data["context"] = context.to_dict()
        
        logger.error(f"Error occurred: {log_data}")
    
    def _notify_callbacks(self, error: AIVideoError, context: Optional[ErrorContext] = None) -> None:
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error, context)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def handle_async_error(self, coro: Callable, *args, **kwargs) -> Tuple[Any, Optional[AIVideoError]]:
        """Handle async operation with error handling."""
        try:
            result = await coro(*args, **kwargs)
            return result, None
        except Exception as e:
            error = self.handle_error(e)
            return None, error
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "recovery_strategies": list(self.recovery_strategies.keys())
        }


class Validator:
    """Comprehensive validation with early returns and field-specific validation."""
    
    def __init__(self) -> Any:
        self.validators: Dict[str, Callable] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_validator(self, field_type: str, validator: Callable) -> None:
        """Register validator for field type."""
        self.validators[field_type] = validator
    
    def register_custom_validator(self, field_name: str, validator: Callable) -> None:
        """Register custom validator for specific field."""
        self.custom_validators[field_name] = validator
    
    def validate_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema with early returns."""
        result = ValidationResult(is_valid=True)
        
        # Early return for empty data
        if not data:
            result.add_error("Data cannot be empty")
            return result
        
        # Early return for invalid schema
        if not schema:
            result.add_error("Schema cannot be empty")
            return result
        
        # Validate each field
        for field_name, field_config in schema.items():
            field_value = data.get(field_name)
            
            # Skip optional fields that are None
            if field_value is None and field_config.get("required", False):
                result.add_error(f"Required field '{field_name}' is missing", field_name)
                continue
            
            if field_value is None:
                continue
            
            # Validate field
            field_result = self._validate_field(field_name, field_value, field_config)
            if not field_result.is_valid:
                result.errors.extend(field_result.errors)
                result.field_errors[field_name] = field_result.errors
        
        # Update overall validity
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def _validate_field(self, field_name: str, value: Any, config: Dict[str, Any]) -> ValidationResult:
        """Validate individual field with early returns."""
        result = ValidationResult(is_valid=True)
        
        # Early return for custom validator
        if field_name in self.custom_validators:
            try:
                is_valid = self.custom_validators[field_name](value)
                if not is_valid:
                    result.add_error(f"Custom validation failed for field '{field_name}'")
                return result
            except Exception as e:
                result.add_error(f"Custom validation error for field '{field_name}': {e}")
                return result
        
        # Get field type
        field_type = config.get("type", "string")
        
        # Early return for unknown type
        if field_type not in self.validators:
            result.add_error(f"Unknown field type '{field_type}' for field '{field_name}'")
            return result
        
        # Validate using registered validator
        try:
            is_valid = self.validators[field_type](value, config)
            if not is_valid:
                result.add_error(f"Validation failed for field '{field_name}'")
        except Exception as e:
            result.add_error(f"Validation error for field '{field_name}': {e}")
        
        return result
    
    def validate_string(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate string field."""
        if not isinstance(value, str):
            return False
        
        min_length = config.get("min_length", 0)
        max_length = config.get("max_length")
        
        if len(value) < min_length:
            return False
        
        if max_length and len(value) > max_length:
            return False
        
        return True
    
    def validate_integer(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate integer field."""
        if not isinstance(value, int):
            return False
        
        min_value = config.get("min_value")
        max_value = config.get("max_value")
        
        if min_value is not None and value < min_value:
            return False
        
        if max_value is not None and value > max_value:
            return False
        
        return True
    
    def validate_float(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate float field."""
        if not isinstance(value, (int, float)):
            return False
        
        min_value = config.get("min_value")
        max_value = config.get("max_value")
        
        if min_value is not None and value < min_value:
            return False
        
        if max_value is not None and value > max_value:
            return False
        
        return True
    
    def validate_boolean(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate boolean field."""
        return isinstance(value, bool)
    
    def validate_list(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate list field."""
        if not isinstance(value, list):
            return False
        
        min_length = config.get("min_length", 0)
        max_length = config.get("max_length")
        
        if len(value) < min_length:
            return False
        
        if max_length and len(value) > max_length:
            return False
        
        return True
    
    def validate_dict(self, value: Any, config: Dict[str, Any]) -> bool:
        """Validate dictionary field."""
        if not isinstance(value, dict):
            return False
        
        required_keys = config.get("required_keys", [])
        for key in required_keys:
            if key not in value:
                return False
        
        return True


class AsyncErrorHandler:
    """Async error handling with proper resource management."""
    
    def __init__(self, error_handler: ErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.active_operations: Dict[str, asyncio.Task] = {}
    
    @asynccontextmanager
    async def operation_context(self, operation_id: str, context: Optional[ErrorContext] = None):
        """Context manager for async operations with error handling."""
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            error = self.error_handler.handle_error(e, context)
            logger.error(f"Operation '{operation_id}' failed after {time.time() - start_time:.2f}s: {error}")
            raise error
        finally:
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
    
    async def run_with_retry(self, coro: Callable, max_retries: int = 3, delay: float = 1.0, 
                           context: Optional[ErrorContext] = None, *args, **kwargs) -> Any:
        """Run coroutine with retry logic and early returns."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await coro(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                
                # Early return for non-retryable errors
                if isinstance(e, (ValidationError, SecurityError, ConfigurationError)):
                    raise self.error_handler.handle_error(e, context)
                
                # Early return for last attempt
                if attempt == max_retries - 1:
                    raise self.error_handler.handle_error(e, context)
                
                # Wait before retry
                await asyncio.sleep(delay * (2 ** attempt))
        
        # This should never be reached, but just in case
        raise self.error_handler.handle_error(last_error, context)
    
    async def run_with_timeout(self, coro: Callable, timeout: float, 
                             context: Optional[ErrorContext] = None, *args, **kwargs) -> Any:
        """Run coroutine with timeout and early returns."""
        try:
            result = await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            error = PerformanceError(f"Operation timed out after {timeout}s", context=context)
            raise self.error_handler.handle_error(error, context)
        except Exception as e:
            raise self.error_handler.handle_error(e, context)


# Global instances
error_handler = ErrorHandler()
validator = Validator()
async_error_handler = AsyncErrorHandler(error_handler)

# Register default validators
validator.register_validator("string", validator.validate_string)
validator.register_validator("integer", validator.validate_integer)
validator.register_validator("float", validator.validate_float)
validator.register_validator("boolean", validator.validate_boolean)
validator.register_validator("list", validator.validate_list)
validator.register_validator("dict", validator.validate_dict)


# Decorators for error handling
def handle_errors(context_operation: Optional[str] = None):
    """Decorator for error handling with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            context = ErrorContext(operation=context_operation or func.__name__)
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error = error_handler.handle_error(e, context)
                raise error
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            context = ErrorContext(operation=context_operation or func.__name__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = error_handler.handle_error(e, context)
                raise error
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def validate_input(schema: Dict[str, Any]):
    """Decorator for input validation with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Validate kwargs
            validation_result = validator.validate_data(kwargs, schema)
            if not validation_result.is_valid:
                raise ValidationError(f"Input validation failed: {validation_result.errors}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Validate kwargs
            validation_result = validator.validate_data(kwargs, schema)
            if not validation_result.is_valid:
                raise ValidationError(f"Input validation failed: {validation_result.errors}")
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Utility functions with early returns
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> ValidationResult:
    """Validate required fields with early returns."""
    result = ValidationResult(is_valid=True)
    
    # Early return for empty required fields
    if not required_fields:
        return result
    
    # Early return for empty data
    if not data:
        result.add_error("Data cannot be empty")
        return result
    
    # Check each required field
    for field in required_fields:
        if field not in data or data[field] is None:
            result.add_error(f"Required field '{field}' is missing", field)
    
    result.is_valid = len(result.errors) == 0
    return result


def validate_field_types(data: Dict[str, Any], field_types: Dict[str, Type]) -> ValidationResult:
    """Validate field types with early returns."""
    result = ValidationResult(is_valid=True)
    
    # Early return for empty field types
    if not field_types:
        return result
    
    # Early return for empty data
    if not data:
        result.add_error("Data cannot be empty")
        return result
    
    # Check each field type
    for field, expected_type in field_types.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                result.add_error(f"Field '{field}' must be of type {expected_type.__name__}", field)
    
    result.is_valid = len(result.errors) == 0
    return result


async def safe_execute(coro: Callable, *args, **kwargs) -> Tuple[Any, Optional[AIVideoError]]:
    """Safely execute coroutine with error handling."""
    try:
        result = await coro(*args, **kwargs)
        return result, None
    except Exception as e:
        error = error_handler.handle_error(e)
        return None, error


def is_recoverable_error(error: AIVideoError) -> bool:
    """Check if error is recoverable with early returns."""
    # Early return for security errors
    if isinstance(error, SecurityError):
        return False
    
    # Early return for validation errors
    if isinstance(error, ValidationError):
        return False
    
    # Early return for configuration errors
    if isinstance(error, ConfigurationError):
        return False
    
    # All other errors are potentially recoverable
    return True


def should_retry_error(error: AIVideoError, retry_count: int, max_retries: int = 3) -> bool:
    """Check if error should be retried with early returns."""
    # Early return for max retries reached
    if retry_count >= max_retries:
        return False
    
    # Early return for non-recoverable errors
    if not is_recoverable_error(error):
        return False
    
    # Retryable error types
    retryable_types = (WorkflowError, PerformanceError, ResourceError)
    return isinstance(error, retryable_types)


# Export main components
__all__ = [
    "ErrorHandler",
    "Validator", 
    "AsyncErrorHandler",
    "ErrorContext",
    "ValidationResult",
    "error_handler",
    "validator",
    "async_error_handler",
    "handle_errors",
    "validate_input",
    "validate_required_fields",
    "validate_field_types",
    "safe_execute",
    "is_recoverable_error",
    "should_retry_error"
] 