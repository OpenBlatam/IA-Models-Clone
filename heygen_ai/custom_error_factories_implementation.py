from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import logging
import traceback
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
import functools
from error_logging_implementation import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Custom Error Types and Error Factories Implementation
===================================================

This module demonstrates:
- Custom error types with specialized behavior
- Error factories for consistent error creation
- Error builders with fluent interfaces
- Domain-specific error types
- Error factories with configuration
- Error type hierarchies and inheritance
"""


# Import from existing error logging implementation
    BaseAppException, ErrorSeverity, ErrorCategory, ErrorContext,
    ValidationError, AuthenticationError, AuthorizationError,
    DatabaseError, ExternalAPIError, ConfigurationError
)


# ============================================================================
# Custom Error Types
# ============================================================================

class MLTrainingError(BaseAppException):
    """Machine Learning training specific error"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        model_name: str,
        training_step: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.MODEL_TRAINING,
            **kwargs
        )
        self.model_name = model_name
        self.training_step = training_step
        self.epoch = epoch
        self.batch = batch
        self.loss = loss
    
    def get_training_context(self) -> Dict[str, Any]:
        """Get training-specific context"""
        return {
            "model_name": self.model_name,
            "training_step": self.training_step,
            "epoch": self.epoch,
            "batch": self.batch,
            "loss": self.loss,
            "error_id": self.error_id,
            "timestamp": self.timestamp
        }


class DataProcessingError(BaseAppException):
    """Data processing specific error"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        data_source: str,
        processing_step: str,
        record_count: Optional[int] = None,
        failed_records: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.DATA_PROCESSING,
            **kwargs
        )
        self.data_source = data_source
        self.processing_step = processing_step
        self.record_count = record_count
        self.failed_records = failed_records or []
    
    def add_failed_record(self, record: Dict[str, Any], reason: str):
        """Add a failed record to the error context"""
        self.failed_records.append({
            "record": record,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            "data_source": self.data_source,
            "processing_step": self.processing_step,
            "total_records": self.record_count,
            "failed_records": len(self.failed_records),
            "success_rate": ((self.record_count - len(self.failed_records)) / self.record_count * 100) if self.record_count else 0
        }


class APIError(BaseAppException):
    """API-specific error with HTTP status codes"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        status_code: int,
        endpoint: str,
        method: str,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=self._get_severity_from_status(status_code),
            category=ErrorCategory.EXTERNAL_API,
            context=ErrorContext.API_CALL,
            **kwargs
        )
        self.status_code = status_code
        self.endpoint = endpoint
        self.method = method
        self.request_data = request_data or {}
        self.response_data = response_data or {}
    
    def _get_severity_from_status(self, status_code: int) -> ErrorSeverity:
        """Map HTTP status code to error severity"""
        if status_code >= 500:
            return ErrorSeverity.HIGH
        elif status_code >= 400:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def get_api_context(self) -> Dict[str, Any]:
        """Get API-specific context"""
        return {
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "method": self.method,
            "request_data": self.request_data,
            "response_data": self.response_data
        }


class FileOperationError(BaseAppException):
    """File operation specific error"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        file_path: str,
        operation: str,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            context=ErrorContext.FILE_OPERATION,
            **kwargs
        )
        self.file_path = file_path
        self.operation = operation
        self.file_size = file_size
        self.file_type = file_type
    
    def get_file_context(self) -> Dict[str, Any]:
        """Get file-specific context"""
        return {
            "file_path": self.file_path,
            "operation": self.operation,
            "file_size": self.file_size,
            "file_type": self.file_type
        }


class ConfigurationError(BaseAppException):
    """Configuration specific error with environment context"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        config_key: str,
        environment: str,
        config_source: str,
        required_value: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=ErrorContext.CONFIGURATION_LOADING,
            **kwargs
        )
        self.config_key = config_key
        self.environment = environment
        self.config_source = config_source
        self.required_value = required_value
    
    def get_config_context(self) -> Dict[str, Any]:
        """Get configuration-specific context"""
        return {
            "config_key": self.config_key,
            "environment": self.environment,
            "config_source": self.config_source,
            "required_value": self.required_value
        }


# ============================================================================
# Error Factory Base Classes
# ============================================================================

class ErrorFactory(ABC):
    """Abstract base class for error factories"""
    
    @abstractmethod
    def create_error(self, **kwargs) -> BaseAppException:
        """Create an error instance"""
        pass
    
    @abstractmethod
    def get_error_type(self) -> Type[BaseAppException]:
        """Get the error type this factory creates"""
        pass


class ErrorBuilder:
    """Fluent interface for building errors"""
    
    def __init__(self, error_type: Type[BaseAppException]):
        
    """__init__ function."""
self.error_type = error_type
        self.message = ""
        self.user_message = ""
        self.severity = ErrorSeverity.MEDIUM
        self.category = ErrorCategory.UNKNOWN
        self.context = ErrorContext.USER_INPUT
        self.additional_data = {}
    
    def with_message(self, message: str) -> 'ErrorBuilder':
        """Set the technical message"""
        self.message = message
        return self
    
    def with_user_message(self, user_message: str) -> 'ErrorBuilder':
        """Set the user-friendly message"""
        self.user_message = user_message
        return self
    
    def with_severity(self, severity: ErrorSeverity) -> 'ErrorBuilder':
        """Set the error severity"""
        self.severity = severity
        return self
    
    def with_category(self, category: ErrorCategory) -> 'ErrorBuilder':
        """Set the error category"""
        self.category = category
        return self
    
    def with_context(self, context: ErrorContext) -> 'ErrorBuilder':
        """Set the error context"""
        self.context = context
        return self
    
    def with_data(self, key: str, value: Any) -> 'ErrorBuilder':
        """Add additional data"""
        self.additional_data[key] = value
        return self
    
    def build(self) -> BaseAppException:
        """Build and return the error"""
        return self.error_type(
            message=self.message,
            user_message=self.user_message,
            severity=self.severity,
            category=self.category,
            context=self.context,
            **self.additional_data
        )


# ============================================================================
# Specific Error Factories
# ============================================================================

class ValidationErrorFactory(ErrorFactory):
    """Factory for creating validation errors"""
    
    def __init__(self) -> Any:
        self.error_type = ValidationError
    
    def create_error(self, **kwargs) -> ValidationError:
        """Create a validation error"""
        return ValidationError(**kwargs)
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def invalid_email(self, email: str, field: str = "email") -> ValidationError:
        """Create invalid email error"""
        return ValidationError(
            message=f"Invalid email format: {email}",
            user_message="Please enter a valid email address",
            field=field
        )
    
    def required_field(self, field: str) -> ValidationError:
        """Create required field error"""
        return ValidationError(
            message=f"Required field missing: {field}",
            user_message=f"The {field} field is required",
            field=field
        )
    
    def field_too_short(self, field: str, min_length: int, actual_length: int) -> ValidationError:
        """Create field too short error"""
        return ValidationError(
            message=f"Field {field} too short: {actual_length} < {min_length}",
            user_message=f"The {field} must be at least {min_length} characters long",
            field=field,
            additional_data={"min_length": min_length, "actual_length": actual_length}
        )
    
    def invalid_format(self, field: str, expected_format: str, actual_value: str) -> ValidationError:
        """Create invalid format error"""
        return ValidationError(
            message=f"Invalid format for {field}: {actual_value}, expected: {expected_format}",
            user_message=f"The {field} format is invalid. Expected: {expected_format}",
            field=field,
            additional_data={"expected_format": expected_format, "actual_value": actual_value}
        )


class MLTrainingErrorFactory(ErrorFactory):
    """Factory for creating ML training errors"""
    
    def __init__(self) -> Any:
        self.error_type = MLTrainingError
    
    def create_error(self, **kwargs) -> MLTrainingError:
        """Create an ML training error"""
        return MLTrainingError(**kwargs)
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def gpu_memory_exceeded(self, model_name: str, required_memory: str, available_memory: str) -> MLTrainingError:
        """Create GPU memory exceeded error"""
        return MLTrainingError(
            message=f"GPU memory exceeded for model {model_name}: required {required_memory}, available {available_memory}",
            user_message="Model training failed due to insufficient GPU memory. Please try with a smaller model or batch size.",
            model_name=model_name,
            training_step="memory_allocation",
            additional_data={"required_memory": required_memory, "available_memory": available_memory}
        )
    
    def convergence_failed(self, model_name: str, epochs: int, final_loss: float) -> MLTrainingError:
        """Create convergence failed error"""
        return MLTrainingError(
            message=f"Model {model_name} failed to converge after {epochs} epochs, final loss: {final_loss}",
            user_message="Model training failed to converge. Please check your data and hyperparameters.",
            model_name=model_name,
            training_step="convergence",
            epoch=epochs,
            loss=final_loss
        )
    
    def data_loading_failed(self, model_name: str, data_path: str, reason: str) -> MLTrainingError:
        """Create data loading failed error"""
        return MLTrainingError(
            message=f"Failed to load training data for model {model_name} from {data_path}: {reason}",
            user_message="Failed to load training data. Please check the data path and format.",
            model_name=model_name,
            training_step="data_loading",
            additional_data={"data_path": data_path, "reason": reason}
        )


class DataProcessingErrorFactory(ErrorFactory):
    """Factory for creating data processing errors"""
    
    def __init__(self) -> Any:
        self.error_type = DataProcessingError
    
    def create_error(self, **kwargs) -> DataProcessingError:
        """Create a data processing error"""
        return DataProcessingError(**kwargs)
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def file_not_found(self, data_source: str, file_path: str) -> DataProcessingError:
        """Create file not found error"""
        return DataProcessingError(
            message=f"Data file not found: {file_path}",
            user_message="The specified data file could not be found. Please check the file path.",
            data_source=data_source,
            processing_step="file_loading",
            additional_data={"file_path": file_path}
        )
    
    def invalid_format(self, data_source: str, expected_format: str, actual_format: str) -> DataProcessingError:
        """Create invalid format error"""
        return DataProcessingError(
            message=f"Invalid data format: expected {expected_format}, got {actual_format}",
            user_message="The data format is not supported. Please check the file format.",
            data_source=data_source,
            processing_step="format_validation",
            additional_data={"expected_format": expected_format, "actual_format": actual_format}
        )
    
    def corrupted_data(self, data_source: str, record_count: int, corrupted_count: int) -> DataProcessingError:
        """Create corrupted data error"""
        return DataProcessingError(
            message=f"Data corruption detected: {corrupted_count}/{record_count} records corrupted",
            user_message="Some data records appear to be corrupted. Please check your data source.",
            data_source=data_source,
            processing_step="data_validation",
            record_count=record_count,
            additional_data={"corrupted_count": corrupted_count}
        )


class APIErrorFactory(ErrorFactory):
    """Factory for creating API errors"""
    
    def __init__(self) -> Any:
        self.error_type = APIError
    
    def create_error(self, **kwargs) -> APIError:
        """Create an API error"""
        return APIError(**kwargs)
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def timeout(self, endpoint: str, method: str, timeout_seconds: int) -> APIError:
        """Create timeout error"""
        return APIError(
            message=f"API timeout: {method} {endpoint} timed out after {timeout_seconds}s",
            user_message="The request timed out. Please try again later.",
            status_code=408,
            endpoint=endpoint,
            method=method,
            additional_data={"timeout_seconds": timeout_seconds}
        )
    
    def rate_limited(self, endpoint: str, method: str, retry_after: int) -> APIError:
        """Create rate limit error"""
        return APIError(
            message=f"Rate limited: {method} {endpoint} rate limit exceeded",
            user_message="Too many requests. Please try again later.",
            status_code=429,
            endpoint=endpoint,
            method=method,
            additional_data={"retry_after": retry_after}
        )
    
    def server_error(self, endpoint: str, method: str, status_code: int, response_data: Dict[str, Any]) -> APIError:
        """Create server error"""
        return APIError(
            message=f"Server error: {method} {endpoint} returned {status_code}",
            user_message="The server encountered an error. Please try again later.",
            status_code=status_code,
            endpoint=endpoint,
            method=method,
            response_data=response_data
        )


class FileOperationErrorFactory(ErrorFactory):
    """Factory for creating file operation errors"""
    
    def __init__(self) -> Any:
        self.error_type = FileOperationError
    
    def create_error(self, **kwargs) -> FileOperationError:
        """Create a file operation error"""
        return FileOperationError(**kwargs)
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def file_not_found(self, file_path: str, operation: str) -> FileOperationError:
        """Create file not found error"""
        return FileOperationError(
            message=f"File not found: {file_path}",
            user_message="The specified file could not be found. Please check the file path.",
            file_path=file_path,
            operation=operation
        )
    
    def permission_denied(self, file_path: str, operation: str) -> FileOperationError:
        """Create permission denied error"""
        return FileOperationError(
            message=f"Permission denied: {operation} on {file_path}",
            user_message="You don't have permission to perform this operation on the file.",
            file_path=file_path,
            operation=operation
        )
    
    def disk_full(self, file_path: str, operation: str, required_space: int, available_space: int) -> FileOperationError:
        """Create disk full error"""
        return FileOperationError(
            message=f"Disk full: {operation} on {file_path} requires {required_space} bytes, available {available_space}",
            user_message="Not enough disk space to complete the operation. Please free up some space.",
            file_path=file_path,
            operation=operation,
            additional_data={"required_space": required_space, "available_space": available_space}
        )


# ============================================================================
# Error Factory Registry
# ============================================================================

class ErrorFactoryRegistry:
    """Registry for managing error factories"""
    
    def __init__(self) -> Any:
        self._factories: Dict[str, ErrorFactory] = {}
        self._default_factory: Optional[ErrorFactory] = None
    
    def register_factory(self, name: str, factory: ErrorFactory) -> None:
        """Register an error factory"""
        self._factories[name] = factory
    
    def get_factory(self, name: str) -> Optional[ErrorFactory]:
        """Get an error factory by name"""
        return self._factories.get(name)
    
    def set_default_factory(self, factory: ErrorFactory) -> None:
        """Set the default error factory"""
        self._default_factory = factory
    
    def create_error(self, factory_name: str, **kwargs) -> BaseAppException:
        """Create an error using a specific factory"""
        factory = self.get_factory(factory_name)
        if factory is None:
            if self._default_factory is None:
                raise ValueError(f"No factory found for '{factory_name}' and no default factory set")
            factory = self._default_factory
        
        return factory.create_error(**kwargs)
    
    def list_factories(self) -> List[str]:
        """List all registered factory names"""
        return list(self._factories.keys())


# ============================================================================
# Error Context Manager with Factories
# ============================================================================

class ErrorContextManager:
    """Enhanced error context manager with factory support"""
    
    def __init__(self, registry: ErrorFactoryRegistry):
        
    """__init__ function."""
self.registry = registry
        self.context = {}
    
    def set_context(self, **kwargs) -> 'ErrorContextManager':
        """Set context for error handling"""
        self.context.update(kwargs)
        return self
    
    @contextmanager
    def error_context(self, factory_name: Optional[str] = None):
        """Context manager for automatic error handling with factory support"""
        try:
            yield
        except Exception as e:
            if factory_name and isinstance(e, BaseAppException):
                # Re-raise custom exceptions as-is
                raise
            
            # Create error using factory if specified
            if factory_name:
                factory = self.registry.get_factory(factory_name)
                if factory:
                    # Convert generic exception to factory-specific error
                    error = factory.create_error(
                        message=str(e),
                        user_message="An error occurred during processing",
                        **self.context
                    )
                    raise error
            
            # Re-raise original exception
            raise


# ============================================================================
# Error Decorators with Factories
# ============================================================================

def handle_errors_with_factory(factory_name: str, registry: ErrorFactoryRegistry):
    """Decorator for automatic error handling with factory"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, BaseAppException):
                    raise
                
                # Convert to factory-specific error
                factory = registry.get_factory(factory_name)
                if factory:
                    error = factory.create_error(
                        message=str(e),
                        user_message="An error occurred during processing"
                    )
                    raise error
                raise
        return wrapper
    return decorator


def log_errors_with_factory(factory_name: str, registry: ErrorFactoryRegistry, logger):
    """Decorator for automatic error logging with factory"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, BaseAppException):
                    logger.log_error(e)
                    raise
                
                # Convert to factory-specific error
                factory = registry.get_factory(factory_name)
                if factory:
                    error = factory.create_error(
                        message=str(e),
                        user_message="An error occurred during processing"
                    )
                    logger.log_error(error)
                    raise error
                
                logger.log_error(e)
                raise
        return wrapper
    return decorator


# ============================================================================
# Domain-Specific Error Types
# ============================================================================

class ECommerceError(BaseAppException):
    """E-commerce specific error"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        order_id: Optional[str] = None,
        product_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.USER_INPUT,
            **kwargs
        )
        self.order_id = order_id
        self.product_id = product_id
        self.customer_id = customer_id


class FinancialError(BaseAppException):
    """Financial transaction specific error"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        transaction_id: Optional[str] = None,
        amount: Optional[float] = None,
        currency: Optional[str] = None,
        account_id: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.USER_INPUT,
            **kwargs
        )
        self.transaction_id = transaction_id
        self.amount = amount
        self.currency = currency
        self.account_id = account_id


class HealthcareError(BaseAppException):
    """Healthcare specific error with HIPAA compliance"""
    
    def __init__(
        self,
        message: str,
        user_message: str,
        patient_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        service_type: Optional[str] = None,
        **kwargs
    ):
        
    """__init__ function."""
super().__init__(
            message=message,
            user_message=user_message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=ErrorContext.USER_INPUT,
            **kwargs
        )
        self.patient_id = patient_id
        self.provider_id = provider_id
        self.service_type = service_type
    
    def get_hipaa_context(self) -> Dict[str, Any]:
        """Get HIPAA-compliant context (no PHI)"""
        return {
            "provider_id": self.provider_id,
            "service_type": self.service_type,
            "error_id": self.error_id,
            "timestamp": self.timestamp
        }


# ============================================================================
# Error Factory Configuration
# ============================================================================

@dataclass
class ErrorFactoryConfig:
    """Configuration for error factories"""
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False
    log_level: str = "INFO"
    include_stack_trace: bool = True
    include_context: bool = True
    max_context_size: int = 1024
    error_threshold: int = 100
    alert_on_critical: bool = True


class ConfigurableErrorFactory(ErrorFactory):
    """Error factory with configuration support"""
    
    def __init__(self, error_type: Type[BaseAppException], config: ErrorFactoryConfig):
        
    """__init__ function."""
self.error_type = error_type
        self.config = config
        self.error_count = 0
    
    def create_error(self, **kwargs) -> BaseAppException:
        """Create an error with configuration"""
        # Increment error count
        self.error_count += 1
        
        # Add configuration-based context
        if self.config.include_context:
            kwargs.setdefault('additional_data', {}).update({
                'factory_config': {
                    'enable_logging': self.config.enable_logging,
                    'enable_metrics': self.config.enable_metrics,
                    'error_count': self.error_count
                }
            })
        
        # Create error
        error = self.error_type(**kwargs)
        
        # Check thresholds
        if self.error_count >= self.config.error_threshold:
            self._handle_threshold_exceeded(error)
        
        # Alert on critical errors
        if self.config.alert_on_critical and error.severity == ErrorSeverity.CRITICAL:
            self._alert_critical_error(error)
        
        return error
    
    def get_error_type(self) -> Type[BaseAppException]:
        return self.error_type
    
    def _handle_threshold_exceeded(self, error: BaseAppException):
        """Handle error threshold exceeded"""
        if self.config.enable_logging:
            logging.critical(f"Error threshold exceeded: {self.error_count} errors")
    
    def _alert_critical_error(self, error: BaseAppException):
        """Alert on critical error"""
        if self.config.enable_logging:
            logging.critical(f"Critical error detected: {error.message}")


# ============================================================================
# Example Usage Functions
# ============================================================================

def demonstrate_error_factories():
    """Demonstrate error factory usage"""
    
    # Create factories
    validation_factory = ValidationErrorFactory()
    ml_factory = MLTrainingErrorFactory()
    api_factory = APIErrorFactory()
    
    # Create errors using factories
    print("Creating errors using factories:")
    
    # Validation errors
    email_error = validation_factory.invalid_email("test@", "email")
    print(f"Email error: {email_error.user_message}")
    
    required_error = validation_factory.required_field("username")
    print(f"Required field error: {required_error.user_message}")
    
    # ML training errors
    gpu_error = ml_factory.gpu_memory_exceeded("bert-large", "8GB", "4GB")
    print(f"GPU error: {gpu_error.user_message}")
    
    convergence_error = ml_factory.convergence_failed("cnn-model", 100, 0.85)
    print(f"Convergence error: {gpu_error.user_message}")
    
    # API errors
    timeout_error = api_factory.timeout("/api/users", "GET", 30)
    print(f"Timeout error: {timeout_error.user_message}")
    
    rate_limit_error = api_factory.rate_limited("/api/data", "POST", 60)
    print(f"Rate limit error: {rate_limit_error.user_message}")


def demonstrate_error_builder():
    """Demonstrate error builder usage"""
    
    print("\nCreating errors using builder pattern:")
    
    # Build a custom validation error
    custom_error = (ErrorBuilder(ValidationError)
                   .with_message("Custom validation failed")
                   .with_user_message("Please check your input")
                   .with_severity(ErrorSeverity.HIGH)
                   .with_category(ErrorCategory.VALIDATION)
                   .with_data("field", "custom_field")
                   .with_data("value", "invalid_value")
                   .build())
    
    print(f"Custom error: {custom_error.user_message}")
    print(f"Error data: {custom_error.additional_data}")


def demonstrate_error_registry():
    """Demonstrate error factory registry"""
    
    # Create registry
    registry = ErrorFactoryRegistry()
    
    # Register factories
    registry.register_factory("validation", ValidationErrorFactory())
    registry.register_factory("ml", MLTrainingErrorFactory())
    registry.register_factory("api", APIErrorFactory())
    
    # Set default factory
    registry.set_default_factory(ValidationErrorFactory())
    
    print(f"\nRegistered factories: {registry.list_factories()}")
    
    # Create errors using registry
    validation_error = registry.create_error("validation", 
                                           message="Test validation error",
                                           user_message="Validation failed")
    print(f"Registry validation error: {validation_error.user_message}")
    
    # Create error with unknown factory (uses default)
    default_error = registry.create_error("unknown", 
                                        message="Test default error",
                                        user_message="Default error")
    print(f"Default error: {default_error.user_message}")


def demonstrate_configurable_factory():
    """Demonstrate configurable error factory"""
    
    # Create configuration
    config = ErrorFactoryConfig(
        enable_logging=True,
        enable_metrics=True,
        alert_on_critical=True,
        error_threshold=5
    )
    
    # Create configurable factory
    factory = ConfigurableErrorFactory(ValidationError, config)
    
    print("\nCreating errors with configurable factory:")
    
    # Create multiple errors to test threshold
    for i in range(7):
        error = factory.create_error(
            message=f"Test error {i}",
            user_message=f"Test error {i}"
        )
        print(f"Error {i}: {error.user_message}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_error_factories()
    demonstrate_error_builder()
    demonstrate_error_registry()
    demonstrate_configurable_factory() 