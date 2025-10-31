"""
Error Factories and Custom Error Types for Video-OpusClip

Provides custom error types and factory functions for consistent error handling
across the video processing system.
"""

from typing import Dict, Any, Optional, Union, Type, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import uuid

try:
    from .error_handling import (
        ErrorCode, VideoProcessingError, ValidationError, ProcessingError,
        ExternalServiceError, ResourceError, CriticalSystemError, SecurityError,
        ConfigurationError
    )
except ImportError:
    # Fallback for direct execution
    from error_handling import (
        ErrorCode, VideoProcessingError, ValidationError, ProcessingError,
        ExternalServiceError, ResourceError, CriticalSystemError, SecurityError,
        ConfigurationError
    )

# =============================================================================
# ERROR CATEGORIES
# =============================================================================

class ErrorCategory(Enum):
    """Categories of errors for better organization and handling."""
    
    # Input and Validation
    VALIDATION = "validation"
    INPUT = "input"
    FORMAT = "format"
    
    # Processing
    PROCESSING = "processing"
    ENCODING = "encoding"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    
    # Resources
    RESOURCE = "resource"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    
    # External Services
    EXTERNAL = "external"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    
    # System
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    CRITICAL = "critical"
    
    # AI/ML Specific
    MODEL = "model"
    INFERENCE = "inference"
    TRAINING = "training"
    PIPELINE = "pipeline"

# =============================================================================
# ERROR CONTEXT
# =============================================================================

@dataclass
class ErrorContext:
    """Context information for error handling and debugging."""
    
    # Request information
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Operation information
    operation: Optional[str] = None
    component: Optional[str] = None
    step: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    
    # Timing information
    start_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "operation": self.operation,
            "component": self.component,
            "step": self.step,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_usage": self.resource_usage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration": self.duration,
            "metadata": self.metadata
        }

# =============================================================================
# CUSTOM ERROR TYPES
# =============================================================================

class VideoValidationError(ValidationError):
    """Custom validation error for video processing."""
    
    def __init__(self, message: str, field: str, value: Any = None, 
                 context: Optional[ErrorContext] = None):
        super().__init__(message, field, value)
        self.context = context or ErrorContext()
        self.category = ErrorCategory.VALIDATION

class VideoProcessingError(ProcessingError):
    """Custom processing error for video operations."""
    
    def __init__(self, message: str, operation: str, 
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, operation, details)
        self.context = context or ErrorContext()
        self.category = ErrorCategory.PROCESSING

class VideoEncodingError(ProcessingError):
    """Custom error for video encoding operations."""
    
    def __init__(self, message: str, video_id: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "video_encoding", details)
        self.video_id = video_id
        self.context = context or ErrorContext()
        self.category = ErrorCategory.ENCODING

class VideoExtractionError(ProcessingError):
    """Custom error for video extraction operations."""
    
    def __init__(self, message: str, extraction_type: str,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"video_extraction_{extraction_type}", details)
        self.extraction_type = extraction_type
        self.context = context or ErrorContext()
        self.category = ErrorCategory.EXTRACTION

class VideoAnalysisError(ProcessingError):
    """Custom error for video analysis operations."""
    
    def __init__(self, message: str, analysis_type: str,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"video_analysis_{analysis_type}", details)
        self.analysis_type = analysis_type
        self.context = context or ErrorContext()
        self.category = ErrorCategory.ANALYSIS

class ModelInferenceError(ProcessingError):
    """Custom error for AI model inference operations."""
    
    def __init__(self, message: str, model_name: str,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"model_inference_{model_name}", details)
        self.model_name = model_name
        self.context = context or ErrorContext()
        self.category = ErrorCategory.INFERENCE

class PipelineError(ProcessingError):
    """Custom error for pipeline operations."""
    
    def __init__(self, message: str, pipeline_name: str, stage: str,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"pipeline_{pipeline_name}_{stage}", details)
        self.pipeline_name = pipeline_name
        self.stage = stage
        self.context = context or ErrorContext()
        self.category = ErrorCategory.PIPELINE

class ResourceExhaustionError(ResourceError):
    """Custom error for resource exhaustion."""
    
    def __init__(self, message: str, resource: str, 
                 available: Optional[Any] = None, required: Optional[Any] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, resource, available, required)
        self.context = context or ErrorContext()
        self.category = ErrorCategory.RESOURCE

class MemoryError(ResourceError):
    """Custom error for memory-related issues."""
    
    def __init__(self, message: str, memory_type: str,
                 available: Optional[Any] = None, required: Optional[Any] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, f"memory_{memory_type}", available, required)
        self.memory_type = memory_type
        self.context = context or ErrorContext()
        self.category = ErrorCategory.MEMORY

class StorageError(ResourceError):
    """Custom error for storage-related issues."""
    
    def __init__(self, message: str, storage_type: str,
                 available: Optional[Any] = None, required: Optional[Any] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, f"storage_{storage_type}", available, required)
        self.storage_type = storage_type
        self.context = context or ErrorContext()
        self.category = ErrorCategory.STORAGE

class NetworkError(ResourceError):
    """Custom error for network-related issues."""
    
    def __init__(self, message: str, network_operation: str,
                 context: Optional[ErrorContext] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"network_{network_operation}", None, None)
        self.network_operation = network_operation
        self.context = context or ErrorContext()
        self.details = details or {}
        self.category = ErrorCategory.NETWORK

class APIError(ExternalServiceError):
    """Custom error for API-related issues."""
    
    def __init__(self, message: str, service: str, endpoint: str,
                 status_code: Optional[int] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, service, status_code)
        self.endpoint = endpoint
        self.context = context or ErrorContext()
        self.category = ErrorCategory.API

class DatabaseError(ExternalServiceError):
    """Custom error for database operations."""
    
    def __init__(self, message: str, operation: str, table: Optional[str] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, "database")
        self.operation = operation
        self.table = table
        self.context = context or ErrorContext()
        self.category = ErrorCategory.DATABASE

class CacheError(ExternalServiceError):
    """Custom error for cache operations."""
    
    def __init__(self, message: str, cache_operation: str,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, "cache")
        self.cache_operation = cache_operation
        self.context = context or ErrorContext()
        self.category = ErrorCategory.CACHE

class ConfigurationError(ConfigurationError):
    """Custom error for configuration issues."""
    
    def __init__(self, message: str, config_key: str, config_type: str,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, config_key)
        self.config_type = config_type
        self.context = context or ErrorContext()
        self.category = ErrorCategory.CONFIGURATION

class SecurityViolationError(SecurityError):
    """Custom error for security violations."""
    
    def __init__(self, message: str, threat_type: str, severity: str,
                 context: Optional[ErrorContext] = None):
        super().__init__(message, threat_type)
        self.severity = severity
        self.context = context or ErrorContext()
        self.category = ErrorCategory.SECURITY

# =============================================================================
# ERROR FACTORY
# =============================================================================

class ErrorFactory:
    """Factory for creating consistent error instances."""
    
    def __init__(self):
        self.error_registry: Dict[str, Type[VideoProcessingError]] = {}
        self._register_default_errors()
    
    def _register_default_errors(self):
        """Register default error types."""
        self.register_error("validation", VideoValidationError)
        self.register_error("processing", VideoProcessingError)
        self.register_error("encoding", VideoEncodingError)
        self.register_error("extraction", VideoExtractionError)
        self.register_error("analysis", VideoAnalysisError)
        self.register_error("inference", ModelInferenceError)
        self.register_error("pipeline", PipelineError)
        self.register_error("resource", ResourceExhaustionError)
        self.register_error("memory", MemoryError)
        self.register_error("storage", StorageError)
        self.register_error("network", NetworkError)
        self.register_error("api", APIError)
        self.register_error("database", DatabaseError)
        self.register_error("cache", CacheError)
        self.register_error("configuration", ConfigurationError)
        self.register_error("security", SecurityViolationError)
    
    def register_error(self, error_type: str, error_class: Type[VideoProcessingError]):
        """Register a custom error type."""
        self.error_registry[error_type] = error_class
    
    def create_error(self, error_type: str, message: str, **kwargs) -> VideoProcessingError:
        """Create an error instance of the specified type."""
        if error_type not in self.error_registry:
            raise ValueError(f"Unknown error type: {error_type}")
        
        error_class = self.error_registry[error_type]
        return error_class(message, **kwargs)
    
    def create_validation_error(self, field: str, value: Any, message: str,
                               context: Optional[ErrorContext] = None) -> VideoValidationError:
        """Create a validation error."""
        return VideoValidationError(message, field, value, context)
    
    def create_processing_error(self, operation: str, message: str,
                               context: Optional[ErrorContext] = None,
                               details: Optional[Dict[str, Any]] = None) -> VideoProcessingError:
        """Create a processing error."""
        return VideoProcessingError(message, operation, context, details)
    
    def create_encoding_error(self, message: str, video_id: Optional[str] = None,
                             context: Optional[ErrorContext] = None,
                             details: Optional[Dict[str, Any]] = None) -> VideoEncodingError:
        """Create a video encoding error."""
        return VideoEncodingError(message, video_id, context, details)
    
    def create_extraction_error(self, message: str, extraction_type: str,
                               context: Optional[ErrorContext] = None,
                               details: Optional[Dict[str, Any]] = None) -> VideoExtractionError:
        """Create a video extraction error."""
        return VideoExtractionError(message, extraction_type, context, details)
    
    def create_analysis_error(self, message: str, analysis_type: str,
                             context: Optional[ErrorContext] = None,
                             details: Optional[Dict[str, Any]] = None) -> VideoAnalysisError:
        """Create a video analysis error."""
        return VideoAnalysisError(message, analysis_type, context, details)
    
    def create_inference_error(self, message: str, model_name: str,
                              context: Optional[ErrorContext] = None,
                              details: Optional[Dict[str, Any]] = None) -> ModelInferenceError:
        """Create a model inference error."""
        return ModelInferenceError(message, model_name, context, details)
    
    def create_pipeline_error(self, message: str, pipeline_name: str, stage: str,
                             context: Optional[ErrorContext] = None,
                             details: Optional[Dict[str, Any]] = None) -> PipelineError:
        """Create a pipeline error."""
        return PipelineError(message, pipeline_name, stage, context, details)
    
    def create_resource_error(self, message: str, resource: str,
                             available: Optional[Any] = None, required: Optional[Any] = None,
                             context: Optional[ErrorContext] = None) -> ResourceExhaustionError:
        """Create a resource error."""
        return ResourceExhaustionError(message, resource, available, required, context)
    
    def create_memory_error(self, message: str, memory_type: str,
                           available: Optional[Any] = None, required: Optional[Any] = None,
                           context: Optional[ErrorContext] = None) -> MemoryError:
        """Create a memory error."""
        return MemoryError(message, memory_type, available, required, context)
    
    def create_storage_error(self, message: str, storage_type: str,
                            available: Optional[Any] = None, required: Optional[Any] = None,
                            context: Optional[ErrorContext] = None) -> StorageError:
        """Create a storage error."""
        return StorageError(message, storage_type, available, required, context)
    
    def create_network_error(self, message: str, network_operation: str,
                            context: Optional[ErrorContext] = None,
                            details: Optional[Dict[str, Any]] = None) -> NetworkError:
        """Create a network error."""
        return NetworkError(message, network_operation, context, details)
    
    def create_api_error(self, message: str, service: str, endpoint: str,
                        status_code: Optional[int] = None,
                        context: Optional[ErrorContext] = None) -> APIError:
        """Create an API error."""
        return APIError(message, service, endpoint, status_code, context)
    
    def create_database_error(self, message: str, operation: str, table: Optional[str] = None,
                             context: Optional[ErrorContext] = None) -> DatabaseError:
        """Create a database error."""
        return DatabaseError(message, operation, table, context)
    
    def create_cache_error(self, message: str, cache_operation: str,
                          context: Optional[ErrorContext] = None) -> CacheError:
        """Create a cache error."""
        return CacheError(message, cache_operation, context)
    
    def create_configuration_error(self, message: str, config_key: str, config_type: str,
                                  context: Optional[ErrorContext] = None) -> ConfigurationError:
        """Create a configuration error."""
        return ConfigurationError(message, config_key, config_type, context)
    
    def create_security_error(self, message: str, threat_type: str, severity: str,
                             context: Optional[ErrorContext] = None) -> SecurityViolationError:
        """Create a security error."""
        return SecurityViolationError(message, threat_type, severity, context)

# =============================================================================
# ERROR CONTEXT MANAGER
# =============================================================================

class ErrorContextManager:
    """Manager for error context throughout request lifecycle."""
    
    def __init__(self):
        self.context = ErrorContext()
        self.context_stack = []
    
    def set_request_context(self, request_id: str, user_id: Optional[str] = None,
                           session_id: Optional[str] = None):
        """Set request-level context."""
        self.context.request_id = request_id
        self.context.user_id = user_id
        self.context.session_id = session_id
    
    def set_operation_context(self, operation: str, component: Optional[str] = None,
                             step: Optional[str] = None):
        """Set operation-level context."""
        self.context.operation = operation
        self.context.component = component
        self.context.step = step
    
    def set_resource_context(self, resource_type: str, resource_id: Optional[str] = None,
                            resource_usage: Optional[Dict[str, Any]] = None):
        """Set resource-level context."""
        self.context.resource_type = resource_type
        self.context.resource_id = resource_id
        self.context.resource_usage = resource_usage
    
    def start_timing(self):
        """Start timing for the current operation."""
        self.context.start_time = datetime.utcnow()
    
    def end_timing(self):
        """End timing and calculate duration."""
        if self.context.start_time:
            self.context.duration = (datetime.utcnow() - self.context.start_time).total_seconds()
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the context."""
        self.context.metadata[key] = value
    
    def push_context(self):
        """Push current context to stack."""
        self.context_stack.append(ErrorContext(
            request_id=self.context.request_id,
            user_id=self.context.user_id,
            session_id=self.context.session_id,
            operation=self.context.operation,
            component=self.context.component,
            step=self.context.step,
            resource_type=self.context.resource_type,
            resource_id=self.context.resource_id,
            resource_usage=self.context.resource_usage,
            start_time=self.context.start_time,
            duration=self.context.duration,
            metadata=self.context.metadata.copy()
        ))
    
    def pop_context(self):
        """Pop context from stack."""
        if self.context_stack:
            self.context = self.context_stack.pop()
    
    def get_context(self) -> ErrorContext:
        """Get current context."""
        return self.context
    
    def clear_context(self):
        """Clear current context."""
        self.context = ErrorContext()

# =============================================================================
# ERROR DECORATORS
# =============================================================================

def with_error_context(error_factory: ErrorFactory, context_manager: ErrorContextManager):
    """Decorator to automatically add error context."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                context_manager.start_timing()
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to existing errors
                if hasattr(e, 'context'):
                    e.context = context_manager.get_context()
                context_manager.end_timing()
                raise
        return wrapper
    return decorator

def handle_errors(error_factory: ErrorFactory, context_manager: ErrorContextManager,
                  error_type: str, operation: str):
    """Decorator to handle errors with consistent error types."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                context_manager.set_operation_context(operation)
                context_manager.start_timing()
                return func(*args, **kwargs)
            except Exception as e:
                context_manager.end_timing()
                
                # Create appropriate error type
                if isinstance(e, VideoProcessingError):
                    # Already a custom error, just add context
                    e.context = context_manager.get_context()
                    raise
                else:
                    # Create new error with context
                    error = error_factory.create_error(
                        error_type,
                        str(e),
                        context=context_manager.get_context(),
                        details={"original_error": type(e).__name__, "original_message": str(e)}
                    )
                    raise error
        return wrapper
    return decorator

# =============================================================================
# ERROR UTILITIES
# =============================================================================

def create_error_context(request_id: Optional[str] = None, 
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        operation: Optional[str] = None,
                        component: Optional[str] = None,
                        step: Optional[str] = None,
                        **metadata) -> ErrorContext:
    """Create an error context with the given parameters."""
    context = ErrorContext(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        operation=operation,
        component=component,
        step=step,
        metadata=metadata
    )
    return context

def enrich_error_with_context(error: Exception, context: ErrorContext):
    """Enrich an existing error with context."""
    if hasattr(error, 'context'):
        # Merge contexts
        for key, value in context.to_dict().items():
            if value is not None and not hasattr(error.context, key):
                setattr(error.context, key, value)
    else:
        # Add context attribute
        error.context = context

def get_error_summary(error: Exception) -> Dict[str, Any]:
    """Get a summary of error information."""
    summary = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_category": getattr(error, 'category', None),
        "timestamp": datetime.utcnow().isoformat(),
        "traceback": traceback.format_exc()
    }
    
    # Add context if available
    if hasattr(error, 'context') and error.context:
        summary["context"] = error.context.to_dict()
    
    # Add additional attributes
    for attr in ['error_code', 'details', 'field', 'value', 'operation', 'service']:
        if hasattr(error, attr):
            summary[attr] = getattr(error, attr)
    
    return summary

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global error factory instance
error_factory = ErrorFactory()

# Global context manager instance
context_manager = ErrorContextManager()

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_validation_error(field: str, value: Any, message: str,
                           context: Optional[ErrorContext] = None) -> VideoValidationError:
    """Create a validation error using the global factory."""
    return error_factory.create_validation_error(field, value, message, context)

def create_processing_error(operation: str, message: str,
                           context: Optional[ErrorContext] = None,
                           details: Optional[Dict[str, Any]] = None) -> VideoProcessingError:
    """Create a processing error using the global factory."""
    return error_factory.create_processing_error(operation, message, context, details)

def create_encoding_error(message: str, video_id: Optional[str] = None,
                         context: Optional[ErrorContext] = None,
                         details: Optional[Dict[str, Any]] = None) -> VideoEncodingError:
    """Create an encoding error using the global factory."""
    return error_factory.create_encoding_error(message, video_id, context, details)

def create_inference_error(message: str, model_name: str,
                          context: Optional[ErrorContext] = None,
                          details: Optional[Dict[str, Any]] = None) -> ModelInferenceError:
    """Create an inference error using the global factory."""
    return error_factory.create_inference_error(message, model_name, context, details)

def create_resource_error(message: str, resource: str,
                         available: Optional[Any] = None, required: Optional[Any] = None,
                         context: Optional[ErrorContext] = None) -> ResourceExhaustionError:
    """Create a resource error using the global factory."""
    return error_factory.create_resource_error(message, resource, available, required, context)

def create_api_error(message: str, service: str, endpoint: str,
                    status_code: Optional[int] = None,
                    context: Optional[ErrorContext] = None) -> APIError:
    """Create an API error using the global factory."""
    return error_factory.create_api_error(message, service, endpoint, status_code, context)

def create_security_error(message: str, threat_type: str, severity: str,
                         context: Optional[ErrorContext] = None) -> SecurityViolationError:
    """Create a security error using the global factory."""
    return error_factory.create_security_error(message, threat_type, severity, context) 