from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Any, Dict
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
AI Video System - Exceptions

Production-ready exception classes for the AI Video System.
"""



class AIVideoError(Exception):
    """Base exception for all AI Video System errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ConfigurationError(AIVideoError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_path = config_path
        if config_path:
            self.details["config_path"] = config_path


class PluginError(AIVideoError):
    """Plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="PLUGIN_ERROR", **kwargs)
        self.plugin_name = plugin_name
        if plugin_name:
            self.details["plugin_name"] = plugin_name


class DependencyError(AIVideoError):
    """Dependency-related errors."""
    
    def __init__(self, message: str, dependency_name: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        self.dependency_name = dependency_name
        if dependency_name:
            self.details["dependency_name"] = dependency_name


class WorkflowError(AIVideoError):
    """Workflow execution errors."""
    
    def __init__(self, message: str, workflow_id: Optional[str] = None, stage: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="WORKFLOW_ERROR", **kwargs)
        self.workflow_id = workflow_id
        self.stage = stage
        if workflow_id:
            self.details["workflow_id"] = workflow_id
        if stage:
            self.details["stage"] = stage


class ValidationError(AIVideoError):
    """Validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class ExtractionError(WorkflowError):
    """Content extraction errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, stage="extraction", **kwargs)
        self.url = url
        if url:
            self.details["url"] = url


class GenerationError(WorkflowError):
    """Video generation errors."""
    
    def __init__(self, message: str, **kwargs):
        
    """__init__ function."""
super().__init__(message, stage="generation", **kwargs)


class StorageError(AIVideoError):
    """Storage-related errors."""
    
    def __init__(self, message: str, storage_path: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="STORAGE_ERROR", **kwargs)
        self.storage_path = storage_path
        if storage_path:
            self.details["storage_path"] = storage_path


class SecurityError(AIVideoError):
    """Security-related errors."""
    
    def __init__(self, message: str, security_check: Optional[str] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.security_check = security_check
        if security_check:
            self.details["security_check"] = security_check


class PerformanceError(AIVideoError):
    """Performance-related errors."""
    
    def __init__(self, message: str, metric: Optional[str] = None, threshold: Optional[float] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="PERFORMANCE_ERROR", **kwargs)
        self.metric = metric
        self.threshold = threshold
        if metric:
            self.details["metric"] = metric
        if threshold is not None:
            self.details["threshold"] = threshold


class ResourceError(AIVideoError):
    """Resource-related errors (memory, CPU, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, current_usage: Optional[float] = None, **kwargs):
        
    """__init__ function."""
super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        if resource_type:
            self.details["resource_type"] = resource_type
        if current_usage is not None:
            self.details["current_usage"] = current_usage


# Convenience functions for error handling
def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> AIVideoError:
    """Convert any exception to an AIVideoError with context."""
    if isinstance(error, AIVideoError):
        if context:
            error.details.update(context)
        return error
    
    # Convert to AIVideoError
    ai_error = AIVideoError(str(error), error_code="UNKNOWN_ERROR")
    if context:
        ai_error.details.update(context)
    ai_error.details["original_error"] = error.__class__.__name__
    return ai_error


def is_recoverable_error(error: AIVideoError) -> bool:
    """Check if an error is recoverable."""
    non_recoverable_codes = {
        "SECURITY_ERROR",
        "VALIDATION_ERROR",
        "CONFIG_ERROR"
    }
    return error.error_code not in non_recoverable_codes


def should_retry_error(error: AIVideoError, retry_count: int, max_retries: int = 3) -> bool:
    """Check if an error should be retried."""
    if retry_count >= max_retries:
        return False
    
    retryable_codes = {
        "WORKFLOW_ERROR",
        "EXTRACTION_ERROR", 
        "GENERATION_ERROR",
        "STORAGE_ERROR",
        "PERFORMANCE_ERROR",
        "RESOURCE_ERROR"
    }
    return error.error_code in retryable_codes 