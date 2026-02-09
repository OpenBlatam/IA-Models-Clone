from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, Optional
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video System - Exceptions

Custom exceptions for the Onyx AI Video system with proper error handling
and integration with Onyx's error handling patterns.
"""



class AIVideoError(Exception):
    """
    Base exception for all AI Video system errors.
    
    Provides structured error information and integrates with
    Onyx's error handling patterns.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code or "AI_VIDEO_ERROR"
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


class PluginError(AIVideoError):
    """Exception raised for plugin-related errors."""
    
    def __init__(
        self, 
        message: str, 
        plugin_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "PLUGIN_ERROR", context, original_error)
        self.plugin_name = plugin_name
        if plugin_name:
            self.context["plugin_name"] = plugin_name
    
    def __str__(self) -> str:
        plugin_info = f" (Plugin: {self.plugin_name})" if self.plugin_name else ""
        return f"{self.error_code}: {self.message}{plugin_info}"


class ValidationError(AIVideoError):
    """Exception raised for validation errors."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "VALIDATION_ERROR", context, original_error)
        self.field = field
        self.value = value
        if field:
            self.context["field"] = field
        if value is not None:
            self.context["value"] = str(value)
    
    def __str__(self) -> str:
        field_info = f" (Field: {self.field})" if self.field else ""
        return f"{self.error_code}: {self.message}{field_info}"


class ConfigurationError(AIVideoError):
    """Exception raised for configuration errors."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "CONFIGURATION_ERROR", context, original_error)
        self.config_key = config_key
        self.config_value = config_value
        if config_key:
            self.context["config_key"] = config_key
        if config_value is not None:
            self.context["config_value"] = str(config_value)
    
    def __str__(self) -> str:
        key_info = f" (Key: {self.config_key})" if self.config_key else ""
        return f"{self.error_code}: {self.message}{key_info}"


class WorkflowError(AIVideoError):
    """Exception raised for workflow-related errors."""
    
    def __init__(
        self, 
        message: str, 
        workflow_step: Optional[str] = None,
        step_number: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "WORKFLOW_ERROR", context, original_error)
        self.workflow_step = workflow_step
        self.step_number = step_number
        if workflow_step:
            self.context["workflow_step"] = workflow_step
        if step_number is not None:
            self.context["step_number"] = step_number
    
    def __str__(self) -> str:
        step_info = f" (Step: {self.workflow_step})" if self.workflow_step else ""
        return f"{self.error_code}: {self.message}{step_info}"


class LLMError(AIVideoError):
    """Exception raised for LLM-related errors."""
    
    def __init__(
        self, 
        message: str, 
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "LLM_ERROR", context, original_error)
        self.llm_provider = llm_provider
        self.model_name = model_name
        if llm_provider:
            self.context["llm_provider"] = llm_provider
        if model_name:
            self.context["model_name"] = model_name
    
    def __str__(self) -> str:
        provider_info = f" (Provider: {self.llm_provider})" if self.llm_provider else ""
        model_info = f" (Model: {self.model_name})" if self.model_name else ""
        return f"{self.error_code}: {self.message}{provider_info}{model_info}"


class ResourceError(AIVideoError):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "RESOURCE_ERROR", context, original_error)
        self.resource_type = resource_type
        self.resource_id = resource_id
        if resource_type:
            self.context["resource_type"] = resource_type
        if resource_id:
            self.context["resource_id"] = resource_id
    
    def __str__(self) -> str:
        resource_info = f" (Resource: {self.resource_type}:{self.resource_id})" if self.resource_type and self.resource_id else ""
        return f"{self.error_code}: {self.message}{resource_info}"


class TimeoutError(AIVideoError):
    """Exception raised for timeout errors."""
    
    def __init__(
        self, 
        message: str, 
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "TIMEOUT_ERROR", context, original_error)
        self.timeout_duration = timeout_duration
        self.operation = operation
        if timeout_duration is not None:
            self.context["timeout_duration"] = timeout_duration
        if operation:
            self.context["operation"] = operation
    
    def __str__(self) -> str:
        operation_info = f" (Operation: {self.operation})" if self.operation else ""
        duration_info = f" (Duration: {self.timeout_duration}s)" if self.timeout_duration is not None else ""
        return f"{self.error_code}: {self.message}{operation_info}{duration_info}"


class SecurityError(AIVideoError):
    """Exception raised for security-related errors."""
    
    def __init__(
        self, 
        message: str, 
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        permission: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "SECURITY_ERROR", context, original_error)
        self.user_id = user_id
        self.resource_id = resource_id
        self.permission = permission
        if user_id:
            self.context["user_id"] = user_id
        if resource_id:
            self.context["resource_id"] = resource_id
        if permission:
            self.context["permission"] = permission
    
    def __str__(self) -> str:
        user_info = f" (User: {self.user_id})" if self.user_id else ""
        resource_info = f" (Resource: {self.resource_id})" if self.resource_id else ""
        return f"{self.error_code}: {self.message}{user_info}{resource_info}"


class PerformanceError(AIVideoError):
    """Exception raised for performance-related errors."""
    
    def __init__(
        self, 
        message: str, 
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        
    """__init__ function."""
super().__init__(message, "PERFORMANCE_ERROR", context, original_error)
        self.metric = metric
        self.threshold = threshold
        self.actual_value = actual_value
        if metric:
            self.context["metric"] = metric
        if threshold is not None:
            self.context["threshold"] = threshold
        if actual_value is not None:
            self.context["actual_value"] = actual_value
    
    def __str__(self) -> str:
        metric_info = f" (Metric: {self.metric})" if self.metric else ""
        threshold_info = f" (Threshold: {self.threshold})" if self.threshold is not None else ""
        return f"{self.error_code}: {self.message}{metric_info}{threshold_info}"


# Error handling utilities
def handle_ai_video_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handle AI Video errors and convert to standardized format.
    
    Args:
        error: The exception that occurred
        context: Optional context information
        
    Returns:
        Standardized error response
    """
    if isinstance(error, AIVideoError):
        error_dict = error.to_dict()
    else:
        error_dict = {
            "error_type": error.__class__.__name__,
            "error_code": "UNKNOWN_ERROR",
            "message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "original_error": None
        }
    
    if context:
        error_dict["context"].update(context)
    
    return error_dict


def create_error_response(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        error: The exception that occurred
        context: Optional context information
        
    Returns:
        API error response
    """
    error_info = handle_ai_video_error(error, context)
    
    return {
        "status": "error",
        "error": error_info,
        "timestamp": datetime.now().isoformat()
    }


# Error codes mapping
ERROR_CODES = {
    "AI_VIDEO_ERROR": "General AI Video system error",
    "PLUGIN_ERROR": "Plugin execution or configuration error",
    "VALIDATION_ERROR": "Input validation error",
    "CONFIGURATION_ERROR": "Configuration error",
    "WORKFLOW_ERROR": "Workflow execution error",
    "LLM_ERROR": "Language model error",
    "RESOURCE_ERROR": "Resource access or allocation error",
    "TIMEOUT_ERROR": "Operation timeout error",
    "SECURITY_ERROR": "Security or access control error",
    "PERFORMANCE_ERROR": "Performance threshold exceeded error"
}


def get_error_description(error_code: str) -> str:
    """Get human-readable description for error code."""
    return ERROR_CODES.get(error_code, "Unknown error")


# Exception chaining utilities
def chain_exceptions(original_error: Exception, new_message: str, error_class: type = AIVideoError, **kwargs) -> AIVideoError:
    """
    Chain exceptions while preserving original error information.
    
    Args:
        original_error: The original exception
        new_message: New error message
        error_class: Exception class to create
        **kwargs: Additional arguments for the exception
        
    Returns:
        New exception with chained information
    """
    return error_class(
        message=new_message,
        original_error=original_error,
        context=kwargs.get("context", {}),
        **{k: v for k, v in kwargs.items() if k != "context"}
    ) 