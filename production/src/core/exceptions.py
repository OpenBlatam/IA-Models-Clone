from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, Optional, List
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚨 Custom Exception Classes
==========================

Production-grade exception handling with proper error codes,
messages, and context information.
"""



class ErrorCode(Enum):
    """Standard error codes for the application"""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Business logic errors
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"
    INVALID_PROMPT = "INVALID_PROMPT"
    AI_SERVICE_UNAVAILABLE = "AI_SERVICE_UNAVAILABLE"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    
    # Infrastructure errors
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # AI/ML specific errors
    MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TOKEN_LIMIT_EXCEEDED = "TOKEN_LIMIT_EXCEEDED"
    CONTENT_FILTERED = "CONTENT_FILTERED"


class BaseException(Exception):
    """Base exception class for the application"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.code.value,
            "message": self.message,
            "details": self.details,
            "context": self.context
        }
    
    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"


class BusinessException(BaseException):
    """Exception for business logic errors"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(message, code, details, context)


class ValidationException(BaseException):
    """Exception for validation errors"""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[List[Dict[str, Any]]] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(
            message,
            ErrorCode.VALIDATION_ERROR,
            details or {"field_errors": field_errors or []},
            context
        )
        self.field_errors = field_errors or []


class NotFoundException(BaseException):
    """Exception for resource not found errors"""
    
    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
if message is None:
            message = f"{resource} not found"
            if resource_id:
                message += f" with id: {resource_id}"
        
        super().__init__(
            message,
            ErrorCode.NOT_FOUND,
            {"resource": resource, "resource_id": resource_id},
            context
        )


class UnauthorizedException(BaseException):
    """Exception for unauthorized access errors"""
    
    def __init__(
        self,
        message: str = "Unauthorized access",
        required_permissions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(
            message,
            ErrorCode.UNAUTHORIZED,
            {"required_permissions": required_permissions or []},
            context
        )


class ForbiddenException(BaseException):
    """Exception for forbidden access errors"""
    
    def __init__(
        self,
        message: str = "Access forbidden",
        required_permissions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(
            message,
            ErrorCode.FORBIDDEN,
            {"required_permissions": required_permissions or []},
            context
        )


class RateLimitException(BaseException):
    """Exception for rate limiting errors"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__(
            message,
            ErrorCode.RATE_LIMITED,
            {
                "retry_after": retry_after,
                "limit": limit
            },
            context
        )


class InsufficientCreditsException(BusinessException):
    """Exception for insufficient credits"""
    
    def __init__(
        self,
        required_credits: int,
        available_credits: int,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Insufficient credits. Required: {required_credits}, Available: {available_credits}"
        super().__init__(
            message,
            ErrorCode.INSUFFICIENT_CREDITS,
            {
                "required_credits": required_credits,
                "available_credits": available_credits
            },
            context
        )


class ContentTooLongException(BusinessException):
    """Exception for content that exceeds length limits"""
    
    def __init__(
        self,
        content_length: int,
        max_length: int,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Content too long. Length: {content_length}, Max: {max_length}"
        super().__init__(
            message,
            ErrorCode.CONTENT_TOO_LONG,
            {
                "content_length": content_length,
                "max_length": max_length
            },
            context
        )


class InvalidPromptException(BusinessException):
    """Exception for invalid prompts"""
    
    def __init__(
        self,
        prompt: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Invalid prompt: {reason}"
        super().__init__(
            message,
            ErrorCode.INVALID_PROMPT,
            {
                "prompt": prompt,
                "reason": reason
            },
            context
        )


class AIServiceUnavailableException(BusinessException):
    """Exception for AI service unavailability"""
    
    def __init__(
        self,
        service_name: str,
        reason: Optional[str] = None,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"AI service '{service_name}' is unavailable"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message,
            ErrorCode.AI_SERVICE_UNAVAILABLE,
            {
                "service_name": service_name,
                "reason": reason,
                "retry_after": retry_after
            },
            context
        )


class ProcessingFailedException(BusinessException):
    """Exception for processing failures"""
    
    def __init__(
        self,
        operation: str,
        reason: str,
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Processing failed for '{operation}': {reason}"
        super().__init__(
            message,
            ErrorCode.PROCESSING_FAILED,
            {
                "operation": operation,
                "reason": reason,
                "retryable": retryable
            },
            context
        )


class DatabaseException(BaseException):
    """Exception for database errors"""
    
    def __init__(
        self,
        operation: str,
        table: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Database error during {operation}"
        if table:
            message += f" on table '{table}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message,
            ErrorCode.DATABASE_ERROR,
            {
                "operation": operation,
                "table": table,
                "reason": reason
            },
            context
        )


class CacheException(BaseException):
    """Exception for cache errors"""
    
    def __init__(
        self,
        operation: str,
        key: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Cache error during {operation}"
        if key:
            message += f" for key '{key}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message,
            ErrorCode.CACHE_ERROR,
            {
                "operation": operation,
                "key": key,
                "reason": reason
            },
            context
        )


class ExternalAPIException(BaseException):
    """Exception for external API errors"""
    
    def __init__(
        self,
        service: str,
        endpoint: str,
        status_code: Optional[int] = None,
        response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"External API error for {service} at {endpoint}"
        if status_code:
            message += f" (Status: {status_code})"
        
        super().__init__(
            message,
            ErrorCode.EXTERNAL_API_ERROR,
            {
                "service": service,
                "endpoint": endpoint,
                "status_code": status_code,
                "response": response
            },
            context
        )


class ConfigurationException(BaseException):
    """Exception for configuration errors"""
    
    def __init__(
        self,
        config_key: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Configuration error for '{config_key}': {reason}"
        super().__init__(
            message,
            ErrorCode.CONFIGURATION_ERROR,
            {
                "config_key": config_key,
                "reason": reason
            },
            context
        )


class ModelLoadingException(BaseException):
    """Exception for AI model loading failures"""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(
            message,
            ErrorCode.MODEL_LOADING_FAILED,
            {
                "model_name": model_name,
                "reason": reason
            },
            context
        )


class InferenceException(BaseException):
    """Exception for AI inference failures"""
    
    def __init__(
        self,
        model_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        reason: str = "Inference failed",
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Inference failed for model '{model_name}': {reason}"
        super().__init__(
            message,
            ErrorCode.INFERENCE_FAILED,
            {
                "model_name": model_name,
                "input_data": input_data,
                "reason": reason
            },
            context
        )


class TokenLimitException(BaseException):
    """Exception for token limit exceeded"""
    
    def __init__(
        self,
        model_name: str,
        input_tokens: int,
        max_tokens: int,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Token limit exceeded for model '{model_name}'. Input: {input_tokens}, Max: {max_tokens}"
        super().__init__(
            message,
            ErrorCode.TOKEN_LIMIT_EXCEEDED,
            {
                "model_name": model_name,
                "input_tokens": input_tokens,
                "max_tokens": max_tokens
            },
            context
        )


class ContentFilteredException(BaseException):
    """Exception for content that was filtered"""
    
    def __init__(
        self,
        content: str,
        filter_reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Content was filtered: {filter_reason}"
        super().__init__(
            message,
            ErrorCode.CONTENT_FILTERED,
            {
                "content": content,
                "filter_reason": filter_reason
            },
            context
        )


# Container exception
class ContainerError(Exception):
    """Exception for dependency injection container errors"""
    pass 