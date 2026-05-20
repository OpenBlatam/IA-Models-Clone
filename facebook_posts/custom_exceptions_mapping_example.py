from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
import traceback
import sys
import inspect
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr
import numpy as np
from pathlib import Path
from enum import Enum
from functools import wraps
import json
import uuid
from datetime import datetime, timezone
import structlog
from typing import Any, List, Dict, Optional
"""
Custom Exceptions with User-Friendly Mapping - Complete Patterns
==============================================================

This file demonstrates custom exceptions with comprehensive mapping:
- Custom exception hierarchy
- User-friendly message mapping
- CLI/API message formatting
- Error categorization and severity
- Exception factory patterns
- Message localization support
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Custom Exceptions
    "BaseCustomException",
    "TimeoutError",
    "InvalidTargetError",
    "ValidationError",
    "ProcessingError",
    "NetworkError",
    "SecurityError",
    "DatabaseError",
    
    # Exception Mapping
    "ExceptionMapper",
    "CLIExceptionMapper",
    "APIExceptionMapper",
    "UserFriendlyMapper",
    
    # Exception Factories
    "ExceptionFactory",
    "TimeoutExceptionFactory",
    "ValidationExceptionFactory",
    "ProcessingExceptionFactory",
    
    # Message Patterns
    "MessagePatterns",
    "CLIMessagePatterns",
    "APIMessagePatterns",
    
    # Common utilities
    "ExceptionResult",
    "ExceptionConfig",
    "ExceptionType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class ExceptionResult(BaseModel):
    """Pydantic model for exception results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether exception handling was successful")
    exception_id: str = Field(description="Unique exception identifier")
    exception_type: str = Field(description="Type of exception")
    original_message: str = Field(description="Original exception message")
    user_friendly_message: str = Field(description="User-friendly message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    context: Dict[str, Any] = Field(default_factory=dict, description="Exception context")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    suggestions: Optional[List[str]] = Field(default=None, description="Suggested solutions")

class ExceptionConfig(BaseModel):
    """Pydantic model for exception configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    enable_user_friendly_messages: bool = Field(default=True, description="Enable user-friendly messages")
    enable_error_codes: bool = Field(default=True, description="Enable error codes")
    enable_suggestions: bool = Field(default=True, description="Enable suggestions")
    enable_context_logging: bool = Field(default=True, description="Enable context logging")
    max_message_length: conint(ge=50, le=1000) = Field(default=200, description="Maximum message length")
    message_format: Literal["cli", "api", "both"] = Field(default="both", description="Message format")
    enable_localization: bool = Field(default=False, description="Enable message localization")
    default_language: constr(strip_whitespace=True) = Field(default="en", description="Default language")

class ExceptionType(BaseModel):
    """Pydantic model for exception type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    exception_type: constr(strip_whitespace=True) = Field(
        pattern=r"^(timeout|validation|processing|network|security|database|general)$"
    )
    description: Optional[str] = Field(default=None)
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class BaseCustomException(Exception):
    """Base class for all custom exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: Literal["low", "medium", "high", "critical"] = "medium",
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        
    """__init__ function."""
self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.suggestions = suggestions or []
        self.timestamp = datetime.now(timezone.utc)
        self.exception_id = str(uuid.uuid4())
        
        # Build error message
        error_msg = f"{self.__class__.__name__}: {message}"
        if error_code:
            error_msg += f" (Code: {error_code})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "exception_id": self.exception_id,
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
            "context": self.context,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly message."""
        return self.message

class TimeoutError(BaseCustomException):
    """Custom timeout exception."""
    
    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Operation '{operation}' timed out after {timeout_duration} seconds"
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            severity="high",
            context=context or {},
            suggestions=[
                "Check network connectivity",
                "Increase timeout duration",
                "Verify target availability",
                "Consider using a different server"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly timeout message."""
        return f"⏰ The operation took too long to complete. Please try again or check your connection."

class InvalidTargetError(BaseCustomException):
    """Custom invalid target exception."""
    
    def __init__(
        self,
        target: str,
        target_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Invalid {target_type}: '{target}' - {reason}"
        super().__init__(
            message=message,
            error_code="INVALID_TARGET_ERROR",
            severity="medium",
            context=context or {},
            suggestions=[
                "Verify the target format",
                "Check for typos in the target",
                "Ensure the target is accessible",
                "Use a different target if available"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly invalid target message."""
        return f"🎯 The target you specified is not valid. Please check the format and try again."

class ValidationError(BaseCustomException):
    """Custom validation exception."""
    
    def __init__(
        self,
        field_name: str,
        field_value: Any,
        validation_rule: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Validation failed for field '{field_name}' with value '{field_value}' - {validation_rule}"
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            severity="medium",
            context=context or {},
            suggestions=[
                "Check the input format",
                "Verify required fields are provided",
                "Ensure values are within acceptable ranges",
                "Review the validation rules"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly validation message."""
        return f"✅ Please check your input and ensure all required fields are correctly filled."

class ProcessingError(BaseCustomException):
    """Custom processing exception."""
    
    def __init__(
        self,
        operation: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Processing failed for operation '{operation}' - {reason}"
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            severity="high",
            context=context or {},
            suggestions=[
                "Check system resources",
                "Verify input data format",
                "Try again with different parameters",
                "Contact support if the issue persists"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly processing message."""
        return f"⚙️ The operation encountered an issue. Please try again or contact support."

class NetworkError(BaseCustomException):
    """Custom network exception."""
    
    def __init__(
        self,
        url: str,
        status_code: Optional[int] = None,
        reason: str = "Network connection failed",
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Network error for '{url}' - {reason}"
        if status_code:
            message += f" (Status: {status_code})"
        
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            severity="high",
            context=context or {},
            suggestions=[
                "Check your internet connection",
                "Verify the URL is correct",
                "Try again in a few moments",
                "Check firewall settings"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly network message."""
        return f"🌐 Network connection issue. Please check your internet connection and try again."

class SecurityError(BaseCustomException):
    """Custom security exception."""
    
    def __init__(
        self,
        security_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Security error: {security_type} - {reason}"
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            severity="critical",
            context=context or {},
            suggestions=[
                "Verify your credentials",
                "Check access permissions",
                "Contact system administrator",
                "Review security policies"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly security message."""
        return f"🔒 Access denied. Please verify your credentials and permissions."

class DatabaseError(BaseCustomException):
    """Custom database exception."""
    
    def __init__(
        self,
        operation: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
message = f"Database error during '{operation}' - {reason}"
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            severity="high",
            context=context or {},
            suggestions=[
                "Check database connectivity",
                "Verify database credentials",
                "Review database logs",
                "Contact database administrator"
            ]
        )
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly database message."""
        return f"💾 Database operation failed. Please try again or contact support."

# ============================================================================
# EXCEPTION MAPPING
# ============================================================================

class ExceptionMapper:
    """Base exception mapper with comprehensive mapping."""
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
self.config = config
        self.mapping_rules = self._setup_mapping_rules()
        self.logger = logging.getLogger(__name__)
    
    def _setup_mapping_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup mapping rules for exceptions."""
        return {
            "TimeoutError": {
                "cli_message": "⏰ Operation timed out. Please try again.",
                "api_message": "Operation timed out. Please retry the request.",
                "error_code": "TIMEOUT_ERROR",
                "severity": "high",
                "suggestions": [
                    "Check network connectivity",
                    "Increase timeout duration",
                    "Verify target availability"
                ]
            },
            "InvalidTargetError": {
                "cli_message": "🎯 Invalid target specified. Please check the format.",
                "api_message": "Invalid target format. Please verify the input.",
                "error_code": "INVALID_TARGET_ERROR",
                "severity": "medium",
                "suggestions": [
                    "Verify the target format",
                    "Check for typos",
                    "Ensure target is accessible"
                ]
            },
            "ValidationError": {
                "cli_message": "✅ Input validation failed. Please check your data.",
                "api_message": "Validation error. Please review the input parameters.",
                "error_code": "VALIDATION_ERROR",
                "severity": "medium",
                "suggestions": [
                    "Check input format",
                    "Verify required fields",
                    "Review validation rules"
                ]
            },
            "ProcessingError": {
                "cli_message": "⚙️ Processing failed. Please try again.",
                "api_message": "Processing error occurred. Please retry the operation.",
                "error_code": "PROCESSING_ERROR",
                "severity": "high",
                "suggestions": [
                    "Check system resources",
                    "Verify input data",
                    "Try different parameters"
                ]
            },
            "NetworkError": {
                "cli_message": "🌐 Network connection failed. Please check your connection.",
                "api_message": "Network error. Please verify connectivity and retry.",
                "error_code": "NETWORK_ERROR",
                "severity": "high",
                "suggestions": [
                    "Check internet connection",
                    "Verify URL correctness",
                    "Check firewall settings"
                ]
            },
            "SecurityError": {
                "cli_message": "🔒 Access denied. Please verify your credentials.",
                "api_message": "Security error. Please check authentication and permissions.",
                "error_code": "SECURITY_ERROR",
                "severity": "critical",
                "suggestions": [
                    "Verify credentials",
                    "Check permissions",
                    "Contact administrator"
                ]
            },
            "DatabaseError": {
                "cli_message": "💾 Database operation failed. Please try again.",
                "api_message": "Database error. Please retry the operation.",
                "error_code": "DATABASE_ERROR",
                "severity": "high",
                "suggestions": [
                    "Check database connectivity",
                    "Verify credentials",
                    "Review database logs"
                ]
            }
        }
    
    def map_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ExceptionResult:
        """Map exception to user-friendly format."""
        try:
            exception_type = type(exception).__name__
            mapping_rule = self.mapping_rules.get(exception_type, {})
            
            # Get user-friendly message
            if isinstance(exception, BaseCustomException):
                user_message = exception.get_user_friendly_message()
                error_code = exception.error_code
                severity = exception.severity
                suggestions = exception.suggestions
            else:
                user_message = mapping_rule.get("cli_message", str(exception))
                error_code = mapping_rule.get("error_code", "UNKNOWN_ERROR")
                severity = mapping_rule.get("severity", "medium")
                suggestions = mapping_rule.get("suggestions", [])
            
            # Truncate message if needed
            if len(user_message) > self.config.max_message_length:
                user_message = user_message[:self.config.max_message_length] + "..."
            
            return ExceptionResult(
                is_successful=True,
                exception_id=str(uuid.uuid4()),
                exception_type=exception_type,
                original_message=str(exception),
                user_friendly_message=user_message,
                error_code=error_code if self.config.enable_error_codes else None,
                severity=severity,
                context=context or {},
                suggestions=suggestions if self.config.enable_suggestions else None
            )
            
        except Exception as exc:
            return ExceptionResult(
                is_successful=False,
                exception_id=str(uuid.uuid4()),
                exception_type="MappingError",
                original_message=str(exception),
                user_friendly_message="An error occurred while processing the error.",
                error_code="MAPPING_ERROR",
                severity="high"
            )

class CLIExceptionMapper(ExceptionMapper):
    """CLI-specific exception mapper."""
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
super().__init__(config)
        self.cli_formatting = self._setup_cli_formatting()
    
    def _setup_cli_formatting(self) -> Dict[str, str]:
        """Setup CLI formatting rules."""
        return {
            "error_prefix": "❌",
            "warning_prefix": "⚠️",
            "info_prefix": "ℹ️",
            "success_prefix": "✅",
            "timeout_prefix": "⏰",
            "network_prefix": "🌐",
            "security_prefix": "🔒",
            "database_prefix": "💾",
            "processing_prefix": "⚙️",
            "validation_prefix": "✅"
        }
    
    def format_cli_message(
        self,
        exception: Exception,
        include_suggestions: bool = True,
        include_context: bool = False
    ) -> str:
        """Format exception for CLI display."""
        try:
            result = self.map_exception(exception)
            
            if not result.is_successful:
                return f"{self.cli_formatting['error_prefix']} Error mapping failed"
            
            # Build CLI message
            message_parts = []
            
            # Add prefix based on exception type
            exception_type = type(exception).__name__
            prefix_key = f"{exception_type.lower().replace('error', '')}_prefix"
            prefix = self.cli_formatting.get(prefix_key, self.cli_formatting['error_prefix'])
            
            message_parts.append(f"{prefix} {result.user_friendly_message}")
            
            # Add error code if enabled
            if self.config.enable_error_codes and result.error_code:
                message_parts.append(f"Code: {result.error_code}")
            
            # Add suggestions if enabled
            if include_suggestions and result.suggestions:
                message_parts.append("\nSuggestions:")
                for suggestion in result.suggestions:
                    message_parts.append(f"  • {suggestion}")
            
            # Add context if enabled
            if include_context and result.context:
                message_parts.append("\nContext:")
                for key, value in result.context.items():
                    message_parts.append(f"  {key}: {value}")
            
            return "\n".join(message_parts)
            
        except Exception as exc:
            return f"{self.cli_formatting['error_prefix']} Error formatting failed: {str(exc)}"

class APIExceptionMapper(ExceptionMapper):
    """API-specific exception mapper."""
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
super().__init__(config)
        self.api_formatting = self._setup_api_formatting()
    
    async def _setup_api_formatting(self) -> Dict[str, Any]:
        """Setup API formatting rules."""
        return {
            "include_timestamp": True,
            "include_error_code": True,
            "include_suggestions": True,
            "include_context": True,
            "response_format": "json"
        }
    
    async def format_api_response(
        self,
        exception: Exception,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """Format exception for API response."""
        try:
            result = self.map_exception(exception)
            
            if not result.is_successful:
                return {
                    "error": "Error mapping failed",
                    "message": "An error occurred while processing the error",
                    "code": "MAPPING_ERROR"
                }
            
            # Build API response
            response = {
                "error": True,
                "message": result.user_friendly_message,
                "code": result.error_code,
                "severity": result.severity,
                "timestamp": result.timestamp.isoformat()
            }
            
            # Add details if requested
            if include_details:
                response.update({
                    "original_message": result.original_message,
                    "exception_type": result.exception_type,
                    "suggestions": result.suggestions,
                    "context": result.context
                })
            
            return response
            
        except Exception as exc:
            return {
                "error": True,
                "message": "Error formatting failed",
                "code": "FORMATTING_ERROR",
                "details": str(exc)
            }

class UserFriendlyMapper:
    """User-friendly exception mapper with comprehensive formatting."""
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
self.config = config
        self.cli_mapper = CLIExceptionMapper(config)
        self.api_mapper = APIExceptionMapper(config)
        self.logger = logging.getLogger(__name__)
    
    def map_exception_user_friendly(
        self,
        exception: Exception,
        format_type: Literal["cli", "api", "both"] = "both",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Map exception to user-friendly format."""
        try:
            result = {
                "cli_message": None,
                "api_response": None,
                "mapping_result": None
            }
            
            # Map the exception
            mapping_result = self.cli_mapper.map_exception(exception, context)
            result["mapping_result"] = mapping_result
            
            # Format based on requested type
            if format_type in ["cli", "both"]:
                result["cli_message"] = self.cli_mapper.format_cli_message(
                    exception,
                    include_suggestions=self.config.enable_suggestions,
                    include_context=self.config.enable_context_logging
                )
            
            if format_type in ["api", "both"]:
                result["api_response"] = self.api_mapper.format_api_response(
                    exception,
                    include_details=self.config.enable_context_logging
                )
            
            return result
            
        except Exception as exc:
            self.logger.error(f"User-friendly mapping failed: {str(exc)}")
            return {
                "cli_message": "❌ Error mapping failed",
                "api_response": {"error": True, "message": "Error mapping failed"},
                "mapping_result": None
            }

# ============================================================================
# EXCEPTION FACTORIES
# ============================================================================

class ExceptionFactory:
    """Factory for creating custom exceptions."""
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_timeout_exception(
        self,
        operation: str,
        timeout_duration: float,
        context: Optional[Dict[str, Any]] = None
    ) -> TimeoutError:
        """Create timeout exception."""
        return TimeoutError(
            operation=operation,
            timeout_duration=timeout_duration,
            context=context
        )
    
    def create_invalid_target_exception(
        self,
        target: str,
        target_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> InvalidTargetError:
        """Create invalid target exception."""
        return InvalidTargetError(
            target=target,
            target_type=target_type,
            reason=reason,
            context=context
        )
    
    def create_validation_exception(
        self,
        field_name: str,
        field_value: Any,
        validation_rule: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """Create validation exception."""
        return ValidationError(
            field_name=field_name,
            field_value=field_value,
            validation_rule=validation_rule,
            context=context
        )
    
    def create_processing_exception(
        self,
        operation: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingError:
        """Create processing exception."""
        return ProcessingError(
            operation=operation,
            reason=reason,
            context=context
        )
    
    def create_network_exception(
        self,
        url: str,
        status_code: Optional[int] = None,
        reason: str = "Network connection failed",
        context: Optional[Dict[str, Any]] = None
    ) -> NetworkError:
        """Create network exception."""
        return NetworkError(
            url=url,
            status_code=status_code,
            reason=reason,
            context=context
        )
    
    def create_security_exception(
        self,
        security_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityError:
        """Create security exception."""
        return SecurityError(
            security_type=security_type,
            reason=reason,
            context=context
        )
    
    def create_database_exception(
        self,
        operation: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DatabaseError:
        """Create database exception."""
        return DatabaseError(
            operation=operation,
            reason=reason,
            context=context
        )

class TimeoutExceptionFactory(ExceptionFactory):
    """Specialized factory for timeout exceptions."""
    
    def create_connection_timeout(
        self,
        target: str,
        timeout_duration: float,
        context: Optional[Dict[str, Any]] = None
    ) -> TimeoutError:
        """Create connection timeout exception."""
        return self.create_timeout_exception(
            operation=f"connection to {target}",
            timeout_duration=timeout_duration,
            context=context
        )
    
    def create_operation_timeout(
        self,
        operation: str,
        timeout_duration: float,
        context: Optional[Dict[str, Any]] = None
    ) -> TimeoutError:
        """Create operation timeout exception."""
        return self.create_timeout_exception(
            operation=operation,
            timeout_duration=timeout_duration,
            context=context
        )

class ValidationExceptionFactory(ExceptionFactory):
    """Specialized factory for validation exceptions."""
    
    def create_field_validation_error(
        self,
        field_name: str,
        field_value: Any,
        expected_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """Create field validation error."""
        return self.create_validation_exception(
            field_name=field_name,
            field_value=field_value,
            validation_rule=f"Expected type: {expected_type}",
            context=context
        )
    
    def create_required_field_error(
        self,
        field_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """Create required field error."""
        return self.create_validation_exception(
            field_name=field_name,
            field_value=None,
            validation_rule="Field is required",
            context=context
        )

class ProcessingExceptionFactory(ExceptionFactory):
    """Specialized factory for processing exceptions."""
    
    def create_data_processing_error(
        self,
        data_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingError:
        """Create data processing error."""
        return self.create_processing_exception(
            operation=f"data processing ({data_type})",
            reason=reason,
            context=context
        )
    
    def create_file_processing_error(
        self,
        file_path: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingError:
        """Create file processing error."""
        return self.create_processing_exception(
            operation=f"file processing ({file_path})",
            reason=reason,
            context=context
        )

# ============================================================================
# MESSAGE PATTERNS
# ============================================================================

class MessagePatterns:
    """Message patterns for exception handling."""
    
    @staticmethod
    def format_error_message(
        exception: Exception,
        include_code: bool = True,
        include_suggestions: bool = True
    ) -> str:
        """Format error message with patterns."""
        try:
            if isinstance(exception, BaseCustomException):
                message = exception.get_user_friendly_message()
                if include_code and exception.error_code:
                    message += f" (Code: {exception.error_code})"
                if include_suggestions and exception.suggestions:
                    message += "\n\nSuggestions:\n"
                    for suggestion in exception.suggestions:
                        message += f"• {suggestion}\n"
                return message
            else:
                return str(exception)
        except Exception:
            return str(exception)

class CLIMessagePatterns:
    """CLI-specific message patterns."""
    
    @staticmethod
    def format_cli_error(
        exception: Exception,
        include_emoji: bool = True,
        include_details: bool = False
    ) -> str:
        """Format CLI error message."""
        try:
            if isinstance(exception, TimeoutError):
                prefix = "⏰" if include_emoji else "TIMEOUT"
                message = "Operation timed out. Please try again."
            elif isinstance(exception, InvalidTargetError):
                prefix = "🎯" if include_emoji else "INVALID_TARGET"
                message = "Invalid target specified. Please check the format."
            elif isinstance(exception, ValidationError):
                prefix = "✅" if include_emoji else "VALIDATION"
                message = "Input validation failed. Please check your data."
            elif isinstance(exception, ProcessingError):
                prefix = "⚙️" if include_emoji else "PROCESSING"
                message = "Processing failed. Please try again."
            elif isinstance(exception, NetworkError):
                prefix = "🌐" if include_emoji else "NETWORK"
                message = "Network connection failed. Please check your connection."
            elif isinstance(exception, SecurityError):
                prefix = "🔒" if include_emoji else "SECURITY"
                message = "Access denied. Please verify your credentials."
            elif isinstance(exception, DatabaseError):
                prefix = "💾" if include_emoji else "DATABASE"
                message = "Database operation failed. Please try again."
            else:
                prefix = "❌" if include_emoji else "ERROR"
                message = str(exception)
            
            formatted_message = f"{prefix} {message}"
            
            if include_details and isinstance(exception, BaseCustomException):
                if exception.error_code:
                    formatted_message += f"\nCode: {exception.error_code}"
                if exception.suggestions:
                    formatted_message += "\n\nSuggestions:"
                    for suggestion in exception.suggestions:
                        formatted_message += f"\n• {suggestion}"
            
            return formatted_message
            
        except Exception:
            return f"❌ Error: {str(exception)}"

class APIMessagePatterns:
    """API-specific message patterns."""
    
    @staticmethod
    async def format_api_error(
        exception: Exception,
        include_details: bool = True,
        include_timestamp: bool = True
    ) -> Dict[str, Any]:
        """Format API error response."""
        try:
            response = {
                "error": True,
                "message": str(exception),
                "timestamp": datetime.now(timezone.utc).isoformat() if include_timestamp else None
            }
            
            if isinstance(exception, BaseCustomException):
                response.update({
                    "user_friendly_message": exception.get_user_friendly_message(),
                    "error_code": exception.error_code,
                    "severity": exception.severity,
                    "suggestions": exception.suggestions if include_details else None,
                    "context": exception.context if include_details else None
                })
            
            # Remove None values
            response = {k: v for k, v in response.items() if v is not None}
            
            return response
            
        except Exception:
            return {
                "error": True,
                "message": "Error formatting failed",
                "timestamp": datetime.now(timezone.utc).isoformat() if include_timestamp else None
            }

# ============================================================================
# MAIN CUSTOM EXCEPTIONS MODULE
# ============================================================================

class MainCustomExceptionsModule:
    """Main custom exceptions module with proper imports and exports."""
    
    # Define main exports
    __all__ = [
        # Custom Exceptions
        "BaseCustomException",
        "TimeoutError",
        "InvalidTargetError",
        "ValidationError",
        "ProcessingError",
        "NetworkError",
        "SecurityError",
        "DatabaseError",
        
        # Exception Mapping
        "ExceptionMapper",
        "CLIExceptionMapper",
        "APIExceptionMapper",
        "UserFriendlyMapper",
        
        # Exception Factories
        "ExceptionFactory",
        "TimeoutExceptionFactory",
        "ValidationExceptionFactory",
        "ProcessingExceptionFactory",
        
        # Message Patterns
        "MessagePatterns",
        "CLIMessagePatterns",
        "APIMessagePatterns",
        
        # Common utilities
        "ExceptionResult",
        "ExceptionConfig",
        "ExceptionType",
        
        # Main functions
        "raise_custom_exception",
        "map_exception_user_friendly",
        "format_exception_message"
    ]
    
    def __init__(self, config: ExceptionConfig):
        
    """__init__ function."""
self.config = config
        self.exception_factory = ExceptionFactory(config)
        self.user_friendly_mapper = UserFriendlyMapper(config)
        self.cli_mapper = CLIExceptionMapper(config)
        self.api_mapper = APIExceptionMapper(config)
    
    def raise_custom_exception(
        self,
        exception_type: str,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseCustomException:
        """Raise custom exception with comprehensive context."""
        try:
            if exception_type == "timeout":
                return self.exception_factory.create_timeout_exception(
                    operation=message,
                    timeout_duration=context.get("timeout_duration", 30.0),
                    context=context
                )
            elif exception_type == "invalid_target":
                return self.exception_factory.create_invalid_target_exception(
                    target=context.get("target", ""),
                    target_type=context.get("target_type", "target"),
                    reason=message,
                    context=context
                )
            elif exception_type == "validation":
                return self.exception_factory.create_validation_exception(
                    field_name=context.get("field_name", "unknown"),
                    field_value=context.get("field_value"),
                    validation_rule=message,
                    context=context
                )
            elif exception_type == "processing":
                return self.exception_factory.create_processing_exception(
                    operation=context.get("operation", "unknown"),
                    reason=message,
                    context=context
                )
            elif exception_type == "network":
                return self.exception_factory.create_network_exception(
                    url=context.get("url", ""),
                    status_code=context.get("status_code"),
                    reason=message,
                    context=context
                )
            elif exception_type == "security":
                return self.exception_factory.create_security_exception(
                    security_type=context.get("security_type", "authentication"),
                    reason=message,
                    context=context
                )
            elif exception_type == "database":
                return self.exception_factory.create_database_exception(
                    operation=context.get("operation", "unknown"),
                    reason=message,
                    context=context
                )
            else:
                # Default to processing error
                return self.exception_factory.create_processing_exception(
                    operation="unknown",
                    reason=message,
                    context=context
                )
                
        except Exception as exc:
            # Fallback to generic exception
            return BaseCustomException(
                message=f"Error creating custom exception: {str(exc)}",
                error_code="EXCEPTION_CREATION_ERROR",
                severity="high"
            )
    
    def map_exception_user_friendly(
        self,
        exception: Exception,
        format_type: Literal["cli", "api", "both"] = "both"
    ) -> Dict[str, Any]:
        """Map exception to user-friendly format."""
        try:
            return self.user_friendly_mapper.map_exception_user_friendly(
                exception=exception,
                format_type=format_type
            )
        except Exception as exc:
            return {
                "cli_message": f"❌ Error mapping failed: {str(exc)}",
                "api_response": {"error": True, "message": "Error mapping failed"},
                "mapping_result": None
            }
    
    def format_exception_message(
        self,
        exception: Exception,
        format_type: Literal["cli", "api"] = "cli"
    ) -> str:
        """Format exception message for display."""
        try:
            if format_type == "cli":
                return CLIMessagePatterns.format_cli_error(
                    exception,
                    include_emoji=True,
                    include_details=True
                )
            elif format_type == "api":
                response = APIMessagePatterns.format_api_error(
                    exception,
                    include_details=True,
                    include_timestamp=True
                )
                return json.dumps(response, indent=2)
            else:
                return str(exception)
        except Exception as exc:
            return f"Error formatting message: {str(exc)}"

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_custom_exceptions_mapping():
    """Demonstrate custom exceptions with user-friendly mapping."""
    
    print("🚨 Demonstrating Custom Exceptions with User-Friendly Mapping")
    print("=" * 70)
    
    # Initialize custom exceptions module
    config = ExceptionConfig(
        enable_user_friendly_messages=True,
        enable_error_codes=True,
        enable_suggestions=True,
        enable_context_logging=True,
        max_message_length=200,
        message_format="both",
        enable_localization=False,
        default_language="en"
    )
    
    main_module = MainCustomExceptionsModule(config)
    
    # Example 1: Timeout Error
    print("\n⏰ Timeout Error Example:")
    
    timeout_exception = main_module.raise_custom_exception(
        exception_type="timeout",
        message="Database query timed out",
        context={
            "timeout_duration": 30.0,
            "query": "SELECT * FROM large_table",
            "user_id": "12345"
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(timeout_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 2: Invalid Target Error
    print("\n🎯 Invalid Target Error Example:")
    
    invalid_target_exception = main_module.raise_custom_exception(
        exception_type="invalid_target",
        message="Invalid IP address format",
        context={
            "target": "256.256.256.256",
            "target_type": "IP address",
            "expected_format": "IPv4 or IPv6"
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(invalid_target_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 3: Validation Error
    print("\n✅ Validation Error Example:")
    
    validation_exception = main_module.raise_custom_exception(
        exception_type="validation",
        message="Email format is invalid",
        context={
            "field_name": "email",
            "field_value": "invalid-email",
            "expected_format": "user@example.com"
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(validation_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 4: Processing Error
    print("\n⚙️ Processing Error Example:")
    
    processing_exception = main_module.raise_custom_exception(
        exception_type="processing",
        message="Failed to process image file",
        context={
            "operation": "image_processing",
            "file_path": "/path/to/image.jpg",
            "file_size": "15MB",
            "supported_formats": ["JPEG", "PNG", "GIF"]
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(processing_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 5: Network Error
    print("\n🌐 Network Error Example:")
    
    network_exception = main_module.raise_custom_exception(
        exception_type="network",
        message="Connection refused by server",
        context={
            "url": "https://api.example.com/data",
            "status_code": 503,
            "retry_count": 3,
            "timeout": 10.0
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(network_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 6: Security Error
    print("\n🔒 Security Error Example:")
    
    security_exception = main_module.raise_custom_exception(
        exception_type="security",
        message="Invalid authentication token",
        context={
            "security_type": "authentication",
            "token_type": "JWT",
            "expired_at": "2024-01-01T00:00:00Z",
            "user_id": "12345"
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(security_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")
    
    # Example 7: Database Error
    print("\n💾 Database Error Example:")
    
    database_exception = main_module.raise_custom_exception(
        exception_type="database",
        message="Connection pool exhausted",
        context={
            "operation": "user_query",
            "database": "postgresql",
            "connection_count": 100,
            "max_connections": 50
        }
    )
    
    mapping_result = main_module.map_exception_user_friendly(database_exception)
    print(f"CLI Message: {mapping_result['cli_message']}")
    print(f"API Response: {json.dumps(mapping_result['api_response'], indent=2)}")

def show_custom_exceptions_benefits():
    """Show the benefits of custom exceptions with mapping."""
    
    benefits = {
        "custom_exceptions": [
            "Structured error hierarchy",
            "Comprehensive error context",
            "User-friendly error messages",
            "Error categorization and severity"
        ],
        "exception_mapping": [
            "CLI and API message formatting",
            "Consistent error presentation",
            "Localized error messages",
            "Context-aware suggestions"
        ],
        "user_experience": [
            "Clear and actionable error messages",
            "Helpful suggestions for resolution",
            "Appropriate error severity levels",
            "Consistent error formatting"
        ],
        "debugging": [
            "Detailed error context preservation",
            "Error correlation and tracking",
            "Structured error logging",
            "Performance impact monitoring"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate custom exceptions with mapping
    asyncio.run(demonstrate_custom_exceptions_mapping())
    
    benefits = show_custom_exceptions_benefits()
    
    print("\n🎯 Key Custom Exceptions Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Custom exceptions with user-friendly mapping completed successfully!") 