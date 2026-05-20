from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import traceback
import sys
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, ConfigDict, ValidationError
from pydantic.types import conint, confloat, constr, EmailStr, HttpUrl
import numpy as np
from pathlib import Path
from enum import Enum
from functools import wraps
import time
import random
from typing import Any, List, Dict, Optional
"""
Error Handling and Validation - Complete Patterns
===============================================

This file demonstrates comprehensive error handling and validation patterns:
- Custom error types and error factories
- Validation patterns with Pydantic
- Error handling middleware
- Guard clauses and early returns
- Comprehensive logging and monitoring
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Error Types
    "ValidationError",
    "ProcessingError",
    "NetworkError",
    "SecurityError",
    "DatabaseError",
    
    # Error Factories
    "ErrorFactory",
    "ValidationErrorFactory",
    "ProcessingErrorFactory",
    
    # Validation Patterns
    "ValidationPatterns",
    "InputValidator",
    "DataValidator",
    "SchemaValidator",
    
    # Error Handling
    "ErrorHandler",
    "ErrorMiddleware",
    "ErrorLogger",
    "ErrorMonitor",
    
    # Common utilities
    "ValidationResult",
    "ErrorConfig",
    "ErrorType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class ValidationResult(BaseModel):
    """Pydantic model for validation results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_valid: bool = Field(description="Whether validation was successful")
    validation_type: str = Field(description="Type of validation performed")
    data: Optional[Any] = Field(default=None, description="Validated data")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")
    execution_time: Optional[float] = Field(default=None, description="Validation time in seconds")

class ErrorConfig(BaseModel):
    """Pydantic model for error handling configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    log_errors: bool = Field(default=True, description="Log errors")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="ERROR",
        description="Log level for errors"
    )
    include_stack_trace: bool = Field(default=True, description="Include stack trace in errors")
    max_error_length: conint(ge=100, le=10000) = Field(default=1000, description="Maximum error message length")
    retry_on_error: bool = Field(default=False, description="Retry on error")
    max_retries: conint(ge=0, le=10) = Field(default=3, description="Maximum retries")
    error_timeout: confloat(gt=0.0) = Field(default=30.0, description="Error handling timeout")

class ErrorType(BaseModel):
    """Pydantic model for error type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    error_type: constr(strip_whitespace=True) = Field(
        pattern=r"^(validation|processing|network|security|database|system)$"
    )
    description: Optional[str] = Field(default=None)
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")

# ============================================================================
# CUSTOM ERROR TYPES
# ============================================================================

class ValidationError(Exception):
    """Custom validation error with comprehensive details."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_type: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ):
        
    """__init__ function."""
self.message = message
        self.field_name = field_name
        self.field_value = field_value
        self.validation_type = validation_type
        self.suggestions = suggestions or []
        self.timestamp = time.time()
        
        # Build error message
        error_msg = f"Validation Error: {message}"
        if field_name:
            error_msg += f" (Field: {field_name})"
        if field_value is not None:
            error_msg += f" (Value: {field_value})"
        if validation_type:
            error_msg += f" (Type: {validation_type})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": "ValidationError",
            "message": self.message,
            "field_name": self.field_name,
            "field_value": self.field_value,
            "validation_type": self.validation_type,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }

class ProcessingError(Exception):
    """Custom processing error with operation details."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        operation_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 3
    ):
        
    """__init__ function."""
self.message = message
        self.operation = operation
        self.operation_data = operation_data or {}
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.timestamp = time.time()
        
        # Build error message
        error_msg = f"Processing Error: {message}"
        if operation:
            error_msg += f" (Operation: {operation})"
        if retry_count > 0:
            error_msg += f" (Retry: {retry_count}/{max_retries})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": "ProcessingError",
            "message": self.message,
            "operation": self.operation,
            "operation_data": self.operation_data,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timestamp": self.timestamp
        }

class NetworkError(Exception):
    """Custom network error with connection details."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        retry_after: Optional[float] = None
    ):
        
    """__init__ function."""
self.message = message
        self.url = url
        self.status_code = status_code
        self.response_time = response_time
        self.retry_after = retry_after
        self.timestamp = time.time()
        
        # Build error message
        error_msg = f"Network Error: {message}"
        if url:
            error_msg += f" (URL: {url})"
        if status_code:
            error_msg += f" (Status: {status_code})"
        if response_time:
            error_msg += f" (Response Time: {response_time:.2f}s)"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": "NetworkError",
            "message": self.message,
            "url": self.url,
            "status_code": self.status_code,
            "response_time": self.response_time,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp
        }

class SecurityError(Exception):
    """Custom security error with security details."""
    
    def __init__(
        self,
        message: str,
        security_level: Optional[str] = None,
        threat_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        
    """__init__ function."""
self.message = message
        self.security_level = security_level
        self.threat_type = threat_type
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.timestamp = time.time()
        
        # Build error message
        error_msg = f"Security Error: {message}"
        if security_level:
            error_msg += f" (Level: {security_level})"
        if threat_type:
            error_msg += f" (Threat: {threat_type})"
        if ip_address:
            error_msg += f" (IP: {ip_address})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": "SecurityError",
            "message": self.message,
            "security_level": self.security_level,
            "threat_type": self.threat_type,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp
        }

class DatabaseError(Exception):
    """Custom database error with database details."""
    
    def __init__(
        self,
        message: str,
        table_name: Optional[str] = None,
        operation: Optional[str] = None,
        query: Optional[str] = None,
        connection_id: Optional[str] = None
    ):
        
    """__init__ function."""
self.message = message
        self.table_name = table_name
        self.operation = operation
        self.query = query
        self.connection_id = connection_id
        self.timestamp = time.time()
        
        # Build error message
        error_msg = f"Database Error: {message}"
        if table_name:
            error_msg += f" (Table: {table_name})"
        if operation:
            error_msg += f" (Operation: {operation})"
        if connection_id:
            error_msg += f" (Connection: {connection_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": "DatabaseError",
            "message": self.message,
            "table_name": self.table_name,
            "operation": self.operation,
            "query": self.query,
            "connection_id": self.connection_id,
            "timestamp": self.timestamp
        }

# ============================================================================
# ERROR FACTORIES
# ============================================================================

class ErrorFactory:
    """Error factory for creating standardized errors."""
    
    @staticmethod
    def create_validation_error(
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_type: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> ValidationError:
        """Create a validation error."""
        return ValidationError(
            message=message,
            field_name=field_name,
            field_value=field_value,
            validation_type=validation_type,
            suggestions=suggestions
        )
    
    @staticmethod
    def create_processing_error(
        message: str,
        operation: Optional[str] = None,
        operation_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> ProcessingError:
        """Create a processing error."""
        return ProcessingError(
            message=message,
            operation=operation,
            operation_data=operation_data,
            retry_count=retry_count,
            max_retries=max_retries
        )
    
    @staticmethod
    def create_network_error(
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        retry_after: Optional[float] = None
    ) -> NetworkError:
        """Create a network error."""
        return NetworkError(
            message=message,
            url=url,
            status_code=status_code,
            response_time=response_time,
            retry_after=retry_after
        )
    
    @staticmethod
    def create_security_error(
        message: str,
        security_level: Optional[str] = None,
        threat_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityError:
        """Create a security error."""
        return SecurityError(
            message=message,
            security_level=security_level,
            threat_type=threat_type,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @staticmethod
    def create_database_error(
        message: str,
        table_name: Optional[str] = None,
        operation: Optional[str] = None,
        query: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> DatabaseError:
        """Create a database error."""
        return DatabaseError(
            message=message,
            table_name=table_name,
            operation=operation,
            query=query,
            connection_id=connection_id
        )

class ValidationErrorFactory:
    """Factory for creating validation-specific errors."""
    
    @staticmethod
    def field_required(field_name: str) -> ValidationError:
        """Create error for required field."""
        return ErrorFactory.create_validation_error(
            message=f"Field '{field_name}' is required",
            field_name=field_name,
            validation_type="required_field"
        )
    
    @staticmethod
    def invalid_format(field_name: str, field_value: Any, expected_format: str) -> ValidationError:
        """Create error for invalid format."""
        return ErrorFactory.create_validation_error(
            message=f"Invalid format for field '{field_name}'. Expected: {expected_format}",
            field_name=field_name,
            field_value=field_value,
            validation_type="invalid_format",
            suggestions=[f"Ensure the field follows the format: {expected_format}"]
        )
    
    @staticmethod
    def value_out_of_range(field_name: str, field_value: Any, min_value: Any, max_value: Any) -> ValidationError:
        """Create error for value out of range."""
        return ErrorFactory.create_validation_error(
            message=f"Value for field '{field_name}' is out of range. Must be between {min_value} and {max_value}",
            field_name=field_name,
            field_value=field_value,
            validation_type="out_of_range",
            suggestions=[f"Provide a value between {min_value} and {max_value}"]
        )
    
    @staticmethod
    def invalid_email(email: str) -> ValidationError:
        """Create error for invalid email."""
        return ErrorFactory.create_validation_error(
            message=f"Invalid email format: {email}",
            field_name="email",
            field_value=email,
            validation_type="invalid_email",
            suggestions=["Use a valid email format (e.g., user@example.com)"]
        )
    
    @staticmethod
    def password_too_weak(password: str) -> ValidationError:
        """Create error for weak password."""
        return ErrorFactory.create_validation_error(
            message="Password is too weak",
            field_name="password",
            validation_type="weak_password",
            suggestions=[
                "Use at least 8 characters",
                "Include uppercase and lowercase letters",
                "Include numbers and special characters"
            ]
        )

class ProcessingErrorFactory:
    """Factory for creating processing-specific errors."""
    
    @staticmethod
    def timeout_error(operation: str, timeout: float) -> ProcessingError:
        """Create timeout error."""
        return ErrorFactory.create_processing_error(
            message=f"Operation '{operation}' timed out after {timeout} seconds",
            operation=operation,
            operation_data={"timeout": timeout}
        )
    
    @staticmethod
    def resource_not_found(resource_type: str, resource_id: str) -> ProcessingError:
        """Create resource not found error."""
        return ErrorFactory.create_processing_error(
            message=f"{resource_type} with ID '{resource_id}' not found",
            operation="find_resource",
            operation_data={"resource_type": resource_type, "resource_id": resource_id}
        )
    
    @staticmethod
    def permission_denied(operation: str, user_id: str) -> ProcessingError:
        """Create permission denied error."""
        return ErrorFactory.create_processing_error(
            message=f"Permission denied for operation '{operation}'",
            operation=operation,
            operation_data={"user_id": user_id}
        )

# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

class ValidationPatterns:
    """Validation patterns with comprehensive error handling."""
    
    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any],
        required_fields: List[str]
    ) -> ValidationResult:
        """Validate required fields."""
        try:
            start_time = time.time()
            errors = []
            
            for field in required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Field '{field}' is required")
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_type="required_fields",
                data=data if len(errors) == 0 else None,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                validation_type="required_fields",
                errors=[f"Validation error: {str(exc)}"],
                execution_time=None
            )
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email format."""
        try:
            start_time = time.time()
            
            # Basic email validation
            if not email or '@' not in email or '.' not in email:
                raise ValidationErrorFactory.invalid_email(email)
            
            # Check for valid domain
            domain = email.split('@')[1]
            if len(domain) < 3:
                raise ValidationErrorFactory.invalid_email(email)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                is_valid=True,
                validation_type="email",
                data=email,
                execution_time=execution_time
            )
            
        except ValidationError as ve:
            return ValidationResult(
                is_valid=False,
                validation_type="email",
                data=email,
                errors=[ve.message],
                execution_time=time.time() - start_time
            )
    
    @staticmethod
    def validate_password_strength(password: str) -> ValidationResult:
        """Validate password strength."""
        try:
            start_time = time.time()
            errors = []
            warnings = []
            
            # Check length
            if len(password) < 8:
                errors.append("Password must be at least 8 characters long")
            
            # Check for uppercase
            if not any(c.isupper() for c in password):
                warnings.append("Consider including uppercase letters")
            
            # Check for lowercase
            if not any(c.islower() for c in password):
                warnings.append("Consider including lowercase letters")
            
            # Check for numbers
            if not any(c.isdigit() for c in password):
                warnings.append("Consider including numbers")
            
            # Check for special characters
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                warnings.append("Consider including special characters")
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_type="password_strength",
                data=password if len(errors) == 0 else None,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                validation_type="password_strength",
                data=password,
                errors=[f"Password validation error: {str(exc)}"],
                execution_time=None
            )
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float],
        min_value: Union[int, float],
        max_value: Union[int, float],
        field_name: str = "value"
    ) -> ValidationResult:
        """Validate numeric value is within range."""
        try:
            start_time = time.time()
            
            if value < min_value or value > max_value:
                raise ValidationErrorFactory.value_out_of_range(
                    field_name, value, min_value, max_value
                )
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                is_valid=True,
                validation_type="numeric_range",
                data=value,
                metadata={"min_value": min_value, "max_value": max_value},
                execution_time=execution_time
            )
            
        except ValidationError as ve:
            return ValidationResult(
                is_valid=False,
                validation_type="numeric_range",
                data=value,
                errors=[ve.message],
                metadata={"min_value": min_value, "max_value": max_value},
                execution_time=time.time() - start_time
            )

class InputValidator:
    """Input validation with comprehensive error handling."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def validate_user_input(
        self,
        user_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate user input with comprehensive checks."""
        try:
            start_time = time.time()
            errors = []
            warnings = []
            
            # Validate required fields
            required_fields = ["username", "email", "password"]
            required_result = ValidationPatterns.validate_required_fields(user_data, required_fields)
            if not required_result.is_valid:
                errors.extend(required_result.errors)
            
            # Validate email if provided
            if "email" in user_data and user_data["email"]:
                email_result = ValidationPatterns.validate_email(user_data["email"])
                if not email_result.is_valid:
                    errors.extend(email_result.errors)
                else:
                    warnings.extend(email_result.warnings)
            
            # Validate password if provided
            if "password" in user_data and user_data["password"]:
                password_result = ValidationPatterns.validate_password_strength(user_data["password"])
                if not password_result.is_valid:
                    errors.extend(password_result.errors)
                else:
                    warnings.extend(password_result.warnings)
            
            # Validate username format
            if "username" in user_data and user_data["username"]:
                username = user_data["username"]
                if len(username) < 3 or len(username) > 50:
                    errors.append("Username must be between 3 and 50 characters")
                if not username.replace('_', '').isalnum():
                    errors.append("Username must contain only letters, numbers, and underscores")
            
            execution_time = time.time() - start_time
            
            if self.config.log_errors and errors:
                self.logger.error(f"User input validation failed: {errors}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_type="user_input",
                data=user_data if len(errors) == 0 else None,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as exc:
            if self.config.log_errors:
                self.logger.error(f"User input validation error: {exc}")
            
            return ValidationResult(
                is_valid=False,
                validation_type="user_input",
                data=user_data,
                errors=[f"Validation error: {str(exc)}"],
                execution_time=None
            )

class DataValidator:
    """Data validation with comprehensive error handling."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def validate_post_data(
        self,
        post_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate post data with comprehensive checks."""
        try:
            start_time = time.time()
            errors = []
            warnings = []
            
            # Validate required fields
            required_fields = ["content", "author_id"]
            required_result = ValidationPatterns.validate_required_fields(post_data, required_fields)
            if not required_result.is_valid:
                errors.extend(required_result.errors)
            
            # Validate content length
            if "content" in post_data and post_data["content"]:
                content = post_data["content"]
                if len(content) < 1:
                    errors.append("Content cannot be empty")
                elif len(content) > 10000:
                    errors.append("Content must be less than 10,000 characters")
            
            # Validate author_id
            if "author_id" in post_data and post_data["author_id"]:
                try:
                    author_id = int(post_data["author_id"])
                    if author_id <= 0:
                        errors.append("Author ID must be a positive integer")
                except (ValueError, TypeError):
                    errors.append("Author ID must be a valid integer")
            
            # Validate tags
            if "tags" in post_data and post_data["tags"]:
                tags = post_data["tags"]
                if not isinstance(tags, list):
                    errors.append("Tags must be a list")
                elif len(tags) > 10:
                    errors.append("Maximum 10 tags allowed")
                else:
                    for tag in tags:
                        if not isinstance(tag, str) or len(tag) < 1 or len(tag) > 50:
                            errors.append("Each tag must be a string between 1 and 50 characters")
            
            execution_time = time.time() - start_time
            
            if self.config.log_errors and errors:
                self.logger.error(f"Post data validation failed: {errors}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_type="post_data",
                data=post_data if len(errors) == 0 else None,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as exc:
            if self.config.log_errors:
                self.logger.error(f"Post data validation error: {exc}")
            
            return ValidationResult(
                is_valid=False,
                validation_type="post_data",
                data=post_data,
                errors=[f"Validation error: {str(exc)}"],
                execution_time=None
            )

# ============================================================================
# ERROR HANDLING
# ============================================================================

class ErrorHandler:
    """Comprehensive error handler with logging and monitoring."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.error_types = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle error with comprehensive logging and monitoring."""
        try:
            self.error_count += 1
            error_type = type(error).__name__
            
            # Track error types
            if error_type not in self.error_types:
                self.error_types[error_type] = 0
            self.error_types[error_type] += 1
            
            # Build error response
            error_response = {
                "error_type": error_type,
                "message": str(error),
                "timestamp": time.time(),
                "error_count": self.error_count
            }
            
            # Add context if provided
            if context:
                error_response["context"] = context
            
            # Add stack trace if configured
            if self.config.include_stack_trace:
                error_response["stack_trace"] = traceback.format_exc()
            
            # Truncate message if too long
            if len(error_response["message"]) > self.config.max_error_length:
                error_response["message"] = error_response["message"][:self.config.max_error_length] + "..."
            
            # Log error
            if self.config.log_errors:
                log_level = getattr(logging, self.config.log_level.upper())
                self.logger.log(log_level, f"Error handled: {error_response}")
            
            return error_response
            
        except Exception as exc:
            # Fallback error handling
            return {
                "error_type": "ErrorHandlerError",
                "message": f"Error in error handler: {str(exc)}",
                "timestamp": time.time(),
                "original_error": str(error)
            }
    
    def handle_validation_error(
        self,
        error: ValidationError,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle validation error specifically."""
        context = {
            "validation_type": error.validation_type,
            "field_name": error.field_name,
            "field_value": error.field_value,
            "suggestions": error.suggestions
        }
        
        if data:
            context["data"] = data
        
        return self.handle_error(error, context)
    
    def handle_processing_error(
        self,
        error: ProcessingError,
        operation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle processing error specifically."""
        context = {
            "operation": error.operation,
            "retry_count": error.retry_count,
            "max_retries": error.max_retries
        }
        
        if operation_data:
            context["operation_data"] = operation_data
        
        return self.handle_error(error, context)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": self.error_count,
            "error_types": self.error_types,
            "most_common_error": max(self.error_types.items(), key=lambda x: x[1]) if self.error_types else None
        }

class ErrorMiddleware:
    """Middleware for handling errors in async operations."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger(__name__)
    
    async def handle_async_operation(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Handle async operation with error handling."""
        try:
            start_time = time.time()
            
            # Execute operation
            result = await operation(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Log successful operation
            if self.config.log_errors:
                self.logger.info(f"Operation completed successfully in {execution_time:.2f}s")
            
            return result, None
            
        except Exception as exc:
            # Handle error
            error_response = self.error_handler.handle_error(exc, {
                "operation": operation.__name__,
                "args": args,
                "kwargs": kwargs
            })
            
            return None, error_response
    
    def retry_on_error(
        self,
        operation: Callable,
        max_retries: Optional[int] = None
    ) -> Callable:
        """Decorator for retrying operations on error."""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        async def retry_wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await operation(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    
                    if attempt < max_retries:
                        # Wait before retry (exponential backoff)
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        
                        if self.config.log_errors:
                            self.logger.warning(f"Retry {attempt + 1}/{max_retries} for {operation.__name__}")
                    else:
                        # Final attempt failed
                        if self.config.log_errors:
                            self.logger.error(f"Operation {operation.__name__} failed after {max_retries} retries")
            
            # All retries failed
            raise last_error
        
        return retry_wrapper

class ErrorLogger:
    """Comprehensive error logging with structured output."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None
    ) -> None:
        """Log error with structured information."""
        try:
            log_level = getattr(logging, (level or self.config.log_level).upper())
            
            # Build log message
            log_data = {
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": time.time()
            }
            
            if context:
                log_data["context"] = context
            
            if self.config.include_stack_trace:
                log_data["stack_trace"] = traceback.format_exc()
            
            # Log the error
            self.logger.log(log_level, f"Error logged: {log_data}")
            
        except Exception as exc:
            # Fallback logging
            self.logger.error(f"Error in error logger: {exc}")
    
    def log_validation_error(
        self,
        error: ValidationError,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log validation error specifically."""
        context = {
            "validation_type": error.validation_type,
            "field_name": error.field_name,
            "field_value": error.field_value,
            "suggestions": error.suggestions
        }
        
        if data:
            context["data"] = data
        
        self.log_error(error, context, "ERROR")
    
    def log_processing_error(
        self,
        error: ProcessingError,
        operation_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log processing error specifically."""
        context = {
            "operation": error.operation,
            "retry_count": error.retry_count,
            "max_retries": error.max_retries
        }
        
        if operation_data:
            context["operation_data"] = operation_data
        
        self.log_error(error, context, "ERROR")

# ============================================================================
# MAIN ERROR HANDLING MODULE
# ============================================================================

class MainErrorHandlingModule:
    """Main error handling module with proper imports and exports."""
    
    # Define main exports
    __all__ = [
        # Error Types
        "ValidationError",
        "ProcessingError",
        "NetworkError",
        "SecurityError",
        "DatabaseError",
        
        # Error Factories
        "ErrorFactory",
        "ValidationErrorFactory",
        "ProcessingErrorFactory",
        
        # Validation Patterns
        "ValidationPatterns",
        "InputValidator",
        "DataValidator",
        
        # Error Handling
        "ErrorHandler",
        "ErrorMiddleware",
        "ErrorLogger",
        
        # Common utilities
        "ValidationResult",
        "ErrorConfig",
        "ErrorType",
        
        # Main functions
        "handle_validation_error",
        "handle_processing_error",
        "validate_user_input",
        "validate_post_data"
    ]
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.error_handler = ErrorHandler(config)
        self.error_middleware = ErrorMiddleware(config)
        self.error_logger = ErrorLogger(config)
        self.input_validator = InputValidator(config)
        self.data_validator = DataValidator(config)
    
    async def handle_validation_error(
        self,
        error: ValidationError,
        data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Handle validation error with comprehensive logging."""
        try:
            # Log the error
            self.error_logger.log_validation_error(error, data)
            
            # Handle the error
            error_response = self.error_handler.handle_validation_error(error, data)
            
            return ValidationResult(
                is_valid=False,
                validation_type="error_handling",
                errors=[error_response["message"]],
                metadata=error_response
            )
            
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                validation_type="error_handling",
                errors=[f"Error handling failed: {str(exc)}"],
                execution_time=None
            )
    
    async def handle_processing_error(
        self,
        error: ProcessingError,
        operation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle processing error with comprehensive logging."""
        try:
            # Log the error
            self.error_logger.log_processing_error(error, operation_data)
            
            # Handle the error
            error_response = self.error_handler.handle_processing_error(error, operation_data)
            
            return error_response
            
        except Exception as exc:
            return {
                "error_type": "ErrorHandlingError",
                "message": f"Error handling failed: {str(exc)}",
                "timestamp": time.time()
            }
    
    async def validate_user_input(
        self,
        user_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate user input with comprehensive error handling."""
        try:
            return await self.input_validator.validate_user_input(user_data)
            
        except Exception as exc:
            return await self.handle_validation_error(
                ValidationErrorFactory.create_validation_error(
                    f"User input validation failed: {str(exc)}"
                ),
                user_data
            )
    
    async def validate_post_data(
        self,
        post_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate post data with comprehensive error handling."""
        try:
            return await self.data_validator.validate_post_data(post_data)
            
        except Exception as exc:
            return await self.handle_validation_error(
                ValidationErrorFactory.create_validation_error(
                    f"Post data validation failed: {str(exc)}"
                ),
                post_data
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_error_handling_validation():
    """Demonstrate error handling and validation patterns."""
    
    print("🛡️ Demonstrating Error Handling and Validation Patterns")
    print("=" * 60)
    
    # Initialize error handling module
    config = ErrorConfig(
        log_errors=True,
        log_level="ERROR",
        include_stack_trace=True,
        max_error_length=1000,
        retry_on_error=True,
        max_retries=3
    )
    
    main_module = MainErrorHandlingModule(config)
    
    # Example 1: Validation error handling
    print("\n✅ Validation Error Handling:")
    user_data = {
        "username": "john_doe",
        "email": "invalid-email",
        "password": "weak"
    }
    
    validation_result = await main_module.validate_user_input(user_data)
    print(f"Validation successful: {validation_result.is_valid}")
    if not validation_result.is_valid:
        print(f"Errors: {validation_result.errors}")
        print(f"Warnings: {validation_result.warnings}")
    
    # Example 2: Processing error handling
    print("\n⚙️ Processing Error Handling:")
    
    async def failing_operation():
        
    """failing_operation function."""
raise ProcessingErrorFactory.resource_not_found("user", "123")
    
    result, error = await main_module.error_middleware.handle_async_operation(failing_operation)
    if error:
        print(f"Processing error: {error['message']}")
        print(f"Error type: {error['error_type']}")
    
    # Example 3: Custom error creation
    print("\n🔧 Custom Error Creation:")
    
    # Create validation error
    validation_error = ValidationErrorFactory.invalid_email("test@invalid")
    print(f"Validation error: {validation_error.message}")
    print(f"Error details: {validation_error.to_dict()}")
    
    # Create processing error
    processing_error = ProcessingErrorFactory.timeout_error("database_query", 30.0)
    print(f"Processing error: {processing_error.message}")
    print(f"Error details: {processing_error.to_dict()}")
    
    # Example 4: Error statistics
    print("\n📊 Error Statistics:")
    stats = main_module.error_handler.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print(f"Error types: {stats['error_types']}")
    
    # Example 5: Retry mechanism
    print("\n🔄 Retry Mechanism:")
    
    @main_module.error_middleware.retry_on_error
    async def unreliable_operation():
        
    """unreliable_operation function."""
        if random.random() < 0.7:  # 70% chance of failure
            raise ProcessingError("Operation failed", "test_operation")
        return "Success"
    
    try:
        result = await unreliable_operation()
        print(f"Operation result: {result}")
    except Exception as exc:
        print(f"Operation failed after retries: {exc}")

def show_error_handling_benefits():
    """Show the benefits of error handling and validation."""
    
    benefits = {
        "error_handling": [
            "Custom error types for specific scenarios",
            "Error factories for standardized error creation",
            "Comprehensive error logging and monitoring",
            "Retry mechanisms with exponential backoff"
        ],
        "validation": [
            "Input validation with detailed error messages",
            "Data validation with comprehensive checks",
            "Schema validation for API requests/responses",
            "Custom validators for business logic"
        ],
        "logging": [
            "Structured error logging",
            "Error statistics and monitoring",
            "Configurable log levels and formats",
            "Stack trace inclusion for debugging"
        ],
        "recovery": [
            "Graceful error recovery",
            "Retry mechanisms for transient failures",
            "Error context preservation",
            "User-friendly error messages"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate error handling and validation
    asyncio.run(demonstrate_error_handling_validation())
    
    benefits = show_error_handling_benefits()
    
    print("\n🎯 Key Error Handling and Validation Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Error handling and validation patterns completed successfully!") 