"""
Custom Exceptions
=================
"""

from typing import Optional
from .error_codes import ErrorCode, get_error_message


class DogTrainingException(Exception):
    """Base exception for dog training coaching."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        details: Optional[dict] = None
    ):
        self.message = message
        self.error_code = error_code or ErrorCode.UNEXPECTED_ERROR
        self.details = details or {}
        super().__init__(self.message)


class OpenRouterException(DogTrainingException):
    """Exception for OpenRouter API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        details: Optional[dict] = None
    ):
        super().__init__(
            message,
            error_code or ErrorCode.OPENROUTER_ERROR,
            details
        )


class ValidationException(DogTrainingException):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        field: Optional[str] = None,
        details: Optional[dict] = None
    ):
        self.field = field
        super().__init__(
            message,
            error_code or ErrorCode.VALIDATION_ERROR,
            details or {}
        )


class ServiceException(DogTrainingException):
    """Exception for service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        details: Optional[dict] = None
    ):
        super().__init__(
            message,
            error_code or ErrorCode.SERVICE_ERROR,
            details
        )

