"""
Custom Exceptions
=================

Custom exception classes for better error handling.
"""

from typing import Optional


class ManualesHogarAIException(Exception):
    """Base exception for Manuales Hogar AI."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ConfigurationError(ManualesHogarAIException):
    """Configuration error."""

    pass


class DatabaseError(ManualesHogarAIException):
    """Database operation error."""

    pass


class CacheError(ManualesHogarAIException):
    """Cache operation error."""

    pass


class ExternalServiceError(ManualesHogarAIException):
    """External service error (e.g., OpenRouter API)."""

    pass


class ValidationError(ManualesHogarAIException):
    """Input validation error."""

    pass


class RateLimitError(ManualesHogarAIException):
    """Rate limit exceeded."""

    pass


class CircuitBreakerOpenError(ManualesHogarAIException):
    """Circuit breaker is open."""

    pass




