"""
Custom Exceptions
================
Custom exception classes for better error handling.

Provides a hierarchy of exceptions for different error types,
enabling more precise error handling and better error messages.
"""


class BurnoutPreventionError(Exception):
    """
    Base exception for burnout prevention errors.
    
    All custom exceptions inherit from this class, allowing
    catch-all error handling when needed.
    """
    pass


class ValidationError(BurnoutPreventionError):
    """
    Raised when validation fails.
    
    Used for input validation errors, data format issues,
    and constraint violations.
    """
    pass


class APIError(BurnoutPreventionError):
    """
    Raised when API calls fail.
    
    Used for external API errors, network issues,
    and API response parsing failures.
    """
    pass


class CacheError(BurnoutPreventionError):
    """
    Raised when cache operations fail.
    
    Used for cache read/write errors and cache
    configuration issues.
    """
    pass


class ConfigurationError(BurnoutPreventionError):
    """
    Raised when configuration is invalid.
    
    Used for missing or invalid configuration values,
    environment variable issues, and setup problems.
    """
    pass


class ProcessingError(BurnoutPreventionError):
    """
    Raised when processing fails.
    
    Used for background processing errors, queue
    operations, and continuous processing issues.
    """
    pass

