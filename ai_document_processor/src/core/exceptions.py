"""
Core Exceptions - Centralized Error Handling
===========================================

Custom exceptions for the AI Document Processor with proper error hierarchy.
"""

from typing import Optional, Dict, Any


class AIProcessorError(Exception):
    """Base exception for all AI Document Processor errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ProcessingError(AIProcessorError):
    """Raised when document processing fails."""
    pass


class ValidationError(AIProcessorError):
    """Raised when input validation fails."""
    pass


class CacheError(AIProcessorError):
    """Raised when cache operations fail."""
    pass


class ConfigurationError(AIProcessorError):
    """Raised when configuration is invalid."""
    pass


class FileError(AIProcessorError):
    """Raised when file operations fail."""
    pass


class AIServiceError(AIProcessorError):
    """Raised when AI service operations fail."""
    pass


class TransformError(AIProcessorError):
    """Raised when document transformation fails."""
    pass


class NetworkError(AIProcessorError):
    """Raised when network operations fail."""
    pass


class TimeoutError(AIProcessorError):
    """Raised when operations timeout."""
    pass


class ResourceError(AIProcessorError):
    """Raised when system resources are insufficient."""
    pass


class SecurityError(AIProcessorError):
    """Raised when security checks fail."""
    pass

















