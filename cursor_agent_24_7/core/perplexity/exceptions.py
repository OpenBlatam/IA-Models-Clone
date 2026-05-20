"""
Perplexity Exceptions - Custom exceptions for Perplexity system
==============================================================

Custom exception classes for better error handling.
"""


class PerplexityError(Exception):
    """Base exception for Perplexity system."""
    
    def __init__(self, message: str = "An error occurred in the Perplexity system", details: dict = None):
        """
        Initialize Perplexity error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class QueryProcessingError(PerplexityError):
    """Error during query processing."""
    
    def __init__(self, message: str = "Error processing query", query: str = None, details: dict = None):
        """
        Initialize query processing error.
        
        Args:
            message: Error message
            query: The query that caused the error
            details: Additional error details
        """
        if query:
            message = f"{message}: {query[:100]}"
        super().__init__(message, details)
        self.query = query


class LLMProviderError(PerplexityError):
    """Error with LLM provider."""
    
    def __init__(self, message: str = "Error with LLM provider", provider: str = None, details: dict = None):
        """
        Initialize LLM provider error.
        
        Args:
            message: Error message
            provider: The LLM provider that caused the error
            details: Additional error details
        """
        if provider:
            message = f"{message} ({provider})"
        super().__init__(message, details)
        self.provider = provider


class ValidationError(PerplexityError):
    """Error during response validation."""
    
    def __init__(self, message: str = "Validation error", validation_errors: list = None, details: dict = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            validation_errors: List of validation errors
            details: Additional error details
        """
        if validation_errors:
            message = f"{message}: {', '.join(str(e) for e in validation_errors[:5])}"
        super().__init__(message, details)
        self.validation_errors = validation_errors or []


class CacheError(PerplexityError):
    """Error with cache operations."""
    
    def __init__(self, message: str = "Cache operation error", operation: str = None, details: dict = None):
        """
        Initialize cache error.
        
        Args:
            message: Error message
            operation: The cache operation that failed
            details: Additional error details
        """
        if operation:
            message = f"{message} during {operation}"
        super().__init__(message, details)
        self.operation = operation


class FormattingError(PerplexityError):
    """Error during response formatting."""
    
    def __init__(self, message: str = "Error formatting response", format_type: str = None, details: dict = None):
        """
        Initialize formatting error.
        
        Args:
            message: Error message
            format_type: The format type that failed
            details: Additional error details
        """
        if format_type:
            message = f"{message} ({format_type})"
        super().__init__(message, details)
        self.format_type = format_type


class CitationError(PerplexityError):
    """Error during citation processing."""
    
    def __init__(self, message: str = "Error processing citations", citation_count: int = None, details: dict = None):
        """
        Initialize citation error.
        
        Args:
            message: Error message
            citation_count: Number of citations being processed
            details: Additional error details
        """
        if citation_count is not None:
            message = f"{message} (processing {citation_count} citations)"
        super().__init__(message, details)
        self.citation_count = citation_count




