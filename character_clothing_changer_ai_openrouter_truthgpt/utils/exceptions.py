"""
Custom Exceptions
=================

Custom exception classes for the service.
"""


class ClothingChangeError(Exception):
    """Base exception for clothing change operations"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class WorkflowError(ClothingChangeError):
    """Exception for workflow-related errors"""
    pass


class ComfyUIError(ClothingChangeError):
    """Exception for ComfyUI-related errors"""
    pass


class OpenRouterError(ClothingChangeError):
    """Exception for OpenRouter-related errors"""
    pass


class TruthGPTError(ClothingChangeError):
    """Exception for TruthGPT-related errors"""
    pass


class ValidationError(ClothingChangeError):
    """Exception for validation errors"""
    pass


class BatchProcessingError(ClothingChangeError):
    """Exception for batch processing errors"""
    pass


class CacheError(ClothingChangeError):
    """Exception for cache-related errors"""
    pass


class RateLimitError(ClothingChangeError):
    """Exception for rate limit errors"""
    pass

