"""
Custom exceptions for KV Cache.

Provides specific exceptions for better error handling.
"""
from __future__ import annotations


class CacheError(Exception):
    """Base exception for all cache errors."""
    
    def __init__(self, message: str, error_code: str | None = None):
        """
        Initialize cache error.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class CacheMemoryError(CacheError):
    """Memory-related cache errors (OOM, allocation failures)."""
    
    def __init__(self, message: str = "Out of memory"):
        super().__init__(message, error_code="MEMORY_ERROR")


class CacheValidationError(CacheError):
    """Validation-related cache errors (invalid inputs, configs)."""
    
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, error_code="VALIDATION_ERROR")


class CacheDeviceError(CacheError):
    """Device-related cache errors (CUDA unavailable, device mismatch)."""
    
    def __init__(self, message: str = "Device error"):
        super().__init__(message, error_code="DEVICE_ERROR")


class CacheConfigError(CacheError):
    """Configuration-related cache errors."""
    
    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message, error_code="CONFIG_ERROR")


class CacheOperationError(CacheError):
    """Operation-related cache errors (put/get failures)."""
    
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message, error_code="OPERATION_ERROR")


class CacheStrategyError(CacheError):
    """Strategy-related cache errors."""
    
    def __init__(self, message: str = "Strategy error"):
        super().__init__(message, error_code="STRATEGY_ERROR")



