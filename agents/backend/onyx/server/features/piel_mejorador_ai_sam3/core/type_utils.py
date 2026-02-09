"""
Type Utilities for Piel Mejorador AI SAM3
=========================================

Type hints and utilities for better type safety.
"""

from typing import TypeVar, Generic, Optional, Dict, Any, List, Union, Callable, Awaitable
from pathlib import Path

T = TypeVar('T')
R = TypeVar('R')


class Result(Generic[T]):
    """Result type for operations that can fail."""
    
    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self.value = value
        self.error = error
        self.is_success = error is None
        self.is_error = error is not None
    
    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create successful result."""
        return cls(value=value, error=None)
    
    @classmethod
    def failure(cls, error: Exception) -> "Result[T]":
        """Create failed result."""
        return cls(value=None, error=error)
    
    def unwrap(self) -> T:
        """Unwrap result, raising error if failed."""
        if self.is_error:
            raise self.error
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Unwrap result or return default."""
        return self.value if self.is_success else default


# Type aliases for common patterns
FilePath = Union[str, Path]
ConfigDict = Dict[str, Any]
TaskDict = Dict[str, Any]
ServiceResult = Dict[str, Any]

# Async function types
AsyncFunction = Callable[..., Awaitable[Any]]
SyncFunction = Callable[..., Any]




