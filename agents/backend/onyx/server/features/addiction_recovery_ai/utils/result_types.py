"""
Result type utilities
Functional error handling with Result type
"""

from typing import TypeVar, Callable, Optional, Union
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E', bound=Exception)
U = TypeVar('U')


@dataclass
class Result:
    """
    Result type for functional error handling
    
    Either contains a value (Ok) or an error (Err)
    """
    value: Optional[T] = None
    error: Optional[Exception] = None
    is_ok: bool = True
    
    @classmethod
    def ok(cls, value: T) -> 'Result':
        """Create Ok result"""
        return cls(value=value, is_ok=True)
    
    @classmethod
    def err(cls, error: Exception) -> 'Result':
        """Create Err result"""
        return cls(error=error, is_ok=False)
    
    def map(self, func: Callable[[T], U]) -> 'Result':
        """Map over value if Ok"""
        if self.is_ok:
            try:
                return Result.ok(func(self.value))
            except Exception as e:
                return Result.err(e)
        return self
    
    def map_err(self, func: Callable[[Exception], Exception]) -> 'Result':
        """Map over error if Err"""
        if not self.is_ok:
            return Result.err(func(self.error))
        return self
    
    def unwrap(self) -> T:
        """Unwrap value, raise error if Err"""
        if self.is_ok:
            return self.value
        raise self.error
    
    def unwrap_or(self, default: T) -> T:
        """Unwrap value or return default"""
        if self.is_ok:
            return self.value
        return default
    
    def unwrap_or_else(self, func: Callable[[Exception], T]) -> T:
        """Unwrap value or compute from error"""
        if self.is_ok:
            return self.value
        return func(self.error)


def safe_call(func: Callable[[], T]) -> Result:
    """
    Safely call function and return Result
    
    Args:
        func: Function to call
    
    Returns:
        Result with value or error
    """
    try:
        return Result.ok(func())
    except Exception as e:
        return Result.err(e)


async def safe_call_async(func: Callable[[], T]) -> Result:
    """
    Safely call async function and return Result
    
    Args:
        func: Async function to call
    
    Returns:
        Result with value or error
    """
    try:
        return Result.ok(await func())
    except Exception as e:
        return Result.err(e)

