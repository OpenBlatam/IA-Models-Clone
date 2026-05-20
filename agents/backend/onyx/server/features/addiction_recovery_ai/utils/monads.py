"""
Monadic utilities
Monad implementations for functional programming
"""

from typing import TypeVar, Callable, Optional, Any
from functools import wraps

T = TypeVar('T')
U = TypeVar('U')


class Maybe:
    """
    Maybe monad for handling optional values
    """
    
    def __init__(self, value: Optional[T] = None):
        self.value = value
        self.is_just = value is not None
    
    @classmethod
    def just(cls, value: T) -> 'Maybe':
        """Create Just (value present)"""
        return cls(value=value)
    
    @classmethod
    def nothing(cls) -> 'Maybe':
        """Create Nothing (no value)"""
        return cls()
    
    def map(self, func: Callable[[T], U]) -> 'Maybe':
        """Map over value if Just"""
        if self.is_just:
            try:
                return Maybe.just(func(self.value))
            except Exception:
                return Maybe.nothing()
        return self
    
    def flat_map(self, func: Callable[[T], 'Maybe']) -> 'Maybe':
        """FlatMap (bind) for Maybe"""
        if self.is_just:
            return func(self.value)
        return self
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Maybe':
        """Filter Maybe by predicate"""
        if self.is_just and predicate(self.value):
            return self
        return Maybe.nothing()
    
    def get_or_else(self, default: T) -> T:
        """Get value or return default"""
        return self.value if self.is_just else default
    
    def or_else(self, default: 'Maybe') -> 'Maybe':
        """Return self if Just, else default"""
        return self if self.is_just else default


class Either:
    """
    Either monad for handling success or error
    """
    
    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self.value = value
        self.error = error
        self.is_right = error is None
    
    @classmethod
    def right(cls, value: T) -> 'Either':
        """Create Right (success)"""
        return cls(value=value)
    
    @classmethod
    def left(cls, error: Exception) -> 'Either':
        """Create Left (error)"""
        return cls(error=error)
    
    def map(self, func: Callable[[T], U]) -> 'Either':
        """Map over value if Right"""
        if self.is_right:
            try:
                return Either.right(func(self.value))
            except Exception as e:
                return Either.left(e)
        return self
    
    def flat_map(self, func: Callable[[T], 'Either']) -> 'Either':
        """FlatMap (bind) for Either"""
        if self.is_right:
            return func(self.value)
        return self
    
    def map_left(self, func: Callable[[Exception], Exception]) -> 'Either':
        """Map over error if Left"""
        if not self.is_right:
            return Either.left(func(self.error))
        return self
    
    def get_or_else(self, default: T) -> T:
        """Get value or return default"""
        return self.value if self.is_right else default
    
    def fold(self, left_func: Callable[[Exception], U], right_func: Callable[[T], U]) -> U:
        """Fold Either with two functions"""
        if self.is_right:
            return right_func(self.value)
        return left_func(self.error)


def maybe_decorator(func: Callable) -> Callable:
    """
    Decorator to wrap function return in Maybe
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function returning Maybe
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return Maybe.just(result) if result is not None else Maybe.nothing()
        except Exception:
            return Maybe.nothing()
    
    return wrapper


def either_decorator(func: Callable) -> Callable:
    """
    Decorator to wrap function return in Either
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function returning Either
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return Either.right(result)
        except Exception as e:
            return Either.left(e)
    
    return wrapper

