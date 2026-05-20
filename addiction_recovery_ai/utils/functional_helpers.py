"""
Functional programming helpers
Utilities for functional programming patterns
"""

from typing import Callable, List, Any, TypeVar, Optional
from functools import reduce

T = TypeVar('T')
U = TypeVar('U')


def map_function(func: Callable[[T], U]) -> Callable[[List[T]], List[U]]:
    """
    Create a map function for lists
    
    Args:
        func: Function to map over list
    
    Returns:
        Function that maps over lists
    """
    def mapper(items: List[T]) -> List[U]:
        return [func(item) for item in items]
    
    return mapper


def filter_function(predicate: Callable[[T], bool]) -> Callable[[List[T]], List[T]]:
    """
    Create a filter function for lists
    
    Args:
        predicate: Function that returns True for items to keep
    
    Returns:
        Function that filters lists
    """
    def filterer(items: List[T]) -> List[T]:
        return [item for item in items if predicate(item)]
    
    return filterer


def reduce_function(
    func: Callable[[U, T], U],
    initial: Optional[U] = None
) -> Callable[[List[T]], U]:
    """
    Create a reduce function for lists
    
    Args:
        func: Reduction function
        initial: Initial value (optional)
    
    Returns:
        Function that reduces lists
    """
    def reducer(items: List[T]) -> U:
        if initial is not None:
            return reduce(func, items, initial)
        return reduce(func, items)
    
    return reducer


def identity(value: T) -> T:
    """
    Identity function - returns value unchanged
    
    Args:
        value: Any value
    
    Returns:
        Same value
    """
    return value


def constant(value: T) -> Callable[[Any], T]:
    """
    Create a constant function
    
    Args:
        value: Value to return
    
    Returns:
        Function that always returns the same value
    """
    def constant_func(*args, **kwargs) -> T:
        return value
    
    return constant_func


def maybe(func: Callable[[T], U], default: U = None) -> Callable[[Optional[T]], Optional[U]]:
    """
    Create a maybe function (handles None)
    
    Args:
        func: Function to apply
        default: Default value if input is None
    
    Returns:
        Function that handles None values
    """
    def maybe_func(value: Optional[T]) -> Optional[U]:
        if value is None:
            return default
        return func(value)
    
    return maybe_func

