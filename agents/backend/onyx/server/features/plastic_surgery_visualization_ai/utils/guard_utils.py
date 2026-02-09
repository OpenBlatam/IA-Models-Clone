"""Guard and assertion utilities."""

from typing import Any, Callable, Optional
from functools import wraps


def require(condition: bool, message: str = "Requirement not met") -> None:
    """
    Require condition to be true.
    
    Args:
        condition: Condition to check
        message: Error message
        
    Raises:
        ValueError: If condition is False
    """
    if not condition:
        raise ValueError(message)


def ensure(condition: bool, message: str = "Condition not met") -> None:
    """
    Ensure condition is true.
    
    Args:
        condition: Condition to check
        message: Error message
        
    Raises:
        AssertionError: If condition is False
    """
    if not condition:
        raise AssertionError(message)


def guard_not_none(value: Any, name: str = "value") -> Any:
    """
    Guard against None value.
    
    Args:
        value: Value to check
        name: Name for error message
        
    Returns:
        Value if not None
        
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")
    return value


def guard_not_empty(value: Any, name: str = "value") -> Any:
    """
    Guard against empty value.
    
    Args:
        value: Value to check
        name: Name for error message
        
    Returns:
        Value if not empty
        
    Raises:
        ValueError: If value is empty
    """
    from utils.check_utils import is_empty
    
    if is_empty(value):
        raise ValueError(f"{name} cannot be empty")
    return value


def guard_in_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> float:
    """
    Guard value is in range.
    
    Args:
        value: Value to check
        min_val: Minimum value
        max_val: Maximum value
        name: Name for error message
        
    Returns:
        Value if in range
        
    Raises:
        ValueError: If value not in range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return value


def guard_type(value: Any, expected_type: type, name: str = "value") -> Any:
    """
    Guard value is of expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type
        name: Name for error message
        
    Returns:
        Value if correct type
        
    Raises:
        TypeError: If value is wrong type
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type.__name__}")
    return value


def with_guards(**guards: Callable):
    """
    Decorator to add guards to function.
    
    Args:
        **guards: Guard functions keyed by parameter name
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Apply guards
            for param_name, guard_func in guards.items():
                if param_name in bound.arguments:
                    bound.arguments[param_name] = guard_func(bound.arguments[param_name])
            
            return func(*bound.args, **bound.kwargs)
        
        return wrapper
    return decorator

