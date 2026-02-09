"""
Function composition utilities
Helpers for composing pure functions
"""

from typing import Callable, TypeVar, Any

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions into a single function
    
    Args:
        *functions: Functions to compose (right to left)
    
    Returns:
        Composed function
    
    Example:
        f = compose(func3, func2, func1)
        result = f(x)  # equivalent to func3(func2(func1(x)))
    """
    if not functions:
        raise ValueError("At least one function is required")
    
    def composed(value: Any) -> Any:
        result = value
        for func in reversed(functions):
            result = func(result)
        return result
    
    return composed


def pipe(value: Any, *functions: Callable) -> Any:
    """
    Pipe value through multiple functions (left to right)
    
    Args:
        value: Initial value
        *functions: Functions to apply
    
    Returns:
        Final result
    
    Example:
        result = pipe(x, func1, func2, func3)
        # equivalent to func3(func2(func1(x)))
    """
    result = value
    for func in functions:
        result = func(result)
    return result


def curry(func: Callable, arity: int = None) -> Callable:
    """
    Curry a function (partial application)
    
    Args:
        func: Function to curry
        arity: Number of arguments (auto-detect if None)
    
    Returns:
        Curried function
    """
    if arity is None:
        arity = func.__code__.co_argcount
    
    def curried(*args):
        if len(args) >= arity:
            return func(*args[:arity])
        
        def partial(*more_args):
            return curried(*(args + more_args))
        
        return partial
    
    return curried


def partial(func: Callable, *args, **kwargs) -> Callable:
    """
    Partial application of function
    
    Args:
        func: Function to partially apply
        *args: Positional arguments to fix
        **kwargs: Keyword arguments to fix
    
    Returns:
        Partially applied function
    """
    def partial_func(*more_args, **more_kwargs):
        return func(*(args + more_args), **{**kwargs, **more_kwargs})
    
    return partial_func

