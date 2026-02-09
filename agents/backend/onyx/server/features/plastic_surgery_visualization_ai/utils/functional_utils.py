"""Functional programming utilities."""

import time
from typing import Callable, Any, List, TypeVar, Optional
from functools import reduce, partial, wraps
import inspect

T = TypeVar('T')
R = TypeVar('R')


def compose(*funcs: Callable) -> Callable:
    """
    Compose multiple functions.
    
    Args:
        *funcs: Functions to compose (right to left)
        
    Returns:
        Composed function
    """
    def composed(x: Any) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(funcs), x)
    return composed


def pipe(*funcs: Callable) -> Callable:
    """
    Pipe functions (left to right).
    
    Args:
        *funcs: Functions to pipe
        
    Returns:
        Piped function
    """
    def piped(x: Any) -> Any:
        return reduce(lambda acc, f: f(acc), funcs, x)
    return piped


def curry(func: Callable, arity: Optional[int] = None) -> Callable:
    """
    Curry a function.
    
    Args:
        func: Function to curry
        arity: Function arity (auto-detected if None)
        
    Returns:
        Curried function
    """
    if arity is None:
        sig = inspect.signature(func)
        arity = len(sig.parameters)
    
    def curried(*args):
        if len(args) >= arity:
            return func(*args[:arity])
        return partial(curried, *args)
    
    return curried


def flip(func: Callable) -> Callable:
    """
    Flip function arguments.
    
    Args:
        func: Function to flip
        
    Returns:
        Function with flipped arguments
    """
    @wraps(func)
    def flipped(a: Any, b: Any, *args, **kwargs) -> Any:
        return func(b, a, *args, **kwargs)
    return flipped


def partial_apply(func: Callable, *args, **kwargs) -> Callable:
    """
    Partial application (alias for functools.partial).
    
    Args:
        func: Function to partially apply
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Partially applied function
    """
    return partial(func, *args, **kwargs)


def apply(func: Callable, args: tuple = (), kwargs: dict = None) -> Any:
    """
    Apply function with arguments.
    
    Args:
        func: Function to apply
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    if kwargs is None:
        kwargs = {}
    return func(*args, **kwargs)


def bind(func: Callable, *args, **kwargs) -> Callable:
    """
    Bind arguments to function.
    
    Args:
        func: Function to bind
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Bound function
    """
    def bound(*more_args, **more_kwargs):
        merged_kwargs = {**kwargs, **more_kwargs}
        return func(*(args + more_args), **merged_kwargs)
    return bound


def once(func: Callable) -> Callable:
    """
    Make function execute only once.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    called = False
    result = None
    
    def wrapper(*args, **kwargs):
        nonlocal called, result
        if not called:
            result = func(*args, **kwargs)
            called = True
        return result
    
    return wrapper


def debounce(func: Callable, delay: float) -> Callable:
    """
    Debounce function calls.
    
    Args:
        func: Function to debounce
        delay: Delay in seconds
        
    Returns:
        Debounced function
    """
    import asyncio
    last_call_time = None
    
    async def debounced(*args, **kwargs):
        nonlocal last_call_time
        current_time = time.time()
        
        if last_call_time is None or current_time - last_call_time >= delay:
            last_call_time = current_time
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
    
    return debounced


def memoize(func: Callable) -> Callable:
    """
    Memoize function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    def memoized(*args, **kwargs):
        key = str((args, tuple(sorted(kwargs.items()))))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    memoized.cache = cache
    memoized.clear_cache = lambda: cache.clear()
    
    return memoized



