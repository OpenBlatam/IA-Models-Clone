"""
Trampoline utilities
For safe recursion without stack overflow
"""

from typing import TypeVar, Callable, Union
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class Done:
    """Done trampoline - final result"""
    value: T


@dataclass
class More:
    """More trampoline - continue computation"""
    thunk: Callable[[], 'Trampoline']


Trampoline = Union[Done[T], More[T]]


def trampoline(func: Callable[[], Trampoline[T]]) -> T:
    """
    Execute trampolined function
    
    Args:
        func: Function that returns Trampoline
    
    Returns:
        Final result
    """
    result = func()
    
    while isinstance(result, More):
        result = result.thunk()
    
    return result.value


def make_trampoline(func: Callable) -> Callable:
    """
    Convert recursive function to trampolined version
    
    Args:
        func: Recursive function
    
    Returns:
        Trampolined function
    """
    def trampolined(*args, **kwargs):
        def thunk():
            return func(*args, **kwargs)
        
        return trampoline(thunk)
    
    return trampolined

