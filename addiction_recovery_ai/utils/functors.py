"""
Functor utilities
Functor implementations for functional programming
"""

from typing import TypeVar, Callable, List, Any
from abc import ABC, abstractmethod

T = TypeVar('T')
U = TypeVar('U')


class Functor(ABC):
    """Base functor class"""
    
    @abstractmethod
    def map(self, func: Callable[[T], U]) -> 'Functor':
        """Map function over functor"""
        pass


class ListFunctor(Functor):
    """List as functor"""
    
    def __init__(self, values: List[T]):
        self.values = values
    
    def map(self, func: Callable[[T], U]) -> 'ListFunctor':
        """Map function over list"""
        return ListFunctor([func(item) for item in self.values])
    
    def filter(self, predicate: Callable[[T], bool]) -> 'ListFunctor':
        """Filter list by predicate"""
        return ListFunctor([item for item in self.values if predicate(item)])
    
    def flat_map(self, func: Callable[[T], List[U]]) -> 'ListFunctor':
        """FlatMap (bind) for list"""
        result = []
        for item in self.values:
            result.extend(func(item))
        return ListFunctor(result)
    
    def unwrap(self) -> List[T]:
        """Unwrap to list"""
        return self.values


class DictFunctor(Functor):
    """Dictionary as functor"""
    
    def __init__(self, value: dict[str, T]):
        self.value = value
    
    def map(self, func: Callable[[T], U]) -> 'DictFunctor':
        """Map function over dictionary values"""
        return DictFunctor({k: func(v) for k, v in self.value.items()})
    
    def map_keys(self, func: Callable[[str], str]) -> 'DictFunctor':
        """Map function over dictionary keys"""
        return DictFunctor({func(k): v for k, v in self.value.items()})
    
    def filter(self, predicate: Callable[[T], bool]) -> 'DictFunctor':
        """Filter dictionary by predicate on values"""
        return DictFunctor({k: v for k, v in self.value.items() if predicate(v)})
    
    def unwrap(self) -> dict[str, T]:
        """Unwrap to dictionary"""
        return self.value


def list_functor(values: List[T]) -> ListFunctor:
    """Create list functor"""
    return ListFunctor(values)


def dict_functor(value: dict[str, T]) -> DictFunctor:
    """Create dictionary functor"""
    return DictFunctor(value)

