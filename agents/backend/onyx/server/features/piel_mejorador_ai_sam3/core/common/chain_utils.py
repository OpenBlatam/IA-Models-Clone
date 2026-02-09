"""
Chain Utilities for Piel Mejorador AI SAM3
==========================================

Unified chain pattern utilities for fluent method chaining.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Chain:
    """Fluent chain for method chaining."""
    
    def __init__(self, value: T):
        """
        Initialize chain.
        
        Args:
            value: Initial value
        """
        self._value = value
    
    def map(self, func: Callable[[T], R]) -> "Chain":
        """
        Map value through function.
        
        Args:
            func: Mapping function
            
        Returns:
            Self for chaining
        """
        self._value = func(self._value)
        return self
    
    def filter(self, predicate: Callable[[T], bool]) -> "Chain":
        """
        Filter value if it's a collection.
        
        Args:
            predicate: Filter predicate
            
        Returns:
            Self for chaining
        """
        if isinstance(self._value, (list, tuple, set)):
            self._value = type(self._value)(item for item in self._value if predicate(item))
        return self
    
    def reduce(
        self,
        func: Callable[[Any, T], Any],
        initial: Optional[Any] = None
    ) -> "Chain":
        """
        Reduce value if it's a collection.
        
        Args:
            func: Reduction function
            initial: Optional initial value
            
        Returns:
            Self for chaining
        """
        if isinstance(self._value, (list, tuple, set)):
            if initial is None:
                if not self._value:
                    return self
                result = self._value[0]
                for item in self._value[1:]:
                    result = func(result, item)
            else:
                result = initial
                for item in self._value:
                    result = func(result, item)
            self._value = result
        return self
    
    def apply(self, func: Callable[[T], R]) -> "Chain":
        """
        Apply function to value.
        
        Args:
            func: Function to apply
            
        Returns:
            Self for chaining
        """
        self._value = func(self._value)
        return self
    
    def tap(self, func: Callable[[T], None]) -> "Chain":
        """
        Tap into chain without modifying value.
        
        Args:
            func: Function to call
            
        Returns:
            Self for chaining
        """
        func(self._value)
        return self
    
    def value(self) -> T:
        """
        Get current value.
        
        Returns:
            Current value
        """
        return self._value
    
    def __call__(self) -> T:
        """Get value by calling chain."""
        return self._value


class ChainUtils:
    """Unified chain utilities."""
    
    @staticmethod
    def create_chain(value: T) -> Chain:
        """
        Create chain from value.
        
        Args:
            value: Initial value
            
        Returns:
            Chain
        """
        return Chain(value)
    
    @staticmethod
    def chain(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """
        Chain multiple functions.
        
        Args:
            *funcs: Functions to chain
            
        Returns:
            Chained function
        """
        def chained(value: Any) -> Any:
            result = value
            for func in funcs:
                result = func(result)
            return result
        return chained
    
    @staticmethod
    def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """
        Compose functions (right to left).
        
        Args:
            *funcs: Functions to compose
            
        Returns:
            Composed function
        """
        def composed(value: Any) -> Any:
            result = value
            for func in reversed(funcs):
                result = func(result)
            return result
        return composed
    
    @staticmethod
    def pipe(value: T, *funcs: Callable[[Any], Any]) -> Any:
        """
        Pipe value through functions.
        
        Args:
            value: Initial value
            *funcs: Functions to apply
            
        Returns:
            Final value
        """
        result = value
        for func in funcs:
            result = func(result)
        return result


# Convenience functions
def create_chain(value: T) -> Chain:
    """Create chain."""
    return ChainUtils.create_chain(value)


def chain(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Chain functions."""
    return ChainUtils.chain(*funcs)


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions."""
    return ChainUtils.compose(*funcs)


def pipe(value: T, *funcs: Callable[[Any], Any]) -> Any:
    """Pipe value through functions."""
    return ChainUtils.pipe(value, *funcs)




