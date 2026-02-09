"""
Comparator Utilities for Piel Mejorador AI SAM3
===============================================

Unified comparator and sorting utilities.
"""

import logging
from typing import TypeVar, Callable, Optional, List, Any
from abc import ABC, abstractmethod
from functools import cmp_to_key

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Comparator(ABC):
    """Base comparator interface."""
    
    @abstractmethod
    def compare(self, a: T, b: T) -> int:
        """
        Compare two items.
        
        Args:
            a: First item
            b: Second item
            
        Returns:
            Negative if a < b, 0 if a == b, positive if a > b
        """
        pass
    
    def __call__(self, a: T, b: T) -> int:
        """Make comparator callable."""
        return self.compare(a, b)


class FunctionComparator(Comparator):
    """Comparator using a function."""
    
    def __init__(
        self,
        compare_func: Callable[[T, T], int],
        name: Optional[str] = None
    ):
        """
        Initialize function comparator.
        
        Args:
            compare_func: Comparison function
            name: Optional comparator name
        """
        self._compare_func = compare_func
        self.name = name or compare_func.__name__
    
    def compare(self, a: T, b: T) -> int:
        """Compare items."""
        return self._compare_func(a, b)


class KeyComparator(Comparator):
    """Comparator using a key function."""
    
    def __init__(
        self,
        key_func: Callable[[T], Any],
        reverse: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize key comparator.
        
        Args:
            key_func: Key extraction function
            reverse: Whether to reverse order
            name: Optional comparator name
        """
        self._key_func = key_func
        self._reverse = reverse
        self.name = name or f"key_{key_func.__name__}"
    
    def compare(self, a: T, b: T) -> int:
        """Compare items by key."""
        key_a = self._key_func(a)
        key_b = self._key_func(b)
        
        if key_a < key_b:
            return 1 if self._reverse else -1
        elif key_a > key_b:
            return -1 if self._reverse else 1
        return 0


class ChainedComparator(Comparator):
    """Chain of comparators."""
    
    def __init__(self, *comparators: Comparator):
        """
        Initialize chained comparator.
        
        Args:
            *comparators: Comparators to chain
        """
        self._comparators = comparators
    
    def compare(self, a: T, b: T) -> int:
        """Compare using chained comparators."""
        for comparator in self._comparators:
            result = comparator.compare(a, b)
            if result != 0:
                return result
        return 0


class ComparatorUtils:
    """Unified comparator utilities."""
    
    @staticmethod
    def create_comparator(
        compare_func: Callable[[T, T], int],
        name: Optional[str] = None
    ) -> FunctionComparator:
        """
        Create comparator from function.
        
        Args:
            compare_func: Comparison function
            name: Optional comparator name
            
        Returns:
            FunctionComparator
        """
        return FunctionComparator(compare_func, name)
    
    @staticmethod
    def create_key_comparator(
        key_func: Callable[[T], Any],
        reverse: bool = False,
        name: Optional[str] = None
    ) -> KeyComparator:
        """
        Create key-based comparator.
        
        Args:
            key_func: Key extraction function
            reverse: Whether to reverse order
            name: Optional comparator name
            
        Returns:
            KeyComparator
        """
        return KeyComparator(key_func, reverse, name)
    
    @staticmethod
    def create_chained_comparator(*comparators: Comparator) -> ChainedComparator:
        """
        Create chained comparator.
        
        Args:
            *comparators: Comparators to chain
            
        Returns:
            ChainedComparator
        """
        return ChainedComparator(*comparators)
    
    @staticmethod
    def sort_items(
        items: List[T],
        comparator: Comparator,
        reverse: bool = False
    ) -> List[T]:
        """
        Sort items using comparator.
        
        Args:
            items: Items to sort
            comparator: Comparator to use
            reverse: Whether to reverse order
            
        Returns:
            Sorted items
        """
        key_func = cmp_to_key(comparator.compare)
        if reverse:
            key_func = cmp_to_key(lambda a, b: -comparator.compare(a, b))
        return sorted(items, key=key_func)
    
    @staticmethod
    def create_numeric_comparator(reverse: bool = False) -> Comparator:
        """
        Create numeric comparator.
        
        Args:
            reverse: Whether to reverse order
            
        Returns:
            Comparator
        """
        def compare(a: Any, b: Any) -> int:
            if a < b:
                return 1 if reverse else -1
            elif a > b:
                return -1 if reverse else 1
            return 0
        
        return FunctionComparator(compare, name="numeric")
    
    @staticmethod
    def create_string_comparator(reverse: bool = False, case_sensitive: bool = True) -> Comparator:
        """
        Create string comparator.
        
        Args:
            reverse: Whether to reverse order
            case_sensitive: Whether comparison is case sensitive
            
        Returns:
            Comparator
        """
        def compare(a: str, b: str) -> int:
            if not case_sensitive:
                a, b = a.lower(), b.lower()
            if a < b:
                return 1 if reverse else -1
            elif a > b:
                return -1 if reverse else 1
            return 0
        
        return FunctionComparator(compare, name="string")


# Convenience functions
def create_comparator(compare_func: Callable[[T, T], int], **kwargs) -> FunctionComparator:
    """Create comparator."""
    return ComparatorUtils.create_comparator(compare_func, **kwargs)


def create_key_comparator(key_func: Callable[[T], Any], **kwargs) -> KeyComparator:
    """Create key comparator."""
    return ComparatorUtils.create_key_comparator(key_func, **kwargs)


def sort_items(items: List[T], comparator: Comparator, **kwargs) -> List[T]:
    """Sort items."""
    return ComparatorUtils.sort_items(items, comparator, **kwargs)




