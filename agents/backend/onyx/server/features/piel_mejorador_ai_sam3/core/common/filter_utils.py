"""
Filter and Predicate Utilities for Piel Mejorador AI SAM3
========================================================

Unified filter and predicate pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, List, Dict, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Predicate(ABC):
    """Base predicate interface."""
    
    @abstractmethod
    def test(self, item: T) -> bool:
        """Test if item matches predicate."""
        pass
    
    def __and__(self, other: "Predicate") -> "AndPredicate":
        """Combine with AND."""
        return AndPredicate(self, other)
    
    def __or__(self, other: "Predicate") -> "OrPredicate":
        """Combine with OR."""
        return OrPredicate(self, other)
    
    def __invert__(self) -> "NotPredicate":
        """Negate predicate."""
        return NotPredicate(self)


class FunctionPredicate(Predicate):
    """Predicate using a function."""
    
    def __init__(self, test_func: Callable[[T], bool], name: Optional[str] = None):
        """
        Initialize function predicate.
        
        Args:
            test_func: Test function
            name: Optional predicate name
        """
        self._test_func = test_func
        self.name = name or test_func.__name__
    
    def test(self, item: T) -> bool:
        """Test item."""
        return self._test_func(item)


class AndPredicate(Predicate):
    """AND combination of predicates."""
    
    def __init__(self, *predicates: Predicate):
        """
        Initialize AND predicate.
        
        Args:
            *predicates: Predicates to combine
        """
        self._predicates = predicates
    
    def test(self, item: T) -> bool:
        """Test item (all must be true)."""
        return all(p.test(item) for p in self._predicates)


class OrPredicate(Predicate):
    """OR combination of predicates."""
    
    def __init__(self, *predicates: Predicate):
        """
        Initialize OR predicate.
        
        Args:
            *predicates: Predicates to combine
        """
        self._predicates = predicates
    
    def test(self, item: T) -> bool:
        """Test item (any must be true)."""
        return any(p.test(item) for p in self._predicates)


class NotPredicate(Predicate):
    """Negation of predicate."""
    
    def __init__(self, predicate: Predicate):
        """
        Initialize NOT predicate.
        
        Args:
            predicate: Predicate to negate
        """
        self._predicate = predicate
    
    def test(self, item: T) -> bool:
        """Test item (negated)."""
        return not self._predicate.test(item)


class FilterUtils:
    """Unified filter utilities."""
    
    @staticmethod
    def create_predicate(
        test_func: Callable[[T], bool],
        name: Optional[str] = None
    ) -> FunctionPredicate:
        """
        Create predicate from function.
        
        Args:
            test_func: Test function
            name: Optional predicate name
            
        Returns:
            FunctionPredicate
        """
        return FunctionPredicate(test_func, name)
    
    @staticmethod
    def filter_items(items: List[T], predicate: Predicate) -> List[T]:
        """
        Filter items using predicate.
        
        Args:
            items: Items to filter
            predicate: Predicate to use
            
        Returns:
            Filtered items
        """
        return [item for item in items if predicate.test(item)]
    
    @staticmethod
    def filter_dict(
        data: Dict[str, Any],
        key_predicate: Optional[Callable[[str], bool]] = None,
        value_predicate: Optional[Callable[[Any], bool]] = None
    ) -> Dict[str, Any]:
        """
        Filter dictionary.
        
        Args:
            data: Dictionary to filter
            key_predicate: Optional predicate for keys
            value_predicate: Optional predicate for values
            
        Returns:
            Filtered dictionary
        """
        result = {}
        for key, value in data.items():
            if key_predicate and not key_predicate(key):
                continue
            if value_predicate and not value_predicate(value):
                continue
            result[key] = value
        return result
    
    @staticmethod
    def create_equals_predicate(value: Any) -> Predicate:
        """
        Create equals predicate.
        
        Args:
            value: Value to compare
            
        Returns:
            Predicate
        """
        return FunctionPredicate(lambda x: x == value, name=f"equals_{value}")
    
    @staticmethod
    def create_in_predicate(values: List[Any]) -> Predicate:
        """
        Create "in" predicate.
        
        Args:
            values: Values to check
            
        Returns:
            Predicate
        """
        return FunctionPredicate(lambda x: x in values, name="in")
    
    @staticmethod
    def create_range_predicate(min_val: Any, max_val: Any) -> Predicate:
        """
        Create range predicate.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Predicate
        """
        return FunctionPredicate(
            lambda x: min_val <= x <= max_val,
            name=f"range_{min_val}_{max_val}"
        )
    
    @staticmethod
    def create_not_none_predicate() -> Predicate:
        """
        Create not None predicate.
        
        Returns:
            Predicate
        """
        return FunctionPredicate(lambda x: x is not None, name="not_none")
    
    @staticmethod
    def create_not_empty_predicate() -> Predicate:
        """
        Create not empty predicate.
        
        Returns:
            Predicate
        """
        return FunctionPredicate(
            lambda x: x is not None and len(x) > 0 if hasattr(x, '__len__') else x is not None,
            name="not_empty"
        )


# Convenience functions
def create_predicate(test_func: Callable[[T], bool], **kwargs) -> FunctionPredicate:
    """Create predicate."""
    return FilterUtils.create_predicate(test_func, **kwargs)


def filter_items(items: List[T], predicate: Predicate) -> List[T]:
    """Filter items."""
    return FilterUtils.filter_items(items, predicate)




