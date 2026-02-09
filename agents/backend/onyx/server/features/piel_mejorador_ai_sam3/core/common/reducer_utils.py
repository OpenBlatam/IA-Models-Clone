"""
Reducer Utilities for Piel Mejorador AI SAM3
===========================================

Unified data reduction pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, List, Dict
from abc import ABC, abstractmethod
from functools import reduce as functools_reduce

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Reducer(ABC):
    """Base reducer interface."""
    
    @abstractmethod
    def reduce(self, items: List[T], initial: Optional[R] = None) -> R:
        """Reduce items to a single value."""
        pass


class FunctionReducer(Reducer):
    """Reducer using a function."""
    
    def __init__(
        self,
        reduce_func: Callable[[R, T], R],
        name: Optional[str] = None
    ):
        """
        Initialize function reducer.
        
        Args:
            reduce_func: Reduction function (accumulator, item) -> accumulator
            name: Optional reducer name
        """
        self._reduce_func = reduce_func
        self.name = name or reduce_func.__name__
    
    def reduce(self, items: List[T], initial: Optional[R] = None) -> R:
        """
        Reduce items.
        
        Args:
            items: Items to reduce
            initial: Optional initial value
            
        Returns:
            Reduced value
        """
        if not items:
            if initial is None:
                raise ValueError("Cannot reduce empty list without initial value")
            return initial
        
        if initial is None:
            result = items[0]
            for item in items[1:]:
                result = self._reduce_func(result, item)
        else:
            result = initial
            for item in items:
                result = self._reduce_func(result, item)
        
        return result


class SumReducer(Reducer):
    """Sum reducer for numbers."""
    
    def reduce(self, items: List[float], initial: Optional[float] = None) -> float:
        """
        Sum items.
        
        Args:
            items: Numbers to sum
            initial: Optional initial value
            
        Returns:
            Sum
        """
        if initial is None:
            return sum(items)
        return sum(items) + initial


class ProductReducer(Reducer):
    """Product reducer for numbers."""
    
    def reduce(self, items: List[float], initial: Optional[float] = None) -> float:
        """
        Multiply items.
        
        Args:
            items: Numbers to multiply
            initial: Optional initial value (defaults to 1)
            
        Returns:
            Product
        """
        if initial is None:
            initial = 1.0
        
        result = initial
        for item in items:
            result *= item
        return result


class MaxReducer(Reducer):
    """Maximum reducer."""
    
    def reduce(self, items: List[T], initial: Optional[T] = None) -> T:
        """
        Find maximum item.
        
        Args:
            items: Items to compare
            initial: Optional initial value
            
        Returns:
            Maximum item
        """
        if not items:
            if initial is None:
                raise ValueError("Cannot find max of empty list without initial value")
            return initial
        
        if initial is None:
            return max(items)
        return max([initial] + items)


class MinReducer(Reducer):
    """Minimum reducer."""
    
    def reduce(self, items: List[T], initial: Optional[T] = None) -> T:
        """
        Find minimum item.
        
        Args:
            items: Items to compare
            initial: Optional initial value
            
        Returns:
            Minimum item
        """
        if not items:
            if initial is None:
                raise ValueError("Cannot find min of empty list without initial value")
            return initial
        
        if initial is None:
            return min(items)
        return min([initial] + items)


class CountReducer(Reducer):
    """Count reducer."""
    
    def reduce(self, items: List[T], initial: Optional[int] = None) -> int:
        """
        Count items.
        
        Args:
            items: Items to count
            initial: Optional initial count
            
        Returns:
            Count
        """
        count = len(items)
        if initial is not None:
            count += initial
        return count


class ReducerUtils:
    """Unified reducer utilities."""
    
    @staticmethod
    def create_function_reducer(
        reduce_func: Callable[[R, T], R],
        name: Optional[str] = None
    ) -> FunctionReducer:
        """
        Create function reducer.
        
        Args:
            reduce_func: Reduction function
            name: Optional reducer name
            
        Returns:
            FunctionReducer
        """
        return FunctionReducer(reduce_func, name)
    
    @staticmethod
    def create_sum_reducer() -> SumReducer:
        """
        Create sum reducer.
        
        Returns:
            SumReducer
        """
        return SumReducer()
    
    @staticmethod
    def create_product_reducer() -> ProductReducer:
        """
        Create product reducer.
        
        Returns:
            ProductReducer
        """
        return ProductReducer()
    
    @staticmethod
    def create_max_reducer() -> MaxReducer:
        """
        Create max reducer.
        
        Returns:
            MaxReducer
        """
        return MaxReducer()
    
    @staticmethod
    def create_min_reducer() -> MinReducer:
        """
        Create min reducer.
        
        Returns:
            MinReducer
        """
        return MinReducer()
    
    @staticmethod
    def create_count_reducer() -> CountReducer:
        """
        Create count reducer.
        
        Returns:
            CountReducer
        """
        return CountReducer()
    
    @staticmethod
    def reduce(
        items: List[T],
        reduce_func: Callable[[R, T], R],
        initial: Optional[R] = None
    ) -> R:
        """
        Reduce items using function.
        
        Args:
            items: Items to reduce
            reduce_func: Reduction function
            initial: Optional initial value
            
        Returns:
            Reduced value
        """
        return FunctionReducer(reduce_func).reduce(items, initial)
    
    @staticmethod
    def sum_items(items: List[float], initial: Optional[float] = None) -> float:
        """
        Sum items.
        
        Args:
            items: Numbers to sum
            initial: Optional initial value
            
        Returns:
            Sum
        """
        return SumReducer().reduce(items, initial)
    
    @staticmethod
    def product_items(items: List[float], initial: Optional[float] = None) -> float:
        """
        Multiply items.
        
        Args:
            items: Numbers to multiply
            initial: Optional initial value
            
        Returns:
            Product
        """
        return ProductReducer().reduce(items, initial)
    
    @staticmethod
    def max_item(items: List[T], initial: Optional[T] = None) -> T:
        """
        Find maximum item.
        
        Args:
            items: Items to compare
            initial: Optional initial value
            
        Returns:
            Maximum item
        """
        return MaxReducer().reduce(items, initial)
    
    @staticmethod
    def min_item(items: List[T], initial: Optional[T] = None) -> T:
        """
        Find minimum item.
        
        Args:
            items: Items to compare
            initial: Optional initial value
            
        Returns:
            Minimum item
        """
        return MinReducer().reduce(items, initial)


# Convenience functions
def create_function_reducer(reduce_func: Callable[[R, T], R], **kwargs) -> FunctionReducer:
    """Create function reducer."""
    return ReducerUtils.create_function_reducer(reduce_func, **kwargs)


def reduce_items(items: List[T], reduce_func: Callable[[R, T], R], **kwargs) -> R:
    """Reduce items using function."""
    return ReducerUtils.reduce(items, reduce_func, **kwargs)




