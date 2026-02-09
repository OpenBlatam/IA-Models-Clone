"""
Aggregator Utilities for Piel Mejorador AI SAM3
==============================================

Unified aggregator and accumulator pattern utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Aggregator(ABC):
    """Base aggregator interface."""
    
    @abstractmethod
    def aggregate(self, items: List[T]) -> R:
        """Aggregate items."""
        pass


class FunctionAggregator(Aggregator):
    """Aggregator using a function."""
    
    def __init__(
        self,
        aggregate_func: Callable[[List[T]], R],
        name: Optional[str] = None
    ):
        """
        Initialize function aggregator.
        
        Args:
            aggregate_func: Aggregation function
            name: Optional aggregator name
        """
        self._aggregate_func = aggregate_func
        self.name = name or aggregate_func.__name__
    
    def aggregate(self, items: List[T]) -> R:
        """Aggregate items."""
        return self._aggregate_func(items)


class Accumulator:
    """Accumulator for incremental aggregation."""
    
    def __init__(
        self,
        initial_value: Any = 0,
        aggregator: Optional[Callable[[Any, Any], Any]] = None
    ):
        """
        Initialize accumulator.
        
        Args:
            initial_value: Initial value
            aggregator: Aggregation function (default: addition)
        """
        self._value = initial_value
        self._aggregator = aggregator or (lambda a, b: a + b)
        self._count = 0
    
    def add(self, value: Any):
        """
        Add value to accumulator.
        
        Args:
            value: Value to add
        """
        self._value = self._aggregator(self._value, value)
        self._count += 1
    
    @property
    def value(self) -> Any:
        """Get current value."""
        return self._value
    
    @property
    def count(self) -> int:
        """Get count of added values."""
        return self._count
    
    def reset(self, initial_value: Any = 0):
        """
        Reset accumulator.
        
        Args:
            initial_value: New initial value
        """
        self._value = initial_value
        self._count = 0


class GroupingAggregator:
    """Aggregator that groups items by key."""
    
    def __init__(
        self,
        key_func: Callable[[T], Any],
        aggregator: Optional[Callable[[List[T]], Any]] = None
    ):
        """
        Initialize grouping aggregator.
        
        Args:
            key_func: Function to extract key
            aggregator: Aggregation function for each group
        """
        self._key_func = key_func
        self._aggregator = aggregator or (lambda items: items)
    
    def aggregate(self, items: List[T]) -> Dict[Any, Any]:
        """
        Aggregate items by grouping.
        
        Args:
            items: Items to aggregate
            
        Returns:
            Dictionary of groups
        """
        groups = defaultdict(list)
        for item in items:
            key = self._key_func(item)
            groups[key].append(item)
        
        return {
            key: self._aggregator(group_items)
            for key, group_items in groups.items()
        }


class AggregatorUtils:
    """Unified aggregator utilities."""
    
    @staticmethod
    def create_aggregator(
        aggregate_func: Callable[[List[T]], R],
        name: Optional[str] = None
    ) -> FunctionAggregator:
        """
        Create aggregator from function.
        
        Args:
            aggregate_func: Aggregation function
            name: Optional aggregator name
            
        Returns:
            FunctionAggregator
        """
        return FunctionAggregator(aggregate_func, name)
    
    @staticmethod
    def create_accumulator(
        initial_value: Any = 0,
        aggregator: Optional[Callable[[Any, Any], Any]] = None
    ) -> Accumulator:
        """
        Create accumulator.
        
        Args:
            initial_value: Initial value
            aggregator: Aggregation function
            
        Returns:
            Accumulator
        """
        return Accumulator(initial_value, aggregator)
    
    @staticmethod
    def create_grouping_aggregator(
        key_func: Callable[[T], Any],
        aggregator: Optional[Callable[[List[T]], Any]] = None
    ) -> GroupingAggregator:
        """
        Create grouping aggregator.
        
        Args:
            key_func: Key extraction function
            aggregator: Aggregation function for groups
            
        Returns:
            GroupingAggregator
        """
        return GroupingAggregator(key_func, aggregator)
    
    @staticmethod
    def sum_aggregator() -> Aggregator:
        """Create sum aggregator."""
        return FunctionAggregator(sum, name="sum")
    
    @staticmethod
    def avg_aggregator() -> Aggregator:
        """Create average aggregator."""
        def avg(items: List[float]) -> float:
            return sum(items) / len(items) if items else 0.0
        return FunctionAggregator(avg, name="avg")
    
    @staticmethod
    def min_aggregator() -> Aggregator:
        """Create min aggregator."""
        return FunctionAggregator(min, name="min")
    
    @staticmethod
    def max_aggregator() -> Aggregator:
        """Create max aggregator."""
        return FunctionAggregator(max, name="max")
    
    @staticmethod
    def count_aggregator() -> Aggregator:
        """Create count aggregator."""
        return FunctionAggregator(len, name="count")


# Convenience functions
def create_aggregator(aggregate_func: Callable[[List[T]], R], **kwargs) -> FunctionAggregator:
    """Create aggregator."""
    return AggregatorUtils.create_aggregator(aggregate_func, **kwargs)


def create_accumulator(initial_value: Any = 0, **kwargs) -> Accumulator:
    """Create accumulator."""
    return AggregatorUtils.create_accumulator(initial_value, **kwargs)




