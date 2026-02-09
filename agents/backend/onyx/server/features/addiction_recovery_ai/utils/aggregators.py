"""
Aggregation utilities
Data aggregation functions
"""

from typing import TypeVar, List, Callable, Dict, Any
from collections import defaultdict

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def group_by_key(
    items: List[T],
    key_func: Callable[[T], K]
) -> Dict[K, List[T]]:
    """
    Group items by key
    
    Args:
        items: List of items
        key_func: Function to extract key
    
    Returns:
        Dictionary grouped by key
    """
    result: Dict[K, List[T]] = defaultdict(list)
    
    for item in items:
        key = key_func(item)
        result[key].append(item)
    
    return dict(result)


def aggregate(
    items: List[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], V],
    aggregator: Callable[[List[V]], Any]
) -> Dict[K, Any]:
    """
    Aggregate values by key
    
    Args:
        items: List of items
        key_func: Function to extract key
        value_func: Function to extract value
        aggregator: Function to aggregate values
    
    Returns:
        Dictionary with aggregated values
    """
    grouped = group_by_key(items, key_func)
    
    return {
        key: aggregator([value_func(item) for item in group])
        for key, group in grouped.items()
    }


def sum_by(
    items: List[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], float]
) -> Dict[K, float]:
    """
    Sum values by key
    
    Args:
        items: List of items
        key_func: Function to extract key
        value_func: Function to extract numeric value
    
    Returns:
        Dictionary with sums by key
    """
    return aggregate(items, key_func, value_func, sum)


def count_by(
    items: List[T],
    key_func: Callable[[T], K]
) -> Dict[K, int]:
    """
    Count items by key
    
    Args:
        items: List of items
        key_func: Function to extract key
    
    Returns:
        Dictionary with counts by key
    """
    result: Dict[K, int] = defaultdict(int)
    
    for item in items:
        key = key_func(item)
        result[key] += 1
    
    return dict(result)


def average_by(
    items: List[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], float]
) -> Dict[K, float]:
    """
    Average values by key
    
    Args:
        items: List of items
        key_func: Function to extract key
        value_func: Function to extract numeric value
    
    Returns:
        Dictionary with averages by key
    """
    grouped = group_by_key(items, key_func)
    
    return {
        key: sum(value_func(item) for item in group) / len(group)
        for key, group in grouped.items()
    }

