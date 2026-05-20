"""
Comparator utilities
Comparison functions for sorting and filtering
"""

from typing import Callable, TypeVar, Any

T = TypeVar('T')


def compare_by(key_func: Callable[[T], Any], reverse: bool = False) -> Callable[[T, T], int]:
    """
    Create comparator function by key
    
    Args:
        key_func: Function to extract comparison key
        reverse: Whether to reverse order
    
    Returns:
        Comparator function
    """
    def comparator(a: T, b: T) -> int:
        key_a = key_func(a)
        key_b = key_func(b)
        
        if key_a < key_b:
            return 1 if reverse else -1
        if key_a > key_b:
            return -1 if reverse else 1
        return 0
    
    return comparator


def compare_multiple(*comparators: Callable[[T, T], int]) -> Callable[[T, T], int]:
    """
    Combine multiple comparators
    
    Args:
        *comparators: Comparators to combine
    
    Returns:
        Combined comparator
    """
    def combined(a: T, b: T) -> int:
        for comparator in comparators:
            result = comparator(a, b)
            if result != 0:
                return result
        return 0
    
    return combined


def natural_order() -> Callable[[Any, Any], int]:
    """Natural order comparator"""
    def comparator(a: Any, b: Any) -> int:
        if a < b:
            return -1
        if a > b:
            return 1
        return 0
    
    return comparator


def reverse_order() -> Callable[[Any, Any], int]:
    """Reverse order comparator"""
    def comparator(a: Any, b: Any) -> int:
        if a < b:
            return 1
        if a > b:
            return -1
        return 0
    
    return comparator

