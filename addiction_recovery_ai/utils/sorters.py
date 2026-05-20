"""
Sorting utilities
Advanced sorting functions
"""

from typing import TypeVar, List, Callable, Optional

T = TypeVar('T')


def sort_by(
    items: List[T],
    key_func: Callable[[T], Any],
    reverse: bool = False
) -> List[T]:
    """
    Sort items by key function
    
    Args:
        items: List to sort
        key_func: Function to extract sort key
        reverse: Whether to reverse order
    
    Returns:
        Sorted list
    """
    return sorted(items, key=key_func, reverse=reverse)


def sort_by_multiple(
    items: List[T],
    *key_funcs: Callable[[T], Any]
) -> List[T]:
    """
    Sort items by multiple keys
    
    Args:
        items: List to sort
        *key_funcs: Key functions in priority order
    
    Returns:
        Sorted list
    """
    def sort_key(item: T) -> tuple:
        return tuple(key_func(item) for key_func in key_funcs)
    
    return sorted(items, key=sort_key)


def stable_sort(
    items: List[T],
    key_func: Optional[Callable[[T], Any]] = None
) -> List[T]:
    """
    Stable sort (preserves order of equal elements)
    
    Args:
        items: List to sort
        key_func: Optional key function
    
    Returns:
        Sorted list
    """
    if key_func:
        return sorted(items, key=key_func)
    return sorted(items)


def partial_sort(
    items: List[T],
    k: int,
    key_func: Optional[Callable[[T], Any]] = None
) -> List[T]:
    """
    Partial sort (first k elements)
    
    Args:
        items: List to sort
        k: Number of elements to sort
        key_func: Optional key function
    
    Returns:
        Partially sorted list
    """
    if key_func:
        sorted_items = sorted(items, key=key_func)
    else:
        sorted_items = sorted(items)
    
    return sorted_items[:k]

