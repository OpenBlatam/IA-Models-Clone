"""Sorting utilities."""

from typing import List, Callable, Optional, Any


def sort_by_key(items: List[dict], key: str, reverse: bool = False) -> List[dict]:
    """
    Sort list of dictionaries by key.
    
    Args:
        items: List of dictionaries
        key: Key to sort by
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    return sorted(items, key=lambda x: x.get(key), reverse=reverse)


def sort_by_keys(items: List[dict], keys: List[str], reverse: bool = False) -> List[dict]:
    """
    Sort list of dictionaries by multiple keys.
    
    Args:
        items: List of dictionaries
        keys: Keys to sort by (priority order)
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    return sorted(items, key=lambda x: tuple(x.get(k) for k in keys), reverse=reverse)


def sort_by_function(items: List, key_func: Callable, reverse: bool = False) -> List:
    """
    Sort list by function.
    
    Args:
        items: List to sort
        key_func: Function to extract sort key
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    return sorted(items, key=key_func, reverse=reverse)


def sort_natural(items: List[str], reverse: bool = False) -> List[str]:
    """
    Natural sort (handles numbers in strings).
    
    Args:
        items: List of strings
        reverse: Reverse order
        
    Returns:
        Naturally sorted list
    """
    import re
    
    def natural_key(text):
        return [
            int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', text)
        ]
    
    return sorted(items, key=natural_key, reverse=reverse)


def sort_stable(items: List, key: Optional[Callable] = None, reverse: bool = False) -> List:
    """
    Stable sort (preserves order of equal elements).
    
    Args:
        items: List to sort
        key: Optional key function
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    if key:
        return sorted(items, key=key, reverse=reverse)
    else:
        return sorted(items, reverse=reverse)


def sort_multiple(items: List, *sort_keys: Callable, reverse: bool = False) -> List:
    """
    Sort by multiple criteria.
    
    Args:
        items: List to sort
        *sort_keys: Key functions (priority order)
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    def multi_key(item):
        return tuple(key(item) for key in sort_keys)
    
    return sorted(items, key=multi_key, reverse=reverse)


def is_sorted(items: List, reverse: bool = False) -> bool:
    """
    Check if list is sorted.
    
    Args:
        items: List to check
        reverse: Check for reverse order
        
    Returns:
        True if sorted
    """
    if reverse:
        return all(items[i] >= items[i+1] for i in range(len(items)-1))
    else:
        return all(items[i] <= items[i+1] for i in range(len(items)-1))

