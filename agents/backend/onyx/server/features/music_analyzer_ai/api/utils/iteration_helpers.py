"""
Iteration helper functions.

This module provides utilities for common iteration patterns
like batch processing, indexed iteration, and zipping.
"""

from typing import Any, List, Tuple, Callable, Optional, Iterator
from itertools import zip_longest


def enumerate_with_default(
    items: List[Any],
    start: int = 0,
    default: Any = None
) -> Iterator[Tuple[int, Any]]:
    """
    Enumerate items with default value for empty list.
    
    Args:
        items: List to enumerate
        start: Starting index
        default: Default value if list is empty
    
    Returns:
        Iterator of (index, item) tuples
    
    Example:
        for i, item in enumerate_with_default(items, start=1):
            process(i, item)
    """
    if not items:
        yield (start, default)
    else:
        for index, item in enumerate(items, start=start):
            yield (index, item)


def zip_safe(*iterables: List[Any], default: Any = None) -> Iterator[Tuple[Any, ...]]:
    """
    Zip multiple lists safely, using default for shorter lists.
    
    Args:
        *iterables: Variable number of lists to zip
        default: Default value for missing items
    
    Returns:
        Iterator of tuples
    
    Example:
        for name, age, city in zip_safe(names, ages, cities, default=""):
            process(name, age, city)
    """
    return zip_longest(*iterables, fillvalue=default)


def batch_iterate(
    items: List[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    """
    Iterate over items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
    
    Returns:
        Iterator of batches
    
    Example:
        for batch in batch_iterate(tracks, batch_size=10):
            process_batch(batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def indexed_map(
    items: List[Any],
    transform: Callable[[int, Any], Any],
    start: int = 0
) -> List[Any]:
    """
    Map items with index information.
    
    Args:
        items: List of items
        transform: Function (index, item) -> transformed_item
        start: Starting index
    
    Returns:
        List of transformed items
    
    Example:
        indexed = indexed_map(
            tracks,
            lambda i, track: {**track, "index": i}
        )
    """
    return [transform(i, item) for i, item in enumerate(items, start=start)]


def pairwise(items: List[Any]) -> Iterator[Tuple[Any, Any]]:
    """
    Iterate over pairs of consecutive items.
    
    Args:
        items: List of items
    
    Returns:
        Iterator of (item, next_item) tuples
    
    Example:
        for current, next_item in pairwise(tracks):
            compare(current, next_item)
    """
    for i in range(len(items) - 1):
        yield (items[i], items[i + 1])


def window(
    items: List[Any],
    size: int,
    step: int = 1
) -> Iterator[List[Any]]:
    """
    Create sliding window over items.
    
    Args:
        items: List of items
        size: Window size
        step: Step size between windows
    
    Returns:
        Iterator of windows
    
    Example:
        for window_items in window(tracks, size=3, step=1):
            process_window(window_items)
    """
    for i in range(0, len(items) - size + 1, step):
        yield items[i:i + size]


def partition(
    items: List[Any],
    predicate: Callable[[Any], bool]
) -> Tuple[List[Any], List[Any]]:
    """
    Partition items into two lists based on predicate.
    
    Args:
        items: List of items
        predicate: Function to test items
    
    Returns:
        Tuple of (matching_items, non_matching_items)
    
    Example:
        valid, invalid = partition(tracks, lambda t: t.get("id") is not None)
    """
    matching = []
    non_matching = []
    
    for item in items:
        if predicate(item):
            matching.append(item)
        else:
            non_matching.append(item)
    
    return (matching, non_matching)


def group_by_key(
    items: List[Any],
    key_func: Callable[[Any], Any]
) -> Dict[Any, List[Any]]:
    """
    Group items by a key function.
    
    Args:
        items: List of items
        key_func: Function to extract key from item
    
    Returns:
        Dictionary mapping keys to lists of items
    
    Example:
        by_genre = group_by_key(tracks, lambda t: t.get("genre", "unknown"))
    """
    from .data_transformation_helpers import group_by
    return group_by(items, key_func)


def flatten_iterable(nested: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested iterable.
    
    Args:
        nested: List of lists
    
    Returns:
        Flattened list
    
    Example:
        flat = flatten_iterable([[1, 2], [3, 4]])  # [1, 2, 3, 4]
    """
    from .collection_helpers import flatten_list
    return flatten_list(nested)








