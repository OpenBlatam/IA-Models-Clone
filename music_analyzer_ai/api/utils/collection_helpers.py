"""
Collection utility helper functions.

This module provides utilities for working with collections (lists, dicts, sets)
with common patterns like empty checks, filtering, and transformations.
"""

from typing import Any, List, Dict, Optional, Callable, Set, Tuple
from .type_helpers import is_empty, is_not_empty, is_dict, is_list


def is_empty_collection(value: Any) -> bool:
    """
    Check if value is an empty collection (list, dict, set, tuple).
    
    Args:
        value: Value to check
    
    Returns:
        True if value is an empty collection
    
    Example:
        if is_empty_collection(items):
            return []
    """
    if value is None:
        return True
    
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    
    return False


def is_not_empty_collection(value: Any) -> bool:
    """
    Check if value is a non-empty collection.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a non-empty collection
    
    Example:
        if is_not_empty_collection(items):
            process(items)
    """
    return not is_empty_collection(value)


def get_length(value: Any, default: int = 0) -> int:
    """
    Get length of value, with default if not a collection.
    
    Args:
        value: Value to get length of
        default: Default length if value is not a collection
    
    Returns:
        Length of value or default
    
    Example:
        count = get_length(items, default=0)
        size = get_length(data, default=-1)
    """
    if value is None:
        return default
    
    if isinstance(value, (list, dict, set, tuple, str)):
        return len(value)
    
    return default


def first_item(items: List[Any], default: Any = None) -> Any:
    """
    Get first item from list, with default if empty.
    
    Args:
        items: List to get first item from
        default: Default value if list is empty
    
    Returns:
        First item or default
    
    Example:
        first = first_item(tracks, default=None)
        name = first_item(names, default="Unknown")
    """
    if is_not_empty_collection(items):
        return items[0]
    return default


def last_item(items: List[Any], default: Any = None) -> Any:
    """
    Get last item from list, with default if empty.
    
    Args:
        items: List to get last item from
        default: Default value if list is empty
    
    Returns:
        Last item or default
    
    Example:
        last = last_item(tracks, default=None)
    """
    if is_not_empty_collection(items):
        return items[-1]
    return default


def get_item(items: List[Any], index: int, default: Any = None) -> Any:
    """
    Get item at index, with default if index out of range.
    
    Args:
        items: List to get item from
        index: Index to get
        default: Default value if index out of range
    
    Returns:
        Item at index or default
    
    Example:
        track = get_item(tracks, 0, default=None)
        artist = get_item(artists, 1, default="Unknown")
    """
    if is_not_empty_collection(items) and 0 <= index < len(items):
        return items[index]
    return default


def filter_empty(items: List[Any]) -> List[Any]:
    """
    Filter out empty items from list.
    
    Args:
        items: List to filter
    
    Returns:
        List with empty items removed
    
    Example:
        clean = filter_empty([1, None, "", [], "text", 0])
        # Returns: [1, "text", 0]
    """
    return [item for item in items if is_not_empty(item)]


def filter_none(items: List[Any]) -> List[Any]:
    """
    Filter out None values from list.
    
    Args:
        items: List to filter
    
    Returns:
        List with None values removed
    
    Example:
        clean = filter_none([1, None, 2, None, 3])
        # Returns: [1, 2, 3]
    """
    return [item for item in items if item is not None]


def unique_items(items: List[Any], key: Optional[Callable[[Any], Any]] = None) -> List[Any]:
    """
    Get unique items from list, optionally using a key function.
    
    Args:
        items: List to get unique items from
        key: Optional function to extract comparison key
    
    Returns:
        List of unique items
    
    Example:
        unique = unique_items([1, 2, 2, 3, 3, 3])  # [1, 2, 3]
        unique_names = unique_items(tracks, key=lambda t: t.get("name"))
    """
    if not items:
        return []
    
    if key is None:
        # Simple unique
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        # Unique by key
        seen = set()
        result = []
        for item in items:
            item_key = key(item)
            if item_key not in seen:
                seen.add(item_key)
                result.append(item)
        return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    
    Example:
        chunks = chunk_list([1, 2, 3, 4, 5, 6], chunk_size=2)
        # Returns: [[1, 2], [3, 4], [5, 6]]
    """
    if chunk_size <= 0:
        return [items]
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists into a single list.
    
    Args:
        nested: List of lists
    
    Returns:
        Flattened list
    
    Example:
        flat = flatten_list([[1, 2], [3, 4], [5]])
        # Returns: [1, 2, 3, 4, 5]
    """
    result = []
    for sublist in nested:
        if is_list(sublist):
            result.extend(sublist)
        else:
            result.append(sublist)
    return result


def batch_process(
    items: List[Any],
    batch_size: int,
    processor: Callable[[List[Any]], Any]
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Function to process each batch
    
    Returns:
        List of processed results
    
    Example:
        results = batch_process(
            tracks,
            batch_size=10,
            processor=lambda batch: [process_track(t) for t in batch]
        )
    """
    chunks = chunk_list(items, batch_size)
    results = []
    
    for chunk in chunks:
        result = processor(chunk)
        if is_list(result):
            results.extend(result)
        else:
            results.append(result)
    
    return results


def dict_keys_exist(data: Dict[str, Any], *keys: str) -> bool:
    """
    Check if all specified keys exist in dictionary.
    
    Args:
        data: Dictionary to check
        *keys: Keys to check for
    
    Returns:
        True if all keys exist
    
    Example:
        if dict_keys_exist(track, "id", "name", "artists"):
            process_track(track)
    """
    if not is_dict(data):
        return False
    
    return all(key in data for key in keys)


def dict_get_many(data: Dict[str, Any], *keys: str, default: Any = None) -> Dict[str, Any]:
    """
    Get multiple values from dictionary at once.
    
    Args:
        data: Dictionary to get values from
        *keys: Keys to get
        default: Default value for missing keys
    
    Returns:
        Dictionary mapping keys to values
    
    Example:
        values = dict_get_many(track, "id", "name", "popularity", default=None)
        # Returns: {"id": "...", "name": "...", "popularity": 75}
    """
    if not is_dict(data):
        return {key: default for key in keys}
    
    return {key: data.get(key, default) for key in keys}








