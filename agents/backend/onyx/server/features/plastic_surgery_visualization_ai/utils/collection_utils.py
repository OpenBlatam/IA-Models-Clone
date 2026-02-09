"""Collection utilities."""

from typing import List, Dict, Any, Callable, Optional, Set, TypeVar
from collections import defaultdict, Counter
from itertools import groupby

T = TypeVar('T')


def group_by_key(items: List[Dict], key: str) -> Dict[Any, List[Dict]]:
    """
    Group list of dictionaries by key.
    
    Args:
        items: List of dictionaries
        key: Key to group by
        
    Returns:
        Dictionary grouped by key
    """
    grouped = defaultdict(list)
    for item in items:
        grouped[item.get(key)].append(item)
    return dict(grouped)


def sort_by_key(items: List[Dict], key: str, reverse: bool = False) -> List[Dict]:
    """
    Sort list of dictionaries by key.
    
    Args:
        items: List of dictionaries
        key: Key to sort by
        reverse: Reverse order
        
    Returns:
        Sorted list
    """
    return sorted(items, key=lambda x: x.get(key, ''), reverse=reverse)


def count_items(items: List[Any]) -> Dict[Any, int]:
    """
    Count occurrences of items.
    
    Args:
        items: List of items
        
    Returns:
        Dictionary with counts
    """
    return dict(Counter(items))


def remove_none(items: List[Any]) -> List[Any]:
    """
    Remove None values from list.
    
    Args:
        items: List of items
        
    Returns:
        List without None values
    """
    return [item for item in items if item is not None]


def remove_empty(items: List[Any]) -> List[Any]:
    """
    Remove empty values from list.
    
    Args:
        items: List of items
        
    Returns:
        List without empty values
    """
    return [item for item in items if item]


def zip_dicts(*dicts: Dict) -> List[Dict]:
    """
    Zip multiple dictionaries into list of dictionaries.
    
    Args:
        *dicts: Dictionaries to zip
        
    Returns:
        List of dictionaries with combined keys
    """
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    return [{k: d.get(k) for d in dicts} for k in all_keys]


def invert_dict(d: Dict) -> Dict:
    """
    Invert dictionary (swap keys and values).
    
    Args:
        d: Dictionary to invert
        
    Returns:
        Inverted dictionary
    """
    return {v: k for k, v in d.items()}


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries (later ones override earlier).
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_dict_keys(d: Dict, *keys: str) -> Dict:
    """
    Get subset of dictionary with specified keys.
    
    Args:
        d: Dictionary
        *keys: Keys to extract
        
    Returns:
        Dictionary with only specified keys
    """
    return {k: d[k] for k in keys if k in d}


def set_dict_defaults(d: Dict, defaults: Dict) -> Dict:
    """
    Set default values for dictionary keys.
    
    Args:
        d: Dictionary
        defaults: Default values
        
    Returns:
        Dictionary with defaults applied
    """
    result = defaults.copy()
    result.update(d)
    return result


# List utilities from list_utils.py
def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(lst: List[List[T]]) -> List[T]:
    """Flatten list of lists."""
    return [item for sublist in lst for item in sublist]


def unique_list(lst: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """Get unique items from list."""
    if key:
        seen = set()
        result = []
        for item in lst:
            item_key = key(item)
            if item_key not in seen:
                seen.add(item_key)
                result.append(item)
        return result
    return list(dict.fromkeys(lst))


def group_by(lst: List[T], key: Callable[[T], Any]) -> dict:
    """Group list items by key function."""
    result = defaultdict(list)
    for item in lst:
        result[key(item)].append(item)
    return dict(result)


def partition_list(lst: List[T], predicate: Callable[[T], bool]) -> tuple[List[T], List[T]]:
    """Partition list into two lists based on predicate."""
    true_list = []
    false_list = []
    for item in lst:
        if predicate(item):
            true_list.append(item)
        else:
            false_list.append(item)
    return true_list, false_list


def find_first(lst: List[T], predicate: Callable[[T], bool], default: Optional[T] = None) -> Optional[T]:
    """Find first item matching predicate."""
    for item in lst:
        if predicate(item):
            return item
    return default


def find_all(lst: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """Find all items matching predicate."""
    return [item for item in lst if predicate(item)]


def remove_duplicates(lst: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """Remove duplicates from list (alias for unique_list)."""
    return unique_list(lst, key)


def rotate_list(lst: List[T], n: int) -> List[T]:
    """Rotate list by n positions."""
    if not lst:
        return lst
    n = n % len(lst)
    return lst[-n:] + lst[:-n]

