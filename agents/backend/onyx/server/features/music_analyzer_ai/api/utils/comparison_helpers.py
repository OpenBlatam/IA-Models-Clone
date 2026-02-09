"""
Comparison and sorting helper functions.

This module provides utilities for comparing, sorting, and ranking
data structures with common patterns.
"""

from typing import Any, List, Dict, Callable, Optional, Tuple
from operator import itemgetter


def sort_by_key(
    items: List[Dict[str, Any]],
    key: str,
    reverse: bool = False,
    default: Any = None
) -> List[Dict[str, Any]]:
    """
    Sort a list of dictionaries by a key.
    
    Args:
        items: List of dictionaries to sort
        key: Key to sort by
        reverse: Whether to sort in reverse order
        default: Default value for missing keys
    
    Returns:
        Sorted list
    
    Example:
        sorted_tracks = sort_by_key(tracks, "popularity", reverse=True)
        sorted_by_name = sort_by_key(tracks, "name", default="")
    """
    return sorted(
        items,
        key=lambda x: x.get(key, default),
        reverse=reverse
    )


def sort_by_multiple_keys(
    items: List[Dict[str, Any]],
    keys: List[Tuple[str, bool]],
    defaults: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Sort a list of dictionaries by multiple keys.
    
    Args:
        items: List of dictionaries to sort
        keys: List of tuples (key, reverse) for sorting
        defaults: Optional dict of default values for keys
    
    Returns:
        Sorted list
    
    Example:
        sorted_tracks = sort_by_multiple_keys(
            tracks,
            [("popularity", True), ("name", False)],
            defaults={"popularity": 0, "name": ""}
        )
    """
    defaults = defaults or {}
    
    def sort_key(item):
        return tuple(
            item.get(key, defaults.get(key, None)) if not reverse else -item.get(key, defaults.get(key, 0))
            for key, reverse in keys
        )
    
    return sorted(items, key=sort_key)


def rank_by_key(
    items: List[Dict[str, Any]],
    key: str,
    reverse: bool = True
) -> List[Dict[str, Any]]:
    """
    Rank items by a key and add rank field.
    
    Args:
        items: List of dictionaries to rank
        key: Key to rank by
        reverse: Whether higher values rank better (True) or lower (False)
    
    Returns:
        List with "rank" field added
    
    Example:
        ranked = rank_by_key(tracks, "popularity", reverse=True)
        # Items now have "rank": 1, 2, 3, etc.
    """
    sorted_items = sort_by_key(items, key, reverse=reverse)
    
    for rank, item in enumerate(sorted_items, start=1):
        item["rank"] = rank
    
    return sorted_items


def compare_dicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare two dictionaries and return differences.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        keys: Optional list of keys to compare (all if None)
    
    Returns:
        Dictionary with comparison results:
        {
            "same": [list of keys with same values],
            "different": [list of keys with different values],
            "only_in_first": [list of keys only in dict1],
            "only_in_second": [list of keys only in dict2],
            "differences": {key: {"first": value1, "second": value2}}
        }
    
    Example:
        comparison = compare_dicts(track1, track2, keys=["name", "popularity"])
    """
    keys_to_compare = keys or set(dict1.keys()) | set(dict2.keys())
    
    same = []
    different = []
    only_in_first = []
    only_in_second = []
    differences = {}
    
    for key in keys_to_compare:
        val1 = dict1.get(key)
        val2 = dict2.get(key)
        
        if key not in dict1:
            only_in_second.append(key)
        elif key not in dict2:
            only_in_first.append(key)
        elif val1 == val2:
            same.append(key)
        else:
            different.append(key)
            differences[key] = {"first": val1, "second": val2}
    
    return {
        "same": same,
        "different": different,
        "only_in_first": only_in_first,
        "only_in_second": only_in_second,
        "differences": differences
    }


def find_similar_items(
    items: List[Dict[str, Any]],
    target: Dict[str, Any],
    similarity_keys: List[str],
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find items similar to target based on key values.
    
    Args:
        items: List of items to search
        target: Target item to compare against
        similarity_keys: Keys to use for similarity comparison
        threshold: Similarity threshold (0.0-1.0)
    
    Returns:
        List of similar items with similarity scores
    
    Example:
        similar = find_similar_items(
            tracks,
            target_track,
            ["genre", "energy", "tempo"],
            threshold=0.8
        )
    """
    similar = []
    
    for item in items:
        matches = 0
        total = len(similarity_keys)
        
        for key in similarity_keys:
            if key in target and key in item:
                if target[key] == item[key]:
                    matches += 1
        
        similarity = matches / total if total > 0 else 0
        
        if similarity >= threshold:
            item_copy = item.copy()
            item_copy["similarity_score"] = similarity
            similar.append(item_copy)
    
    # Sort by similarity score
    return sort_by_key(similar, "similarity_score", reverse=True)


def group_and_sort(
    items: List[Dict[str, Any]],
    group_key: str,
    sort_key: Optional[str] = None,
    reverse: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group items by a key and optionally sort within groups.
    
    Args:
        items: List of items to group
        group_key: Key to group by
        sort_key: Optional key to sort within groups
        reverse: Whether to sort in reverse
    
    Returns:
        Dictionary mapping group values to sorted lists
    
    Example:
        by_genre = group_and_sort(
            tracks,
            "genre",
            sort_key="popularity",
            reverse=True
        )
    """
    from .data_transformation_helpers import group_by
    
    grouped = group_by(items, lambda x: x.get(group_key, "unknown"))
    
    if sort_key:
        for key in grouped:
            grouped[key] = sort_by_key(grouped[key], sort_key, reverse=reverse)
    
    return grouped








