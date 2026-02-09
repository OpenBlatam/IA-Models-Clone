"""Merging utilities."""

from typing import List, Dict, Any, Callable


def merge_lists(*lists: List) -> List:
    """
    Merge multiple lists.
    
    Args:
        *lists: Lists to merge
        
    Returns:
        Merged list
    """
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def merge_lists_unique(*lists: List) -> List:
    """
    Merge multiple lists, removing duplicates.
    
    Args:
        *lists: Lists to merge
        
    Returns:
        Merged list without duplicates
    """
    result = []
    seen = set()
    
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    
    return result


def merge_dicts_deep(*dicts: Dict) -> Dict:
    """
    Deep merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge (later override earlier)
        
    Returns:
        Deeply merged dictionary
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts_deep(result[key], value)
            else:
                result[key] = value
    
    return result


def merge_with_strategy(
    *dicts: Dict,
    strategy: Callable = lambda old, new: new
) -> Dict:
    """
    Merge dictionaries with custom strategy.
    
    Args:
        *dicts: Dictionaries to merge
        strategy: Function to resolve conflicts (old, new) -> value
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] = strategy(result[key], value)
            else:
                result[key] = value
    
    return result


def merge_sets(*sets: set) -> set:
    """
    Merge multiple sets.
    
    Args:
        *sets: Sets to merge
        
    Returns:
        Merged set
    """
    result = set()
    for s in sets:
        result.update(s)
    return result


def merge_tuples(*tuples: tuple) -> tuple:
    """
    Merge multiple tuples.
    
    Args:
        *tuples: Tuples to merge
        
    Returns:
        Merged tuple
    """
    result = []
    for t in tuples:
        result.extend(t)
    return tuple(result)

