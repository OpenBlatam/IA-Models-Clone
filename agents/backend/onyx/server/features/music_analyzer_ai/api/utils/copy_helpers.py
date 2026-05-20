"""
Copy and cloning helper functions.

This module provides utilities for safely copying and cloning
data structures with various strategies.
"""

from typing import Any, Dict, List, Optional
import copy


def safe_copy(value: Any, deep: bool = False) -> Any:
    """
    Safely copy a value (dict, list, or other).
    
    Args:
        value: Value to copy
        deep: Whether to perform deep copy
    
    Returns:
        Copied value
    
    Example:
        copied = safe_copy(data_dict)
        deep_copied = safe_copy(nested_data, deep=True)
    """
    if value is None:
        return None
    
    if deep:
        return copy.deepcopy(value)
    else:
        return copy.copy(value)


def copy_dict(data: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
    """
    Copy a dictionary.
    
    Args:
        data: Dictionary to copy
        deep: Whether to perform deep copy
    
    Returns:
        Copied dictionary
    
    Example:
        copied = copy_dict(original_dict)
        deep_copied = copy_dict(nested_dict, deep=True)
    """
    if deep:
        return copy.deepcopy(data)
    else:
        return data.copy()


def copy_list(items: List[Any], deep: bool = False) -> List[Any]:
    """
    Copy a list.
    
    Args:
        items: List to copy
        deep: Whether to perform deep copy
    
    Returns:
        Copied list
    
    Example:
        copied = copy_list(original_list)
        deep_copied = copy_list(nested_list, deep=True)
    """
    if deep:
        return copy.deepcopy(items)
    else:
        return items.copy()


def merge_dicts(*dicts: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones overriding earlier ones.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        deep: Whether to deep copy values
    
    Returns:
        Merged dictionary
    
    Example:
        merged = merge_dicts(defaults, user_config, overrides)
    """
    result = {}
    
    for d in dicts:
        if d:
            if deep:
                result.update(copy.deepcopy(d))
            else:
                result.update(d)
    
    return result


def clone_with_updates(
    original: Dict[str, Any],
    updates: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Clone a dictionary and apply updates.
    
    Args:
        original: Original dictionary
        updates: Updates to apply
        deep: Whether to deep copy
    
    Returns:
        Cloned dictionary with updates applied
    
    Example:
        updated = clone_with_updates(
            original_track,
            {"popularity": 100, "name": "New Name"}
        )
    """
    cloned = safe_copy(original, deep=deep)
    cloned.update(updates)
    return cloned








