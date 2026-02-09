"""Comparison utilities."""

from typing import Any, Callable, Optional


def compare_values(a: Any, b: Any) -> int:
    """
    Compare two values.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def is_equal(a: Any, b: Any, tolerance: float = 0.0) -> bool:
    """
    Check if two values are equal (with optional tolerance for floats).
    
    Args:
        a: First value
        b: Second value
        tolerance: Tolerance for float comparison
        
    Returns:
        True if values are equal
    """
    if tolerance > 0 and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    return a == b


def is_greater_than(a: Any, b: Any) -> bool:
    """
    Check if a is greater than b.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        True if a > b
    """
    return a > b


def is_less_than(a: Any, b: Any) -> bool:
    """
    Check if a is less than b.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        True if a < b
    """
    return a < b


def is_greater_or_equal(a: Any, b: Any) -> bool:
    """
    Check if a is greater than or equal to b.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        True if a >= b
    """
    return a >= b


def is_less_or_equal(a: Any, b: Any) -> bool:
    """
    Check if a is less than or equal to b.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        True if a <= b
    """
    return a <= b


def min_value(*values: Any) -> Any:
    """
    Get minimum value.
    
    Args:
        *values: Values to compare
        
    Returns:
        Minimum value
    """
    return min(values)


def max_value(*values: Any) -> Any:
    """
    Get maximum value.
    
    Args:
        *values: Values to compare
        
    Returns:
        Maximum value
    """
    return max(values)


def compare_lists(list1: list, list2: list) -> dict:
    """
    Compare two lists.
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        Dictionary with comparison results
    """
    set1 = set(list1)
    set2 = set(list2)
    
    return {
        "equal": list1 == list2,
        "same_length": len(list1) == len(list2),
        "only_in_first": list(set1 - set2),
        "only_in_second": list(set2 - set1),
        "common": list(set1 & set2),
    }


def compare_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Compare two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Dictionary with comparison results
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    common_keys = keys1 & keys2
    different_values = {
        key: {"dict1": dict1[key], "dict2": dict2[key]}
        for key in common_keys
        if dict1[key] != dict2[key]
    }
    
    return {
        "equal": dict1 == dict2,
        "only_in_first": list(keys1 - keys2),
        "only_in_second": list(keys2 - keys1),
        "common_keys": list(common_keys),
        "different_values": different_values,
    }

