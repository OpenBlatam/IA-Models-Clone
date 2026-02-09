"""Search utilities."""

from typing import List, Callable, Any, Optional


def linear_search(items: List, target: Any) -> Optional[int]:
    """
    Linear search in list.
    
    Args:
        items: List to search
        target: Target value
        
    Returns:
        Index of target or None
    """
    try:
        return items.index(target)
    except ValueError:
        return None


def binary_search(items: List, target: Any) -> Optional[int]:
    """
    Binary search in sorted list.
    
    Args:
        items: Sorted list to search
        target: Target value
        
    Returns:
        Index of target or None
    """
    left, right = 0, len(items) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None


def search_by_key(items: List[dict], key: str, value: Any) -> Optional[dict]:
    """
    Search for dictionary in list by key-value.
    
    Args:
        items: List of dictionaries
        key: Key to search
        value: Value to match
        
    Returns:
        Matching dictionary or None
    """
    for item in items:
        if item.get(key) == value:
            return item
    return None


def search_all_by_key(items: List[dict], key: str, value: Any) -> List[dict]:
    """
    Search for all dictionaries matching key-value.
    
    Args:
        items: List of dictionaries
        key: Key to search
        value: Value to match
        
    Returns:
        List of matching dictionaries
    """
    return [item for item in items if item.get(key) == value]


def fuzzy_search(text: str, pattern: str, threshold: float = 0.8) -> bool:
    """
    Fuzzy string search.
    
    Args:
        text: Text to search in
        pattern: Pattern to search for
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        True if similarity >= threshold
    """
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, text.lower(), pattern.lower()).ratio()
    return similarity >= threshold


def contains_any(text: str, patterns: List[str], case_sensitive: bool = False) -> bool:
    """
    Check if text contains any of the patterns.
    
    Args:
        text: Text to search
        patterns: List of patterns
        case_sensitive: Case sensitive search
        
    Returns:
        True if any pattern found
    """
    if not case_sensitive:
        text = text.lower()
        patterns = [p.lower() for p in patterns]
    
    return any(pattern in text for pattern in patterns)


def contains_all(text: str, patterns: List[str], case_sensitive: bool = False) -> bool:
    """
    Check if text contains all patterns.
    
    Args:
        text: Text to search
        patterns: List of patterns
        case_sensitive: Case sensitive search
        
    Returns:
        True if all patterns found
    """
    if not case_sensitive:
        text = text.lower()
        patterns = [p.lower() for p in patterns]
    
    return all(pattern in text for pattern in patterns)

