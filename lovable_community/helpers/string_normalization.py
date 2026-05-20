"""
String Normalization Helpers

Helper functions for common string normalization patterns that appear
throughout the codebase to improve consistency and reduce duplication.
"""

from typing import Optional, List
from .validation_common import is_empty_string


def normalize_to_lower(value: str) -> str:
    """
    Normalize a string by stripping whitespace and converting to lowercase.
    
    This helper encapsulates the common pattern of `.strip().lower()` that
    appears 36+ times across the codebase.
    
    Args:
        value: String to normalize
        
    Returns:
        Normalized string (stripped and lowercased)
        
    Raises:
        TypeError: If value is not a string
        
    Example:
        >>> vote_type = normalize_to_lower(vote_type)
        >>> tag = normalize_to_lower(tag)
    """
    if not isinstance(value, str):
        raise TypeError(f"value must be a string, got {type(value).__name__}")
    
    return value.strip().lower()


def normalize_to_lower_or_none(value: Optional[str]) -> Optional[str]:
    """
    Normalize a string by stripping whitespace and converting to lowercase,
    returning None if value is None or empty.
    
    Args:
        value: String to normalize (can be None)
        
    Returns:
        Normalized string (stripped and lowercased) or None if value is None or empty
        
    Example:
        >>> vote_type = normalize_to_lower_or_none(vote_type)
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        return None
    
    normalized = value.strip().lower()
    return normalized if normalized else None


def normalize_list_to_lower(values: List[str]) -> List[str]:
    """
    Normalize a list of strings by stripping whitespace and converting to lowercase.
    
    Filters out None and empty values.
    
    Args:
        values: List of strings to normalize
        
    Returns:
        List of normalized strings (non-empty only)
        
    Example:
        >>> tags = normalize_list_to_lower(tags)
    """
    if not values:
        return []
    
    normalized = []
    for value in values:
        if value and isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value:
                normalized.append(normalized_value)
    
    return normalized


def build_search_term(query: str) -> str:
    """
    Build a search term for SQL LIKE queries.
    
    This helper encapsulates the common pattern of creating search terms
    with wildcards for database queries.
    
    Args:
        query: Search query string
        
    Returns:
        Search term with wildcards (e.g., "%query%")
        
    Example:
        >>> search_term = build_search_term(query)
        >>> # Results in: "%query%"
    """
    if is_empty_string(query):
        return "%"
    
    normalized = normalize_to_lower(query)
    return f"%{normalized}%"


def join_strings(items: List[str], separator: str = ", ", filter_empty: bool = True) -> str:
    """
    Join a list of strings with a separator.
    
    This helper encapsulates the common pattern of joining strings that appears
    repeatedly across the codebase (comma-separated, space-separated, etc.).
    
    Args:
        items: List of strings to join
        separator: Separator string (default: ", ")
        filter_empty: Whether to filter out empty strings (default: True)
        
    Returns:
        Joined string
        
    Example:
        >>> tags = join_strings(["python", "ai"], ", ")
        >>> # Returns: "python, ai"
        >>> patterns = join_strings(patterns, " OR ")
    """
    if not items:
        return ""
    
    if filter_empty:
        items = [item for item in items if item]
    
    if not items:
        return ""
    
    return separator.join(items)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in a string by collapsing multiple spaces into one.
    
    This helper encapsulates the common pattern of normalizing whitespace
    that appears in search query normalization.
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized whitespace
        
    Example:
        >>> normalized = normalize_whitespace("hello    world")
        >>> # Returns: "hello world"
    """
    if is_empty_string(text):
        return ""
    
    normalized = normalize_to_lower(text)
    return " ".join(normalized.split())

