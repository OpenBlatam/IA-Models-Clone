"""
String manipulation utilities
"""

from typing import Optional


def truncate_string(value: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to max length
    
    Args:
        value: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if not value or len(value) <= max_length:
        return value
    
    return value[:max_length - len(suffix)] + suffix


def normalize_string(value: str) -> str:
    """
    Normalize string (lowercase, strip)
    
    Args:
        value: String to normalize
    
    Returns:
        Normalized string
    """
    if not value:
        return ""
    
    return value.strip().lower()


def capitalize_words(value: str) -> str:
    """
    Capitalize first letter of each word
    
    Args:
        value: String to capitalize
    
    Returns:
        Capitalized string
    """
    if not value:
        return ""
    
    return " ".join(word.capitalize() for word in value.split())


def remove_special_chars(value: str, keep: Optional[str] = None) -> str:
    """
    Remove special characters from string
    
    Args:
        value: String to clean
        keep: Optional string of characters to keep
    
    Returns:
        Cleaned string
    """
    if not value:
        return ""
    
    if keep:
        return "".join(c for c in value if c.isalnum() or c in keep)
    
    return "".join(c for c in value if c.isalnum() or c.isspace())


def extract_words(value: str) -> list[str]:
    """
    Extract words from string
    
    Args:
        value: String to extract words from
    
    Returns:
        List of words
    """
    if not value:
        return []
    
    return [word.strip() for word in value.split() if word.strip()]

