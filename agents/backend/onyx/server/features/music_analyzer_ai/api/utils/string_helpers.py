"""
String manipulation helper functions.

This module provides utilities for common string operations like
trimming, case conversion, normalization, and formatting.
"""

from typing import Optional, Callable
import re


def normalize_string(value: Optional[str], default: str = "") -> str:
    """
    Normalize a string by trimming and handling None.
    
    Args:
        value: String to normalize (may be None)
        default: Default value if None or empty after trim
    
    Returns:
        Normalized string
    
    Example:
        name = normalize_string(request.name, default="Unknown")
        query = normalize_string(query_param, default="")
    """
    if value is None:
        return default
    
    trimmed = value.strip()
    return trimmed if trimmed else default


def to_snake_case(value: str) -> str:
    """
    Convert string to snake_case.
    
    Args:
        value: String to convert
    
    Returns:
        Snake case string
    
    Example:
        snake = to_snake_case("Track Name")  # "track_name"
        snake = to_snake_case("TrackName")  # "track_name"
    """
    # Insert underscore before uppercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', value)
    # Insert underscore before uppercase letters after lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower().replace(' ', '_').replace('-', '_')


def to_camel_case(value: str) -> str:
    """
    Convert string to camelCase.
    
    Args:
        value: String to convert
    
    Returns:
        Camel case string
    
    Example:
        camel = to_camel_case("track_name")  # "trackName"
        camel = to_camel_case("Track Name")  # "trackName"
    """
    components = re.split(r'[-_\s]+', value)
    return components[0].lower() + ''.join(word.capitalize() for word in components[1:])


def to_pascal_case(value: str) -> str:
    """
    Convert string to PascalCase.
    
    Args:
        value: String to convert
    
    Returns:
        Pascal case string
    
    Example:
        pascal = to_pascal_case("track_name")  # "TrackName"
        pascal = to_pascal_case("track name")  # "TrackName"
    """
    components = re.split(r'[-_\s]+', value)
    return ''.join(word.capitalize() for word in components)


def truncate_string(value: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to max length with suffix.
    
    Args:
        value: String to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    
    Example:
        short = truncate_string("Very long text", max_length=10)  # "Very lo..."
    """
    if len(value) <= max_length:
        return value
    
    return value[:max_length - len(suffix)] + suffix


def sanitize_string(value: str, allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize string by removing or replacing unwanted characters.
    
    Args:
        value: String to sanitize
        allowed_chars: Optional regex pattern of allowed characters
    
    Returns:
        Sanitized string
    
    Example:
        clean = sanitize_string("user@name!", allowed_chars=r"[a-zA-Z0-9_]")
        # "username"
    """
    if allowed_chars:
        return re.sub(f'[^{allowed_chars}]', '', value)
    
    # Default: remove control characters and normalize whitespace
    value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip()


def extract_words(value: str) -> list[str]:
    """
    Extract words from a string.
    
    Args:
        value: String to extract words from
    
    Returns:
        List of words
    
    Example:
        words = extract_words("Track Name Here")  # ["Track", "Name", "Here"]
        words = extract_words("track-name_here")  # ["track", "name", "here"]
    """
    # Split on whitespace, hyphens, underscores
    words = re.split(r'[\s\-_]+', value)
    return [word.strip() for word in words if word.strip()]


def join_words(words: list[str], separator: str = " ", capitalize: bool = False) -> str:
    """
    Join words into a string.
    
    Args:
        words: List of words to join
        separator: Separator between words
        capitalize: Whether to capitalize each word
    
    Returns:
        Joined string
    
    Example:
        text = join_words(["track", "name"], separator=" ", capitalize=True)
        # "Track Name"
    """
    if capitalize:
        words = [word.capitalize() for word in words]
    return separator.join(words)


def slugify(value: str, separator: str = "-") -> str:
    """
    Convert string to URL-friendly slug.
    
    Args:
        value: String to slugify
        separator: Separator character
    
    Returns:
        Slug string
    
    Example:
        slug = slugify("Track Name Here")  # "track-name-here"
        slug = slugify("Track Name!", separator="_")  # "track_name"
    """
    # Convert to lowercase
    value = value.lower()
    
    # Replace spaces and special chars with separator
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', separator, value)
    
    # Remove leading/trailing separators
    return value.strip(separator)


def format_query_string(value: str) -> str:
    """
    Format string for search query (normalize and prepare).
    
    Args:
        value: Query string to format
    
    Returns:
        Formatted query string
    
    Example:
        query = format_query_string("  Track Name  ")  # "Track Name"
        query = format_query_string("track-name")  # "track name"
    """
    # Normalize
    value = normalize_string(value, default="")
    
    # Replace hyphens/underscores with spaces
    value = re.sub(r'[-_]+', ' ', value)
    
    # Normalize whitespace
    value = re.sub(r'\s+', ' ', value)
    
    return value.strip()








