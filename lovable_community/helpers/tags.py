"""
Tag-related helper functions

Functions for parsing, formatting, and extracting tags.
"""

import re
from typing import Optional, List
from .string_normalization import normalize_list_to_lower, join_strings
from .validation_common import is_empty_string, is_empty_list


def parse_tags_string(tags_str: Optional[str]) -> Optional[List[str]]:
    """
    Parses a comma-separated tags string to a list.
    
    Optimized to avoid unnecessary operations.
    
    Args:
        tags_str: String with comma-separated tags
        
    Returns:
        List of sanitized tags or None
    """
    if is_empty_string(tags_str):
        return None
    
    stripped = tags_str.strip()
    if is_empty_string(stripped):
        return None
    
    tags = stripped.split(",")
    normalized_tags = normalize_list_to_lower(tags)
    return normalized_tags if normalized_tags else None


def format_tags_list(tags: Optional[List[str]]) -> Optional[str]:
    """
    Formats a list of tags to a comma-separated string.
    
    Args:
        tags: List of tags
        
    Returns:
        String with comma-separated tags or None
    """
    if is_empty_list(tags):
        return None
    
    # Sanitize and join tags
    sanitized = normalize_list_to_lower(tags)
    if not sanitized:
        return None
    return join_strings(sanitized, ",", filter_empty=True)


def extract_tags_from_text(text: str) -> List[str]:
    """
    Extracts possible tags from text.
    
    Searches for words starting with # and extracts them as tags.
    
    Args:
        text: Text from which to extract tags
        
    Returns:
        List of extracted tags
    """
    if is_empty_string(text):
        return []
    
    # Find words starting with #
    tags = re.findall(r'#(\w+)', text.lower())
    # Remove duplicates while maintaining order
    from .common import remove_duplicates
    
    # Filter by length and remove duplicates
    filtered_tags = [tag for tag in tags if tag and len(tag) <= 50]
    unique_tags = remove_duplicates(filtered_tags)
    
    return unique_tags[:10]  # Maximum 10 tags


def normalize_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalizes a list of tags (lowercase, strip, deduplicate).
    
    Args:
        tags: List of tags
        
    Returns:
        Normalized list or None
    """
    if is_empty_list(tags):
        return None
    
    normalized = normalize_list_to_lower(tags)
    if not normalized:
        return None
    
    # Remove duplicates while maintaining order
    from .common import remove_duplicates
    
    unique_tags = remove_duplicates(normalized)
    return unique_tags if unique_tags else None



