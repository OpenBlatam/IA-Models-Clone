"""
String Utilities
================

String manipulation utilities.
"""

import re
from typing import List, Optional


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to max length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """
    Convert string to URL-friendly slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        Slug string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special chars with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    
    # Remove leading/trailing hyphens
    return text.strip('-')


def camel_to_snake(text: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        text: CamelCase string
        
    Returns:
        snake_case string
    """
    # Insert underscore before uppercase letters
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
    return text.lower()


def snake_to_camel(text: str) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        text: snake_case string
        
    Returns:
        camelCase string
    """
    components = text.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of URLs found
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text with HTML
        
    Returns:
        Text without HTML tags
    """
    return re.sub(r'<[^>]+>', '', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


def mask_sensitive(text: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive information (e.g., API keys).
    
    Args:
        text: Text to mask
        visible_chars: Number of visible characters at start
        
    Returns:
        Masked string
    """
    if len(text) <= visible_chars:
        return '*' * len(text)
    
    return text[:visible_chars] + '*' * (len(text) - visible_chars)


def pluralize(word: str, count: int) -> str:
    """
    Pluralize word based on count.
    
    Args:
        word: Word to pluralize
        count: Count
        
    Returns:
        Pluralized word
    """
    if count == 1:
        return word
    
    # Simple pluralization rules
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word.endswith('s') or word.endswith('x') or word.endswith('z'):
        return word + 'es'
    elif word.endswith('ch') or word.endswith('sh'):
        return word + 'es'
    else:
        return word + 's'

