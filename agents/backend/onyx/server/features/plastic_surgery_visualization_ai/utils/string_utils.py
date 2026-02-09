"""String utilities."""

import re
from typing import Optional
from urllib.parse import urlparse, urljoin


def slugify(text: str, separator: str = "-") -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify
        separator: Separator character
        
    Returns:
        Slug string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and underscores
    text = re.sub(r'[\s_]+', separator, text)
    
    # Remove special characters
    text = re.sub(r'[^\w\-]+', '', text)
    
    # Remove multiple separators
    text = re.sub(rf'{separator}+', separator, text)
    
    # Remove leading/trailing separators
    text = text.strip(separator)
    
    return text


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def camel_to_snake(text: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        text: CamelCase text
        
    Returns:
        snake_case text
    """
    # Insert underscore before uppercase letters
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
    return text.lower()


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        text: snake_case text
        capitalize_first: Whether to capitalize first letter
        
    Returns:
        camelCase text
    """
    components = text.split('_')
    
    if capitalize_first:
        return ''.join(word.capitalize() for word in components)
    else:
        return components[0] + ''.join(word.capitalize() for word in components[1:])


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
    return text.strip()


def extract_emails(text: str) -> list[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of email addresses
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of URLs
    """
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)


def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalize URL (make absolute if base_url provided).
    
    Args:
        url: URL to normalize
        base_url: Base URL for relative URLs
        
    Returns:
        Normalized URL
    """
    if base_url and not url.startswith(('http://', 'https://')):
        return urljoin(base_url, url)
    return url


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text with HTML tags
        
    Returns:
        Text without HTML tags
    """
    pattern = r'<[^>]+>'
    return re.sub(pattern, '', text)


def escape_html(text: str) -> str:
    """
    Escape HTML special characters.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text
    """
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text


def generate_random_string(length: int = 8, charset: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> str:
    """
    Generate a random string.
    
    Args:
        length: String length
        charset: Character set to use
        
    Returns:
        Random string
    """
    import secrets
    return ''.join(secrets.choice(charset) for _ in range(length))

