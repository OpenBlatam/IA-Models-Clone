"""
String Utilities

Utility functions for string manipulation and formatting.
"""

from typing import Optional
import re


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        name: Camel case string
    
    Returns:
        Snake case string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        name: Snake case string
    
    Returns:
        Camel case string
    """
    components = name.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])


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


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Filename to sanitize
    
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    return sanitized


def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        size: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.2f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m {secs:.2f}s"



