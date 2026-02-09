"""
Text processing helper functions

Functions for text sanitization, truncation, formatting, and summarization.
"""

from typing import Optional
from datetime import datetime
from ..models import PublishedChat
from .validation_common import is_empty_string


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitizes text by removing extra spaces and limiting length.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (optional)
        
    Returns:
        Sanitized text
    """
    if is_empty_string(text):
        return ""
    
    sanitized = text.strip()
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()
    
    return sanitized


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncates text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formats a datetime to string.
    
    Args:
        dt: Datetime to format
        format_str: Desired format
        
    Returns:
        Formatted string
    """
    if not dt:
        return ""
    
    return dt.strftime(format_str)


def get_chat_summary(chat: PublishedChat, max_length: int = 150) -> str:
    """
    Gets a summary of the chat.
    
    Args:
        chat: Chat to get summary from
        max_length: Maximum length of summary
        
    Returns:
        Chat summary
    """
    if chat.description:
        return truncate_text(chat.description, max_length)
    
    if chat.chat_content:
        # Try to extract first lines from content
        content = chat.chat_content.strip()
        if len(content) > max_length:
            return truncate_text(content, max_length)
        return content
    
    return chat.title


def slugify(text: str) -> str:
    """
    Converts text to slug format.
    
    Args:
        text: Text to convert
        
    Returns:
        Slug string
    """
    if is_empty_string(text):
        return ""
    
    import re
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def calculate_time_ago(dt: datetime) -> str:
    """
    Calculates time elapsed since a datetime.
    
    Args:
        dt: Reference datetime
        
    Returns:
        Human-readable time string (e.g., "2 hours ago")
    """
    if not dt:
        return ""
    
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"











