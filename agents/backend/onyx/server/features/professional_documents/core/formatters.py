"""
Formatting utilities for professional documents.

Helper functions for formatting various data types and values.
"""

from typing import Optional
from datetime import datetime
from .constants import WORDS_PER_PAGE


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB", "500 KB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1.5s", "2m 30s")
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def format_word_count(word_count: int) -> str:
    """
    Format word count with thousands separator.
    
    Args:
        word_count: Number of words
        
    Returns:
        Formatted string (e.g., "1,234 words")
    """
    return f"{word_count:,} words"


def format_page_count(page_count: int, word_count: Optional[int] = None) -> str:
    """
    Format page count with context.
    
    Args:
        page_count: Number of pages
        word_count: Optional word count for additional context
        
    Returns:
        Formatted string (e.g., "5 pages (~1,375 words)")
    """
    if word_count:
        return f"{page_count} pages (~{format_word_count(word_count)})"
    return f"{page_count} page{'s' if page_count != 1 else ''}"


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        format_str: Format string (default: ISO-like format)
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago").
    
    Args:
        dt: Datetime object
        
    Returns:
        Relative time string
    """
    now = datetime.utcnow()
    delta = now - dt
    
    if delta.total_seconds() < 60:
        return "just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days < 7:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    elif delta.days < 30:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif delta.days < 365:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"






