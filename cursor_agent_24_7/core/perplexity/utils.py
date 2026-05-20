"""
Utility Functions - Helper utilities for Perplexity system
==========================================================

Common utility functions used across the Perplexity modules.
"""

import re
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_citations(text: str) -> List[int]:
    """
    Extract citation numbers from text.
    
    Args:
        text: Text containing citations like [1][2][3]
        
    Returns:
        List of citation numbers
    """
    citations = re.findall(r'\[(\d+)\]', text)
    return [int(c) for c in citations]


def count_citations(text: str) -> int:
    """
    Count total number of citations in text.
    
    Args:
        text: Text to count citations in
        
    Returns:
        Total citation count
    """
    return len(extract_citations(text))


def remove_citations(text: str) -> str:
    """
    Remove all citations from text.
    
    Args:
        text: Text with citations
        
    Returns:
        Text without citations
    """
    return re.sub(r'\[\d+\]', '', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
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


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp as ISO string.
    
    Args:
        timestamp: Datetime object or None for current time
        
    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse ISO timestamp string to datetime.
    
    Args:
        timestamp_str: ISO formatted timestamp string
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse timestamp: {timestamp_str}")
        return None


def sanitize_query(query: str) -> str:
    """
    Sanitize query string for safe processing.
    
    Args:
        query: Query string to sanitize
        
    Returns:
        Sanitized query string
    """
    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
    # Normalize whitespace
    query = normalize_whitespace(query)
    return query.strip()


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time in minutes.
    
    Args:
        text: Text to estimate reading time for
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, int(word_count / words_per_minute))




