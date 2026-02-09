"""
Sanitization utilities for professional documents module.

Functions for cleaning and sanitizing user input and data.
"""

import re
import html
from typing import Optional


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML content by escaping special characters.
    
    Args:
        text: Text that may contain HTML
        
    Returns:
        Escaped HTML string
    """
    return html.escape(text)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f/\\]', '', filename)
    
    # Replace spaces and multiple dots
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = re.sub(r'\.+', '.', sanitized)
    
    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip('._')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        sanitized = name[:max_name_length] + (f'.{ext}' if ext else '')
    
    return sanitized if sanitized else "untitled"


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text content by removing control characters.
    
    Args:
        text: Text to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized text
    """
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Normalize whitespace
    sanitized = re.sub(r'[ \t]+', ' ', sanitized)
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(' ', 1)[0] + '...'
    
    return sanitized.strip()


def sanitize_url(url: str) -> str:
    """
    Sanitize URL by removing dangerous characters.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL
    """
    # Remove control characters and dangerous protocols
    sanitized = re.sub(r'[\x00-\x1f]', '', url)
    
    # Check for dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
    url_lower = sanitized.lower()
    for protocol in dangerous_protocols:
        if url_lower.startswith(protocol):
            raise ValueError(f"Dangerous URL protocol detected: {protocol}")
    
    return sanitized


def sanitize_query(query: str) -> str:
    """
    Sanitize search query by removing dangerous characters.
    
    Args:
        query: Query string to sanitize
        
    Returns:
        Sanitized query
    """
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f]', '', query)
    
    # Remove SQL injection patterns (basic)
    sql_patterns = [
        r'(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+',
        r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b)\s+',
        r';\s*(DROP|DELETE|TRUNCATE)',
    ]
    
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()






