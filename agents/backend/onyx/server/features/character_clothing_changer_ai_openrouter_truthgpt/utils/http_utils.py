"""
HTTP Utilities
==============

HTTP-related utilities.
"""

import re
from typing import Optional, Dict
from urllib.parse import urlparse, parse_qs


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def parse_url(url: str) -> Dict[str, any]:
    """
    Parse URL into components.
    
    Args:
        url: URL to parse
        
    Returns:
        Dictionary with URL components
    """
    parsed = urlparse(url)
    
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parse_qs(parsed.query),
        "fragment": parsed.fragment
    }


def build_url(
    scheme: str,
    netloc: str,
    path: str = "",
    query: Optional[Dict[str, str]] = None,
    fragment: Optional[str] = None
) -> str:
    """
    Build URL from components.
    
    Args:
        scheme: URL scheme (http, https)
        netloc: Network location (domain)
        path: URL path
        query: Query parameters
        fragment: URL fragment
        
    Returns:
        Complete URL string
    """
    url = f"{scheme}://{netloc}"
    
    if path:
        if not path.startswith('/'):
            url += '/'
        url += path
    
    if query:
        query_str = '&'.join(f"{k}={v}" for k, v in query.items())
        url += f"?{query_str}"
    
    if fragment:
        url += f"#{fragment}"
    
    return url


def get_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL
        
    Returns:
        Domain or None
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def is_https(url: str) -> bool:
    """
    Check if URL uses HTTPS.
    
    Args:
        url: URL
        
    Returns:
        True if HTTPS, False otherwise
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme == 'https'
    except Exception:
        return False


def sanitize_url(url: str) -> str:
    """
    Sanitize URL by removing dangerous characters.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL
    """
    # Remove dangerous protocols
    dangerous = ['javascript:', 'data:', 'vbscript:', 'file:']
    url_lower = url.lower()
    
    for protocol in dangerous:
        if url_lower.startswith(protocol):
            return ''
    
    return url

