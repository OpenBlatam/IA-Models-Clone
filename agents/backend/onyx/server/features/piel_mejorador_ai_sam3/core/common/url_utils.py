"""
URL Utilities for Piel Mejorador AI SAM3
========================================

Unified URL parsing, construction, and manipulation utilities.
"""

import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urlunparse, urljoin, urlencode, parse_qs, parse_qsl, quote, unquote
from urllib.parse import ParseResult

logger = logging.getLogger(__name__)


class URLUtils:
    """Unified URL utilities."""
    
    @staticmethod
    def parse(url: str) -> ParseResult:
        """
        Parse URL into components.
        
        Args:
            url: URL string
            
        Returns:
            ParseResult with scheme, netloc, path, params, query, fragment
        """
        return urlparse(url)
    
    @staticmethod
    def build(
        scheme: str = "https",
        netloc: str = "",
        path: str = "",
        params: str = "",
        query: Optional[Dict[str, Any]] = None,
        fragment: str = ""
    ) -> str:
        """
        Build URL from components.
        
        Args:
            scheme: URL scheme (http, https, etc.)
            netloc: Network location (domain)
            path: URL path
            params: URL parameters
            query: Query parameters dictionary
            fragment: URL fragment
            
        Returns:
            Complete URL string
        """
        query_string = ""
        if query:
            # Filter None values
            filtered = {k: v for k, v in query.items() if v is not None}
            query_string = urlencode(filtered)
        
        return urlunparse((scheme, netloc, path, params, query_string, fragment))
    
    @staticmethod
    def join(base: str, *parts: str) -> str:
        """
        Join URL parts.
        
        Args:
            base: Base URL
            *parts: URL parts to join
            
        Returns:
            Joined URL
        """
        result = base.rstrip('/')
        for part in parts:
            part = part.strip('/')
            if part:
                result = f"{result}/{part}"
        return result
    
    @staticmethod
    def add_query_params(url: str, params: Dict[str, Any]) -> str:
        """
        Add query parameters to URL.
        
        Args:
            url: Base URL
            params: Query parameters to add
            
        Returns:
            URL with query parameters
        """
        parsed = urlparse(url)
        existing_params = parse_qs(parsed.query)
        
        # Merge with new params
        for key, value in params.items():
            if value is not None:
                existing_params[key] = [str(value)]
        
        # Build new query string
        new_query = urlencode(existing_params, doseq=True)
        
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
    
    @staticmethod
    def remove_query_params(url: str, *param_names: str) -> str:
        """
        Remove query parameters from URL.
        
        Args:
            url: URL
            *param_names: Parameter names to remove
            
        Returns:
            URL without specified parameters
        """
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        
        for param_name in param_names:
            params.pop(param_name, None)
        
        new_query = urlencode(params, doseq=True)
        
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
    
    @staticmethod
    def get_query_params(url: str) -> Dict[str, List[str]]:
        """
        Get query parameters from URL.
        
        Args:
            url: URL
            
        Returns:
            Dictionary of query parameters
        """
        parsed = urlparse(url)
        return parse_qs(parsed.query)
    
    @staticmethod
    def get_query_param(url: str, param_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get single query parameter value.
        
        Args:
            url: URL
            param_name: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        params = URLUtils.get_query_params(url)
        values = params.get(param_name, [])
        return values[0] if values else default
    
    @staticmethod
    def encode_path(path: str) -> str:
        """
        URL encode path component.
        
        Args:
            path: Path to encode
            
        Returns:
            Encoded path
        """
        # Encode each segment separately
        segments = path.split('/')
        encoded_segments = [quote(segment, safe='') for segment in segments]
        return '/'.join(encoded_segments)
    
    @staticmethod
    def decode_path(path: str) -> str:
        """
        URL decode path component.
        
        Args:
            path: Path to decode
            
        Returns:
            Decoded path
        """
        return unquote(path)
    
    @staticmethod
    def is_absolute(url: str) -> bool:
        """
        Check if URL is absolute.
        
        Args:
            url: URL to check
            
        Returns:
            True if absolute
        """
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    
    @staticmethod
    def is_secure(url: str) -> bool:
        """
        Check if URL uses secure protocol (https).
        
        Args:
            url: URL to check
            
        Returns:
            True if https
        """
        parsed = urlparse(url)
        return parsed.scheme.lower() == 'https'
    
    @staticmethod
    def get_domain(url: str) -> Optional[str]:
        """
        Get domain from URL.
        
        Args:
            url: URL
            
        Returns:
            Domain or None
        """
        parsed = urlparse(url)
        return parsed.netloc
    
    @staticmethod
    def get_path(url: str) -> str:
        """
        Get path from URL.
        
        Args:
            url: URL
            
        Returns:
            Path component
        """
        parsed = urlparse(url)
        return parsed.path
    
    @staticmethod
    def normalize(url: str, base_url: Optional[str] = None) -> str:
        """
        Normalize URL (join with base if provided).
        
        Args:
            url: URL or endpoint
            base_url: Optional base URL
            
        Returns:
            Normalized URL
        """
        if base_url:
            # Use urljoin for proper joining
            return urljoin(base_url.rstrip('/') + '/', url.lstrip('/'))
        return url
    
    @staticmethod
    def validate(url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False


# Convenience functions
def parse_url(url: str) -> ParseResult:
    """Parse URL."""
    return URLUtils.parse(url)


def build_url(**kwargs) -> str:
    """Build URL."""
    return URLUtils.build(**kwargs)


def join_url(base: str, *parts: str) -> str:
    """Join URL parts."""
    return URLUtils.join(base, *parts)


def add_query_params(url: str, params: Dict[str, Any]) -> str:
    """Add query parameters."""
    return URLUtils.add_query_params(url, params)


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """Normalize URL."""
    return URLUtils.normalize(url, base_url)


def validate_url(url: str) -> bool:
    """Validate URL."""
    return URLUtils.validate(url)




