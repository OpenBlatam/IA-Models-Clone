from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import re
from urllib.parse import urlparse, urljoin
from typing import Optional, List
from dataclasses import dataclass
        from urllib.parse import parse_qs
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
URL Value Object
Domain-Driven Design with validation and business logic
"""



@dataclass(frozen=True)
class URL:
    """
    URL value object with validation and business logic
    
    This value object encapsulates URL validation, normalization,
    and domain-specific business rules.
    """
    
    value: str
    
    def __post_init__(self) -> Any:
        """Validate URL after initialization"""
        if not self._is_valid_format():
            raise ValueError(f"Invalid URL format: {self.value}")
    
    def _is_valid_format(self) -> bool:
        """
        Check if URL has valid format
        
        Returns:
            bool: True if URL format is valid
        """
        try:
            result = urlparse(self.value)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def is_valid(self) -> bool:
        """
        Check if URL is valid
        
        Returns:
            bool: True if URL is valid
        """
        return self._is_valid_format()
    
    def get_scheme(self) -> str:
        """
        Get URL scheme
        
        Returns:
            str: URL scheme (http, https, etc.)
        """
        return urlparse(self.value).scheme
    
    def get_domain(self) -> str:
        """
        Get URL domain
        
        Returns:
            str: URL domain
        """
        return urlparse(self.value).netloc
    
    def get_path(self) -> str:
        """
        Get URL path
        
        Returns:
            str: URL path
        """
        return urlparse(self.value).path
    
    def get_query_params(self) -> dict:
        """
        Get URL query parameters
        
        Returns:
            dict: Query parameters
        """
        return parse_qs(urlparse(self.value).query)
    
    def get_fragment(self) -> str:
        """
        Get URL fragment
        
        Returns:
            str: URL fragment
        """
        return urlparse(self.value).fragment
    
    async def is_https(self) -> bool:
        """
        Check if URL uses HTTPS
        
        Returns:
            bool: True if HTTPS
        """
        return self.get_scheme().lower() == 'https'
    
    async def is_http(self) -> bool:
        """
        Check if URL uses HTTP
        
        Returns:
            bool: True if HTTP
        """
        return self.get_scheme().lower() == 'http'
    
    def is_secure(self) -> bool:
        """
        Check if URL is secure (HTTPS)
        
        Returns:
            bool: True if secure
        """
        return self.is_https()
    
    def get_base_url(self) -> str:
        """
        Get base URL (scheme + domain)
        
        Returns:
            str: Base URL
        """
        parsed = urlparse(self.value)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def normalize(self) -> "URL":
        """
        Normalize URL (remove trailing slash, etc.)
        
        Returns:
            URL: Normalized URL
        """
        normalized = self.value.rstrip('/')
        if not normalized:
            normalized = self.value
        return URL(normalized)
    
    def join(self, path: str) -> "URL":
        """
        Join URL with path
        
        Args:
            path: Path to join
            
        Returns:
            URL: Joined URL
        """
        joined = urljoin(self.value, path)
        return URL(joined)
    
    def is_same_domain(self, other: "URL") -> bool:
        """
        Check if URLs have same domain
        
        Args:
            other: Other URL to compare
            
        Returns:
            bool: True if same domain
        """
        return self.get_domain() == other.get_domain()
    
    def is_subdomain(self, other: "URL") -> bool:
        """
        Check if this URL is a subdomain of another
        
        Args:
            other: Parent domain URL
            
        Returns:
            bool: True if subdomain
        """
        this_domain = self.get_domain()
        other_domain = other.get_domain()
        
        return this_domain.endswith(f".{other_domain}")
    
    def get_subdomain(self) -> Optional[str]:
        """
        Get subdomain if exists
        
        Returns:
            Optional[str]: Subdomain or None
        """
        domain_parts = self.get_domain().split('.')
        if len(domain_parts) > 2:
            return domain_parts[0]
        return None
    
    def get_tld(self) -> str:
        """
        Get top-level domain
        
        Returns:
            str: Top-level domain
        """
        domain_parts = self.get_domain().split('.')
        return domain_parts[-1] if domain_parts else ""
    
    def is_localhost(self) -> bool:
        """
        Check if URL is localhost
        
        Returns:
            bool: True if localhost
        """
        domain = self.get_domain().lower()
        return domain in ['localhost', '127.0.0.1', '::1']
    
    def is_ip_address(self) -> bool:
        """
        Check if URL uses IP address
        
        Returns:
            bool: True if IP address
        """
        domain = self.get_domain()
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(ip_pattern, domain))
    
    def get_port(self) -> Optional[int]:
        """
        Get URL port if specified
        
        Returns:
            Optional[int]: Port number or None
        """
        parsed = urlparse(self.value)
        return parsed.port
    
    def has_query_params(self) -> bool:
        """
        Check if URL has query parameters
        
        Returns:
            bool: True if has query parameters
        """
        return bool(urlparse(self.value).query)
    
    def has_fragment(self) -> bool:
        """
        Check if URL has fragment
        
        Returns:
            bool: True if has fragment
        """
        return bool(urlparse(self.value).fragment)
    
    def is_relative(self) -> bool:
        """
        Check if URL is relative
        
        Returns:
            bool: True if relative URL
        """
        return not bool(urlparse(self.value).scheme)
    
    def is_absolute(self) -> bool:
        """
        Check if URL is absolute
        
        Returns:
            bool: True if absolute URL
        """
        return bool(urlparse(self.value).scheme)
    
    def get_depth(self) -> int:
        """
        Get URL depth (number of path segments)
        
        Returns:
            int: URL depth
        """
        path = self.get_path()
        if path == '/':
            return 0
        return len([seg for seg in path.split('/') if seg])
    
    def get_file_extension(self) -> Optional[str]:
        """
        Get file extension if exists
        
        Returns:
            Optional[str]: File extension or None
        """
        path = self.get_path()
        if '.' in path:
            return path.split('.')[-1].lower()
        return None
    
    def is_image_url(self) -> bool:
        """
        Check if URL points to an image
        
        Returns:
            bool: True if image URL
        """
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg']
        ext = self.get_file_extension()
        return ext in image_extensions
    
    def is_document_url(self) -> bool:
        """
        Check if URL points to a document
        
        Returns:
            bool: True if document URL
        """
        doc_extensions = ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt']
        ext = self.get_file_extension()
        return ext in doc_extensions
    
    def sanitize(self) -> "URL":
        """
        Sanitize URL (remove dangerous characters)
        
        Returns:
            URL: Sanitized URL
        """
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"{}|\\^`\[\]]', '', self.value)
        return URL(sanitized)
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"URL(value='{self.value}')"
    
    def __eq__(self, other) -> bool:
        """Compare URLs"""
        if not isinstance(other, URL):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash URL"""
        return hash(self.value)
    
    def __lt__(self, other) -> bool:
        """Compare URLs for sorting"""
        if not isinstance(other, URL):
            return NotImplemented
        return self.value < other.value
    
    @classmethod
    def create(cls, value: str) -> "URL":
        """
        Factory method to create URL
        
        Args:
            value: URL string
            
        Returns:
            URL: New URL instance
            
        Raises:
            ValueError: If URL is invalid
        """
        return cls(value)
    
    @classmethod
    def create_safe(cls, value: str) -> Optional["URL"]:
        """
        Safe factory method that returns None for invalid URLs
        
        Args:
            value: URL string
            
        Returns:
            Optional[URL]: URL instance or None if invalid
        """
        try:
            return cls(value)
        except ValueError:
            return None
    
    @classmethod
    async def create_https(cls, domain: str, path: str = "/") -> "URL":
        """
        Create HTTPS URL
        
        Args:
            domain: Domain name
            path: URL path
            
        Returns:
            URL: HTTPS URL
        """
        url = f"https://{domain.rstrip('/')}{path}"
        return cls(url)
    
    @classmethod
    async def create_http(cls, domain: str, path: str = "/") -> "URL":
        """
        Create HTTP URL
        
        Args:
            domain: Domain name
            path: URL path
            
        Returns:
            URL: HTTP URL
        """
        url = f"http://{domain.rstrip('/')}{path}"
        return cls(url) 