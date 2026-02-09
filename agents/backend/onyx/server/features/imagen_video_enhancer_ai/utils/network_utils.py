"""
Network Utilities
=================

Advanced network utilities.
"""

import socket
import urllib.parse
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


class NetworkUtils:
    """Network utility functions."""
    
    @staticmethod
    def parse_url(url: str) -> Dict[str, Any]:
        """
        Parse URL into components.
        
        Args:
            url: URL string
            
        Returns:
            Parsed URL components
        """
        parsed = urlparse(url)
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password
        }
    
    @staticmethod
    def build_url(
        scheme: str,
        hostname: str,
        path: str = "/",
        port: Optional[int] = None,
        query: Optional[Dict[str, str]] = None,
        fragment: Optional[str] = None
    ) -> str:
        """
        Build URL from components.
        
        Args:
            scheme: URL scheme (http, https)
            hostname: Hostname
            path: Path
            port: Optional port
            query: Optional query parameters
            fragment: Optional fragment
            
        Returns:
            URL string
        """
        if port:
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname
        
        url = f"{scheme}://{netloc}{path}"
        
        if query:
            query_string = urllib.parse.urlencode(query)
            url += f"?{query_string}"
        
        if fragment:
            url += f"#{fragment}"
        
        return url
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL string
            
        Returns:
            True if URL is valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def join_url(base: str, path: str) -> str:
        """
        Join base URL with path.
        
        Args:
            base: Base URL
            path: Path to join
            
        Returns:
            Joined URL
        """
        return urljoin(base, path)
    
    @staticmethod
    def get_hostname(ip_address: str) -> Optional[str]:
        """
        Get hostname from IP address.
        
        Args:
            ip_address: IP address
            
        Returns:
            Hostname or None
        """
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except Exception:
            return None
    
    @staticmethod
    def get_ip_address(hostname: str) -> Optional[str]:
        """
        Get IP address from hostname.
        
        Args:
            hostname: Hostname
            
        Returns:
            IP address or None
        """
        try:
            return socket.gethostbyname(hostname)
        except Exception:
            return None
    
    @staticmethod
    def is_port_open(hostname: str, port: int, timeout: float = 1.0) -> bool:
        """
        Check if port is open.
        
        Args:
            hostname: Hostname
            port: Port number
            timeout: Connection timeout
            
        Returns:
            True if port is open
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((hostname, port))
            sock.close()
            return result == 0
        except Exception:
            return False




