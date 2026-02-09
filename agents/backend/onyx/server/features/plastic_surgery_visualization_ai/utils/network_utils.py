"""Network utilities."""

import socket
from typing import Optional, Tuple
from urllib.parse import urlparse


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Check if port is open.
    
    Args:
        host: Host address
        port: Port number
        timeout: Timeout in seconds
        
    Returns:
        True if port is open
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_local_ip() -> str:
    """
    Get local IP address.
    
    Returns:
        Local IP address
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def parse_url(url: str) -> dict:
    """
    Parse URL into components.
    
    Args:
        url: URL string
        
    Returns:
        Dictionary with URL components
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
    }


def build_url(
    scheme: str,
    hostname: str,
    path: str = "/",
    port: Optional[int] = None,
    query: Optional[dict] = None,
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
    from urllib.parse import urlencode
    
    url = f"{scheme}://{hostname}"
    
    if port:
        url += f":{port}"
    
    url += path
    
    if query:
        url += f"?{urlencode(query)}"
    
    if fragment:
        url += f"#{fragment}"
    
    return url


def is_valid_ip(ip: str) -> bool:
    """
    Check if string is valid IP address.
    
    Args:
        ip: IP address string
        
    Returns:
        True if valid IP
    """
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


def is_valid_hostname(hostname: str) -> bool:
    """
    Check if string is valid hostname.
    
    Args:
        hostname: Hostname string
        
    Returns:
        True if valid hostname
    """
    if not hostname or len(hostname) > 253:
        return False
    
    if hostname[-1] == '.':
        hostname = hostname[:-1]
    
    parts = hostname.split('.')
    
    for part in parts:
        if not part or len(part) > 63:
            return False
        if not part[0].isalnum() or not part[-1].isalnum():
            return False
        if not all(c.isalnum() or c == '-' for c in part):
            return False
    
    return True

