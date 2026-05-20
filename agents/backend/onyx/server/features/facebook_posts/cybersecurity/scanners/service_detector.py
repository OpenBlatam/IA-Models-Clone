from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import re
import asyncio
import socket
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import ssl
import json
from ..core import BaseConfig, ScanResult
from typing import Any, List, Dict, Optional
import logging
"""
Service detection and protocol identification utilities.
Def for CPU-bound analysis, async for network operations.
"""



@dataclass
class ServiceInfo:
    """Information about a detected service."""
    port: int
    service_name: str
    protocol: str
    version: Optional[str] = None
    banner: Optional[str] = None
    ssl_enabled: bool = False
    confidence: float = 0.0

@dataclass
class BannerInfo:
    """Banner information from service."""
    raw_banner: str
    service_name: Optional[str] = None
    version: Optional[str] = None
    os_info: Optional[str] = None
    additional_info: Dict[str, Any] = None

def get_service_signatures() -> Dict[str, List[Dict[str, Any]]]:
    """Get service detection signatures."""
    return {
        "http": [
            {"pattern": r"HTTP/\d+\.\d+", "confidence": 0.9},
            {"pattern": r"Server: .*", "confidence": 0.8},
            {"pattern": r"<title>.*</title>", "confidence": 0.7}
        ],
        "ssh": [
            {"pattern": r"SSH-\d+\.\d+", "confidence": 0.95},
            {"pattern": r"OpenSSH", "confidence": 0.9}
        ],
        "ftp": [
            {"pattern": r"FTP server ready", "confidence": 0.9},
            {"pattern": r"220.*FTP", "confidence": 0.8}
        ],
        "smtp": [
            {"pattern": r"220.*SMTP", "confidence": 0.9},
            {"pattern": r"ESMTP", "confidence": 0.8}
        ],
        "pop3": [
            {"pattern": r"POP3", "confidence": 0.9},
            {"pattern": r"OK.*POP3", "confidence": 0.8}
        ],
        "imap": [
            {"pattern": r"IMAP", "confidence": 0.9},
            {"pattern": r"OK.*IMAP", "confidence": 0.8}
        ],
        "mysql": [
            {"pattern": r"mysql_native_password", "confidence": 0.9},
            {"pattern": r"MySQL", "confidence": 0.8}
        ],
        "postgresql": [
            {"pattern": r"PostgreSQL", "confidence": 0.9},
            {"pattern": r"FATAL.*password", "confidence": 0.7}
        ],
        "redis": [
            {"pattern": r"ERR.*Redis", "confidence": 0.9},
            {"pattern": r"PONG", "confidence": 0.8}
        ]
    }

def extract_version_from_banner(banner: str) -> Optional[str]:
    """Extract version information from service banner."""
    version_patterns = [
        r"(\d+\.\d+\.\d+)",
        r"(\d+\.\d+)",
        r"version[:\s]+([^\s]+)",
        r"v([\d\.]+)"
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, banner, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def extract_os_info_from_banner(banner: str) -> Optional[str]:
    """Extract operating system information from banner."""
    os_patterns = [
        r"(Windows|Linux|Unix|BSD|macOS)",
        r"(Ubuntu|Debian|CentOS|RedHat|Fedora)",
        r"(Apache|nginx|IIS)"
    ]
    
    for pattern in os_patterns:
        match = re.search(pattern, banner, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def analyze_banner(banner: str) -> BannerInfo:
    """Analyze service banner for detailed information."""
    banner_lower = banner.lower()
    
    # Extract version
    version = extract_version_from_banner(banner)
    
    # Extract OS info
    os_info = extract_os_info_from_banner(banner)
    
    # Determine service name
    service_name = None
    signatures = get_service_signatures()
    
    for service, patterns in signatures.items():
        for sig in patterns:
            if re.search(sig["pattern"], banner, re.IGNORECASE):
                service_name = service
                break
        if service_name:
            break
    
    return BannerInfo(
        raw_banner=banner,
        service_name=service_name,
        version=version,
        os_info=os_info,
        additional_info={}
    )

def detect_service_by_banner(banner: str) -> Dict[str, Any]:
    """Detect service type and details from banner."""
    signatures = get_service_signatures()
    detected_services = []
    
    for service_name, patterns in signatures.items():
        max_confidence = 0.0
        for sig in patterns:
            if re.search(sig["pattern"], banner, re.IGNORECASE):
                max_confidence = max(max_confidence, sig["confidence"])
        
        if max_confidence > 0.5:
            detected_services.append({
                "service": service_name,
                "confidence": max_confidence
            })
    
    # Sort by confidence
    detected_services.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "services": detected_services,
        "primary_service": detected_services[0] if detected_services else None,
        "banner_analysis": analyze_banner(banner)
    }

async def grab_banner(host: str, port: int, timeout: float = 5.0) -> Optional[str]:
    """Grab banner from service asynchronously."""
    try:
        # Create socket connection
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        
        # Send common probes
        probes = [
            b'\r\n',
            b'GET / HTTP/1.0\r\n\r\n',
            b'HELP\r\n',
            b'VERSION\r\n',
            b'INFO\r\n'
        ]
        
        banner = None
        for probe in probes:
            try:
                writer.write(probe)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await writer.drain()
                
                # Read response
                response = await asyncio.wait_for(reader.read(1024), timeout=2.0)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if response:
                    banner = response.decode('utf-8', errors='ignore').strip()
                    break
            except Exception:
                continue
        
        writer.close()
        await writer.wait_closed()
        
        return banner
        
    except Exception:
        return None

async def detect_service(host: str, port: int, config: BaseConfig) -> ServiceInfo:
    """Detect service running on specified port."""
    # First try to grab banner
    banner = await grab_banner(host, port, config.timeout)
    
    if banner:
        # Analyze banner for service detection
        detection_result = detect_service_by_banner(banner)
        primary_service = detection_result["primary_service"]
        
        if primary_service:
            return ServiceInfo(
                port=port,
                service_name=primary_service["service"],
                protocol="tcp",
                banner=banner,
                confidence=primary_service["confidence"]
            )
    
    # Fallback to port-based detection
    common_services = {
        21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
        53: "dns", 80: "http", 110: "pop3", 143: "imap",
        443: "https", 993: "imaps", 995: "pop3s",
        3306: "mysql", 5432: "postgresql", 27017: "mongodb",
        6379: "redis", 8080: "http-proxy", 8443: "https-alt"
    }
    
    service_name = common_services.get(port, "unknown")
    
    return ServiceInfo(
        port=port,
        service_name=service_name,
        protocol="tcp",
        banner=banner,
        confidence=0.3 if service_name != "unknown" else 0.1
    )

def identify_protocol(port: int, banner: Optional[str] = None) -> str:
    """Identify protocol based on port and banner."""
    # SSL/TLS ports
    ssl_ports = {443, 993, 995, 8443, 9443}
    if port in ssl_ports:
        return "ssl/tls"
    
    # Common protocols
    protocol_map = {
        21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
        53: "dns", 80: "http", 110: "pop3", 143: "imap",
        3306: "mysql", 5432: "postgresql", 27017: "mongodb",
        6379: "redis", 8080: "http", 8443: "https"
    }
    
    if port in protocol_map:
        return protocol_map[port]
    
    # Try to identify from banner
    if banner:
        banner_lower = banner.lower()
        if "http" in banner_lower:
            return "http"
        elif "ftp" in banner_lower:
            return "ftp"
        elif "ssh" in banner_lower:
            return "ssh"
        elif "smtp" in banner_lower:
            return "smtp"
    
    return "unknown"

async def check_ssl_support(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if port supports SSL/TLS."""
    try:
        context = ssl.create_default_context()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=context),
            timeout=timeout
        )
        
        writer.close()
        await writer.wait_closed()
        return True
        
    except Exception:
        return False

# Named exports
__all__ = [
    'detect_service',
    'grab_banner',
    'identify_protocol',
    'analyze_banner',
    'detect_service_by_banner',
    'check_ssl_support',
    'ServiceInfo',
    'BannerInfo'
] 