from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import socket
import asyncio
import ssl
import time
import struct
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import aiohttp
import aiofiles
import dns.resolver
import dns.reversename
    from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
"""
Network utilities with proper async/def distinction.
Async for network operations, def for CPU-bound analysis.
"""


@dataclass
class NetworkConfig:
    """Configuration for network operations."""
    timeout: float = 5.0
    max_retries: int = 3
    user_agent: str = "SecurityScanner/1.0"
    verify_ssl: bool = True
    follow_redirects: bool = True

@dataclass
class ConnectionResult:
    """Result of connection test."""
    host: str
    port: int
    is_connected: bool
    response_time: float
    protocol: str
    ssl_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class SSLCertificateInfo:
    """SSL certificate information."""
    subject: str
    issuer: str
    valid_from: str
    valid_until: str
    serial_number: str
    fingerprint: str

async def check_connection(host: str, port: int, config: NetworkConfig) -> ConnectionResult:
    """Check connection to host:port asynchronously."""
    start_time = time.time()
    
    try:
        # Create socket connection
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=config.timeout
        )
        
        response_time = time.time() - start_time
        protocol = "tcp"
        ssl_info = None
        
        # Check if SSL/TLS
        if port in [443, 993, 995, 8443]:
            try:
                context = ssl.create_default_context()
                if not config.verify_ssl:
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                
                ssl_info = await get_ssl_info(host, port, context)
                protocol = "ssl/tls"
            except Exception as e:
                ssl_info = {"error": str(e)}
        
        writer.close()
        await writer.wait_closed()
        
        return ConnectionResult(
            host=host,
            port=port,
            is_connected=True,
            response_time=response_time,
            protocol=protocol,
            ssl_info=ssl_info
        )
        
    except Exception as e:
        return ConnectionResult(
            host=host,
            port=port,
            is_connected=False,
            response_time=time.time() - start_time,
            protocol="tcp",
            error_message=str(e)
        )

async def get_ssl_info(host: str, port: int, context: ssl.SSLContext) -> Dict[str, Any]:
    """Get SSL certificate information."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            ssock.connect((host, port))
            cert = ssock.getpeercert()
            
            return {
                "subject": dict(x[0] for x in cert['subject']),
                "issuer": dict(x[0] for x in cert['issuer']),
                "version": cert['version'],
                "serial_number": cert['serialNumber'],
                "not_before": cert['notBefore'],
                "not_after": cert['notAfter'],
                "san": cert.get('subjectAltName', []),
                "cipher": ssock.cipher()
            }
    except Exception as e:
        return {"error": str(e)}

async def validate_url(url: str, config: NetworkConfig) -> Dict[str, Any]:
    """Validate URL accessibility and security."""
    start_time = time.time()
    
    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        headers = {"User-Agent": config.user_agent}
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, ssl=config.verify_ssl, 
                                 allow_redirects=config.follow_redirects) as response:
                
                response_time = time.time() - start_time
                
                return {
                    "url": url,
                    "status_code": response.status,
                    "response_time": response_time,
                    "headers": dict(response.headers),
                    "is_accessible": 200 <= response.status < 400,
                    "content_type": response.headers.get('content-type', ''),
                    "server": response.headers.get('server', ''),
                    "security_headers": extract_security_headers(response.headers)
                }
                
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "response_time": time.time() - start_time,
            "is_accessible": False,
            "error": str(e)
        }

def extract_security_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Extract security-related headers."""
    security_headers = [
        'strict-transport-security',
        'content-security-policy',
        'x-frame-options',
        'x-content-type-options',
        'x-xss-protection',
        'referrer-policy',
        'permissions-policy'
    ]
    
    return {
        header: headers.get(header, '')
        for header in security_headers
        if headers.get(header)
    }

async def test_ssl_certificate(host: str, port: int = 443) -> Dict[str, Any]:
    """Test SSL certificate validity and security."""
    try:
        context = ssl.create_default_context()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            ssock.connect((host, port))
            cert = ssock.getpeercert()
            
            # Analyze certificate
            cert_info = analyze_certificate(cert)
            
            return {
                "host": host,
                "port": port,
                "is_valid": True,
                "certificate_info": cert_info,
                "cipher_suite": ssock.cipher(),
                "protocol_version": ssock.version()
            }
            
    except Exception as e:
        return {
            "host": host,
            "port": port,
            "is_valid": False,
            "error": str(e)
        }

def analyze_certificate(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze SSL certificate for security issues."""
    
    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
    not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
    now = datetime.now()
    
    days_until_expiry = (not_after - now).days
    days_since_issued = (now - not_before).days
    
    # Check for security issues
    issues = []
    if days_until_expiry < 30:
        issues.append("Certificate expires soon")
    if days_until_expiry < 0:
        issues.append("Certificate expired")
    if days_since_issued < 1:
        issues.append("Certificate too new (potential clock skew)")
    
    # Check for weak algorithms
    if 'sha1' in str(cert).lower():
        issues.append("Uses SHA1 (weak)")
    
    return {
        "subject": dict(x[0] for x in cert['subject']),
        "issuer": dict(x[0] for x in cert['issuer']),
        "valid_from": cert['notBefore'],
        "valid_until": cert['notAfter'],
        "days_until_expiry": days_until_expiry,
        "days_since_issued": days_since_issued,
        "serial_number": cert['serialNumber'],
        "version": cert['version'],
        "san": cert.get('subjectAltName', []),
        "issues": issues,
        "is_secure": len(issues) == 0
    }

async def monitor_bandwidth(host: str, port: int, duration: int = 10) -> Dict[str, Any]:
    """Monitor bandwidth usage for a connection."""
    start_time = time.time()
    bytes_sent = 0
    bytes_received = 0
    
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0
        )
        
        # Send test data
        test_data = b"X" * 1024  # 1KB test data
        
        while time.time() - start_time < duration:
            writer.write(test_data)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await writer.drain()
            bytes_sent += len(test_data)
            
            # Try to receive data
            try:
                received = await asyncio.wait_for(reader.read(1024), timeout=1.0)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                bytes_received += len(received)
            except asyncio.TimeoutError:
                pass
        
        writer.close()
        await writer.wait_closed()
        
        elapsed_time = time.time() - start_time
        
        return {
            "host": host,
            "port": port,
            "duration": elapsed_time,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "send_rate": bytes_sent / elapsed_time,
            "receive_rate": bytes_received / elapsed_time,
            "total_bandwidth": (bytes_sent + bytes_received) / elapsed_time
        }
        
    except Exception as e:
        return {
            "host": host,
            "port": port,
            "error": str(e),
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received
        }

async def resolve_dns(hostname: str) -> Dict[str, Any]:
    """Resolve DNS records for hostname."""
    try:
        # A record
        a_records = []
        try:
            answers = dns.resolver.resolve(hostname, 'A')
            a_records = [str(rdata) for rdata in answers]
        except Exception:
            pass
        
        # AAAA record (IPv6)
        aaaa_records = []
        try:
            answers = dns.resolver.resolve(hostname, 'AAAA')
            aaaa_records = [str(rdata) for rdata in answers]
        except Exception:
            pass
        
        # MX record
        mx_records = []
        try:
            answers = dns.resolver.resolve(hostname, 'MX')
            mx_records = [str(rdata.exchange) for rdata in answers]
        except Exception:
            pass
        
        # TXT record
        txt_records = []
        try:
            answers = dns.resolver.resolve(hostname, 'TXT')
            txt_records = [str(rdata) for rdata in answers]
        except Exception:
            pass
        
        return {
            "hostname": hostname,
            "a_records": a_records,
            "aaaa_records": aaaa_records,
            "mx_records": mx_records,
            "txt_records": txt_records,
            "has_records": bool(a_records or aaaa_records or mx_records or txt_records)
        }
        
    except Exception as e:
        return {
            "hostname": hostname,
            "error": str(e),
            "has_records": False
        }

def analyze_network_traffic(packet_data: bytes) -> Dict[str, Any]:
    """Analyze network packet data for security issues."""
    if len(packet_data) < 20:
        return {"error": "Packet too small"}
    
    # Basic IP header analysis
    version = (packet_data[0] >> 4) & 0xF
    header_length = (packet_data[0] & 0xF) * 4
    ttl = packet_data[8]
    protocol = packet_data[9]
    
    issues = []
    if ttl < 32:
        issues.append("Low TTL (potential spoofing)")
    if header_length < 20:
        issues.append("Invalid header length")
    
    return {
        "version": version,
        "header_length": header_length,
        "ttl": ttl,
        "protocol": protocol,
        "packet_size": len(packet_data),
        "issues": issues,
        "is_suspicious": len(issues) > 0
    }

# Named exports for main functionality
__all__ = [
    'check_connection',
    'validate_url',
    'test_ssl_certificate',
    'monitor_bandwidth',
    'resolve_dns',
    'analyze_network_traffic',
    'NetworkConfig',
    'ConnectionResult',
    'SSLCertificateInfo'
] 