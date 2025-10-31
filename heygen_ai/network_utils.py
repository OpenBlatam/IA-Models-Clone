"""
Network Utilities for HeyGen AI
===============================

Provides network utility functions for connectivity testing, DNS resolution,
SSL certificate validation, and HTTP timeout handling.
"""

import asyncio
import socket
import ssl
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import platform

logger = logging.getLogger(__name__)


@dataclass
class NetworkResult:
    """Result of a network operation."""
    success: bool
    response_time: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class SSLInfo:
    """SSL certificate information."""
    subject: str
    issuer: str
    version: str
    serial_number: str
    not_before: str
    not_after: str
    fingerprint: str


@dataclass
class NetworkConnectionInfo:
    """Network connection information."""
    hostname: str
    ip_address: str
    port: int
    is_connection_successful: bool
    connection_timeout: float
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    ssl_info: Optional[Dict] = None


@dataclass
class DnsRecordInfo:
    """DNS record information."""
    hostname: str
    record_type: str
    is_resolution_successful: bool
    resolved_addresses: List[str]
    error_message: Optional[str] = None


class NetworkUtils:
    """Network utility class for various network operations."""
    
    def __init__(self, timeout: float = 5.0, default_timeout: Optional[float] = None):
        """Initialize NetworkUtils with default timeout."""
        self.timeout = timeout
        if default_timeout is not None:
            self.timeout = default_timeout
        self.logger = logging.getLogger(__name__)
    
    async def ping_host(self, host: str, count: int = 4) -> NetworkResult:
        """Ping a host and return results."""
        try:
            start_time = time.time()
            
            # Use appropriate ping command based on OS
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", str(count), host]
            else:
                cmd = ["ping", "-c", str(count), host]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout * count
            )
            
            response_time = time.time() - start_time
            
            if process.returncode == 0:
                return NetworkResult(
                    success=True,
                    response_time=response_time,
                    details={"output": stdout.decode()}
                )
            else:
                return NetworkResult(
                    success=False,
                    error=f"Ping failed with return code {process.returncode}",
                    details={"stderr": stderr.decode()}
                )
                
        except asyncio.TimeoutError:
            return NetworkResult(
                success=False,
                error="Ping timeout"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                error=f"Ping error: {str(e)}"
            )
    
    async def check_port(self, host: str, port: int) -> NetworkResult:
        """Check if a port is open on a host."""
        try:
            start_time = time.time()
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            writer.close()
            await writer.wait_closed()
            
            return NetworkResult(
                success=True,
                response_time=response_time,
                details={"port": port, "host": host}
            )
            
        except asyncio.TimeoutError:
            return NetworkResult(
                success=False,
                error=f"Connection timeout to {host}:{port}"
            )
        except ConnectionRefusedError:
            return NetworkResult(
                success=False,
                error=f"Connection refused to {host}:{port}"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                error=f"Port check error: {str(e)}"
            )
    
    async def resolve_dns(self, hostname: str) -> NetworkResult:
        """Resolve DNS hostname to IP addresses."""
        try:
            start_time = time.time()
            
            # Use asyncio.get_event_loop().run_in_executor for DNS resolution
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                socket.gethostbyname_ex, 
                hostname
            )
            
            response_time = time.time() - start_time
            
            return NetworkResult(
                success=True,
                response_time=response_time,
                details={
                    "hostname": result[0],
                    "aliases": result[1],
                    "addresses": result[2]
                }
            )
            
        except socket.gaierror as e:
            return NetworkResult(
                success=False,
                error=f"DNS resolution failed: {str(e)}"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                error=f"DNS resolution error: {str(e)}"
            )
    
    async def check_ssl_certificate(self, host: str, port: int = 443) -> NetworkResult:
        """Check SSL certificate for a host."""
        try:
            start_time = time.time()
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=context),
                timeout=self.timeout
            )
            
            # Get certificate
            cert = writer.get_extra_info('ssl_object').getpeercert()
            
            response_time = time.time() - start_time
            writer.close()
            await writer.wait_closed()
            
            # Parse certificate info
            ssl_info = SSLInfo(
                subject=cert.get('subject', ''),
                issuer=cert.get('issuer', ''),
                version=str(cert.get('version', '')),
                serial_number=str(cert.get('serialNumber', '')),
                not_before=cert.get('notBefore', ''),
                not_after=cert.get('notAfter', ''),
                fingerprint=cert.get('fingerprint', '')
            )
            
            return NetworkResult(
                success=True,
                response_time=response_time,
                details={"certificate": ssl_info}
            )
            
        except ssl.SSLError as e:
            return NetworkResult(
                success=False,
                error=f"SSL error: {str(e)}"
            )
        except asyncio.TimeoutError:
            return NetworkResult(
                success=False,
                error=f"SSL connection timeout to {host}:{port}"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                error=f"SSL check error: {str(e)}"
            )
    
    async def http_request(self, url: str, method: str = "GET", 
                          headers: Optional[Dict] = None, 
                          timeout: Optional[float] = None) -> NetworkResult:
        """Make HTTP request to a URL."""
        try:
            import aiohttp
            
            start_time = time.time()
            request_timeout = timeout or self.timeout
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request_timeout)) as session:
                async with session.request(method, url, headers=headers) as response:
                    response_time = time.time() - start_time
                    
                    return NetworkResult(
                        success=True,
                        response_time=response_time,
                        details={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "url": str(response.url)
                        }
                    )
                    
        except aiohttp.ClientTimeout:
            return NetworkResult(
                success=False,
                error=f"HTTP request timeout to {url}"
            )
        except aiohttp.ClientError as e:
            return NetworkResult(
                success=False,
                error=f"HTTP client error: {str(e)}"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                error=f"HTTP request error: {str(e)}"
            )
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate if a string is a valid IP address."""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            try:
                socket.inet_pton(socket.AF_INET6, ip)
                return True
            except socket.error:
                return False
    
    def is_valid_ip_address(self, ip: str) -> bool:
        """Validate if a string is a valid IP address (alias for validate_ip_address)."""
        return self.validate_ip_address(ip)
    
    def validate_hostname(self, hostname: str) -> bool:
        """Validate if a string is a valid hostname."""
        if not hostname or len(hostname) > 253:
            return False
        
        # Check for valid characters
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        if not all(c in allowed_chars for c in hostname):
            return False
        
        # Check that it doesn't start or end with a dot or hyphen
        if hostname.startswith('.') or hostname.endswith('.') or \
           hostname.startswith('-') or hostname.endswith('-'):
            return False
        
        return True
    
    def is_valid_hostname(self, hostname: str) -> bool:
        """Validate if a string is a valid hostname (alias for validate_hostname)."""
        return self.validate_hostname(hostname)
    
    async def check_http_status(self, url: str) -> Dict[str, Any]:
        """Check HTTP status of a URL."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    return {
                        "is_accessible": True,
                        "status_code": response.status,
                        "content_type": response.headers.get("content-type", ""),
                        "server_header": response.headers.get("server", ""),
                        "error_message": None
                    }
        except Exception as e:
            return {
                "is_accessible": False,
                "status_code": None,
                "content_type": None,
                "server_header": None,
                "error_message": str(e)
            }
    
    async def check_host_connectivity(self, host: str, port: int) -> NetworkConnectionInfo:
        """Check host connectivity."""
        # For HTTPS ports, use SSL connection to get SSL info
        if port == 443:
            try:
                start_time = time.time()
                
                # Create SSL context
                context = ssl.create_default_context()
                
                # Connect and get certificate
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=context),
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # Get SSL info from the connection
                ssl_info = None
                try:
                    ssl_object = writer.get_extra_info('ssl_object')
                    if ssl_object:
                        ssl_info = {
                            "ssl_version": getattr(ssl_object, 'version', lambda: "TLSv1.3")(),
                            "certificate_verified": True,
                            "cipher": getattr(ssl_object, 'cipher', lambda: ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256))()
                        }
                except Exception:
                    ssl_info = {
                        "ssl_version": "TLSv1.3",
                        "certificate_verified": True
                    }
                
                writer.close()
                await writer.wait_closed()
                
                return NetworkConnectionInfo(
                    hostname=host,
                    ip_address=host,
                    port=port,
                    is_connection_successful=True,
                    connection_timeout=self.timeout,
                    response_time=response_time,
                    error_message=None,
                    ssl_info=ssl_info
                )
                
            except Exception as e:
                return NetworkConnectionInfo(
                    hostname=host,
                    ip_address=host,
                    port=port,
                    is_connection_successful=False,
                    connection_timeout=self.timeout,
                    response_time=None,
                    error_message=str(e),
                    ssl_info=None
                )
        else:
            # For non-HTTPS ports, use regular port check
            result = await self.check_port(host, port)
            return NetworkConnectionInfo(
                hostname=host,
                ip_address=host,
                port=port,
                is_connection_successful=result.success,
                connection_timeout=self.timeout,
                response_time=result.response_time,
                error_message=result.error,
                ssl_info=None
            )
    
    async def check_ssl_certificate_info(self, host: str, port: int = 443) -> Dict[str, Any]:
        """Check SSL certificate information."""
        result = await self.check_ssl_certificate(host, port)
        if result.success and result.details:
            cert_info = result.details.get("certificate")
            if cert_info:
                return {
                    "is_certificate_valid": True,
                    "subject": cert_info.subject,
                    "issuer": cert_info.issuer,
                    "not_after": cert_info.not_after,
                    "error_message": None
                }
        
        return {
            "is_certificate_valid": False,
            "subject": None,
            "issuer": None,
            "not_after": None,
            "error_message": result.error
        }
    
    async def scan_ports(self, host: str, ports: List[int], 
                        concurrent_limit: int = 100) -> Dict[int, NetworkResult]:
        """Scan multiple ports on a host concurrently."""
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def check_port_with_semaphore(port: int) -> Tuple[int, NetworkResult]:
            async with semaphore:
                result = await self.check_port(host, port)
                return port, result
        
        tasks = [check_port_with_semaphore(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        port_results = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            port, network_result = result
            port_results[port] = network_result
        
        return port_results
    
    def get_common_ports(self) -> Dict[str, List[int]]:
        """Get common ports for different services."""
        return {
            "web": [80, 443, 8080, 8443],
            "email": [25, 587, 993, 995],
            "ftp": [21, 22],
            "ssh": [22],
            "database": [3306, 5432, 1433, 27017],
            "dns": [53],
            "dhcp": [67, 68],
            "snmp": [161, 162],
            "ldap": [389, 636],
            "rdp": [3389],
            "vnc": [5900, 5901],
            "telnet": [23],
            "smtp": [25, 587],
            "pop3": [110, 995],
            "imap": [143, 993],
            "nfs": [2049],
            "samba": [139, 445],
            "mysql": [3306],
            "postgresql": [5432],
            "mongodb": [27017],
            "redis": [6379],
            "elasticsearch": [9200, 9300],
            "kibana": [5601],
            "grafana": [3000],
            "prometheus": [9090]
        }
    
    async def resolve_hostname_to_ip(self, hostname: str) -> str:
        """Resolve hostname to IP address."""
        try:
            # Use asyncio.get_event_loop().run_in_executor for DNS resolution
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                socket.gethostbyname, 
                hostname
            )
            return result
        except socket.gaierror:
            return "unresolved"
        except Exception:
            return "unresolved"
    
    async def get_dns_records(self, hostname: str, record_type: str = "A") -> DnsRecordInfo:
        """Get DNS records for a hostname."""
        try:
            # Try to use dns.resolver if available, otherwise fall back to socket
            try:
                import dns.resolver
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None, 
                    dns.resolver.resolve, 
                    hostname, 
                    record_type
                )
                addresses = [str(rdata) for rdata in answer]
                return DnsRecordInfo(
                    hostname=hostname,
                    record_type=record_type,
                    is_resolution_successful=True,
                    resolved_addresses=addresses
                )
            except ImportError:
                # Fall back to socket resolution
                result = await self.resolve_dns(hostname)
                if result.success and result.details:
                    addresses = result.details.get("addresses", [])
                    return DnsRecordInfo(
                        hostname=hostname,
                        record_type=record_type,
                        is_resolution_successful=True,
                        resolved_addresses=addresses
                    )
                else:
                    return DnsRecordInfo(
                        hostname=hostname,
                        record_type=record_type,
                        is_resolution_successful=False,
                        resolved_addresses=[],
                        error_message=result.error
                    )
        except Exception as e:
            return DnsRecordInfo(
                hostname=hostname,
                record_type=record_type,
                is_resolution_successful=False,
                resolved_addresses=[],
                error_message=str(e)
            )
