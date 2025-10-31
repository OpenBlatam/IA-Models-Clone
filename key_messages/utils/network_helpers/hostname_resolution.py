from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator
import structlog
import socket
import asyncio
import aiohttp
import dns.resolver
import dns.reversename
from concurrent.futures import ThreadPoolExecutor
        import time
        import time
        import re
from typing import Any, List, Dict, Optional
import logging
"""
Hostname resolution utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class HostnameResolutionInput(BaseModel):
    """Input model for hostname resolution."""
    hostname: str
    record_types: List[str] = ["A", "AAAA", "MX", "NS", "TXT"]
    timeout: float = 5.0
    use_dns_cache: bool = True
    
    @field_validator('hostname')
    def validate_hostname(cls, v) -> bool:
        if not v:
            raise ValueError("Hostname cannot be empty")
        return v
    
    @field_validator('record_types')
    def validate_record_types(cls, v) -> bool:
        valid_types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "PTR", "SOA"]
        for record_type in v:
            if record_type not in valid_types:
                raise ValueError(f"Invalid record type: {record_type}. Must be one of: {valid_types}")
        return v
    
    @field_validator('timeout')
    def validate_timeout(cls, v) -> bool:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

class ResolutionResult(BaseModel):
    """Result model for hostname resolution."""
    hostname: str
    ip_addresses: List[str]
    records: Dict[str, List[str]]
    reverse_dns: Optional[str] = None
    is_resolvable: bool
    resolution_time: float
    is_successful: bool
    error_message: Optional[str] = None

def resolve_hostname(input_data: HostnameResolutionInput) -> ResolutionResult:
    """
    RORO: Receive HostnameResolutionInput, return ResolutionResult
    
    Resolve hostname to IP addresses and DNS records.
    """
    try:
        start_time = time.time()
        
        # Resolve hostname to IP addresses
        ip_addresses = resolve_hostname_to_ips(input_data.hostname, input_data.timeout)
        
        # Get DNS records
        records = get_dns_records(input_data.hostname, input_data.record_types, input_data.timeout)
        
        # Get reverse DNS if we have IP addresses
        reverse_dns = None
        if ip_addresses:
            reverse_dns = get_reverse_dns(ip_addresses[0], input_data.timeout)
        
        resolution_time = time.time() - start_time
        
        return ResolutionResult(
            hostname=input_data.hostname,
            ip_addresses=ip_addresses,
            records=records,
            reverse_dns=reverse_dns,
            is_resolvable=len(ip_addresses) > 0,
            resolution_time=resolution_time,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Hostname resolution failed", error=str(e))
        return ResolutionResult(
            hostname=input_data.hostname,
            ip_addresses=[],
            records={},
            is_resolvable=False,
            resolution_time=0.0,
            is_successful=False,
            error_message=str(e)
        )

def resolve_hostname_to_ips(hostname: str, timeout: float) -> List[str]:
    """Resolve hostname to IP addresses."""
    try:
        ip_addresses = []
        
        # Try to get all addresses
        try:
            # Get IPv4 addresses
            ipv4_addresses = socket.gethostbyname_ex(hostname)[2]
            ip_addresses.extend(ipv4_addresses)
        except socket.gaierror:
            pass
        
        # Try to get IPv6 addresses
        try:
            # Use getaddrinfo for IPv6 support
            addrinfo = socket.getaddrinfo(hostname, None, socket.AF_INET6)
            for info in addrinfo:
                ipv6_addr = info[4][0]
                if ipv6_addr not in ip_addresses:
                    ip_addresses.append(ipv6_addr)
        except socket.gaierror:
            pass
        
        return ip_addresses
        
    except Exception as e:
        logger.error("IP resolution failed", error=str(e))
        return []

def get_dns_records(hostname: str, record_types: List[str], timeout: float) -> Dict[str, List[str]]:
    """Get DNS records for the hostname."""
    try:
        records = {}
        
        for record_type in record_types:
            try:
                # Set timeout for DNS resolver
                resolver = dns.resolver.Resolver()
                resolver.timeout = timeout
                resolver.lifetime = timeout
                
                # Query DNS records
                answers = resolver.resolve(hostname, record_type)
                
                # Extract record values
                record_values = []
                for answer in answers:
                    if record_type == "MX":
                        record_values.append(f"{answer.preference} {answer.exchange}")
                    elif record_type == "SOA":
                        record_values.append(str(answer))
                    else:
                        record_values.append(str(answer))
                
                records[record_type] = record_values
                
            except dns.resolver.NXDOMAIN:
                records[record_type] = []
            except dns.resolver.NoAnswer:
                records[record_type] = []
            except Exception as e:
                logger.warning(f"Failed to get {record_type} records", error=str(e))
                records[record_type] = []
        
        return records
        
    except Exception as e:
        logger.error("DNS record retrieval failed", error=str(e))
        return {}

def get_reverse_dns(ip_address: str, timeout: float) -> Optional[str]:
    """Get reverse DNS for an IP address."""
    try:
        # Set timeout for DNS resolver
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout
        
        # Create reverse lookup name
        reverse_name = dns.reversename.from_address(ip_address)
        
        # Query PTR record
        answers = resolver.resolve(reverse_name, "PTR")
        
        if answers:
            return str(answers[0])
        
        return None
        
    except Exception as e:
        logger.warning(f"Reverse DNS lookup failed for {ip_address}", error=str(e))
        return None

async def resolve_hostname_async(input_data: HostnameResolutionInput) -> ResolutionResult:
    """
    Async version of hostname resolution.
    """
    try:
        start_time = time.time()
        
        # Use ThreadPoolExecutor for blocking operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Resolve hostname to IP addresses
            ip_addresses = await loop.run_in_executor(
                executor, 
                resolve_hostname_to_ips, 
                input_data.hostname, 
                input_data.timeout
            )
            
            # Get DNS records
            records = await loop.run_in_executor(
                executor,
                get_dns_records,
                input_data.hostname,
                input_data.record_types,
                input_data.timeout
            )
            
            # Get reverse DNS if we have IP addresses
            reverse_dns = None
            if ip_addresses:
                reverse_dns = await loop.run_in_executor(
                    executor,
                    get_reverse_dns,
                    ip_addresses[0],
                    input_data.timeout
                )
        
        resolution_time = time.time() - start_time
        
        return ResolutionResult(
            hostname=input_data.hostname,
            ip_addresses=ip_addresses,
            records=records,
            reverse_dns=reverse_dns,
            is_resolvable=len(ip_addresses) > 0,
            resolution_time=resolution_time,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Async hostname resolution failed", error=str(e))
        return ResolutionResult(
            hostname=input_data.hostname,
            ip_addresses=[],
            records={},
            is_resolvable=False,
            resolution_time=0.0,
            is_successful=False,
            error_message=str(e)
        )

def validate_hostname_format(hostname: str) -> bool:
    """
    Validate hostname format.
    
    Args:
        hostname: Hostname to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        
        # Basic hostname validation regex
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        
        if not re.match(pattern, hostname):
            return False
        
        # Check length
        if len(hostname) > 253:
            return False
        
        # Check individual labels
        labels = hostname.split('.')
        for label in labels:
            if len(label) > 63:
                return False
        
        return True
        
    except Exception as e:
        logger.error("Hostname validation failed", error=str(e))
        return False 