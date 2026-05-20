from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator
import structlog
import socket
import ipaddress
import subprocess
import platform
from dataclasses import dataclass
from enum import Enum
        import urllib.request
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Network information utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class IPVersion(str, Enum):
    """IP version enumeration."""
    IPV4 = "IPv4"
    IPV6 = "IPv6"

@dataclass
class NetworkInterface:
    """Network interface information."""
    name: str
    ip_addresses: List[str]
    mac_address: Optional[str] = None
    is_up: bool = False
    mtu: Optional[int] = None

@dataclass
class IPInfo:
    """IP address information."""
    address: str
    version: IPVersion
    is_private: bool
    is_loopback: bool
    is_multicast: bool
    is_link_local: bool
    network: Optional[str] = None

class NetworkInfoInput(BaseModel):
    """Input model for network information retrieval."""
    include_interfaces: bool = True
    include_routing: bool = False
    include_dns: bool = False
    timeout: float = 5.0
    
    @field_validator('timeout')
    def validate_timeout(cls, v) -> bool:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

class IPValidationInput(BaseModel):
    """Input model for IP address validation."""
    ip_address: str
    allow_private: bool = True
    allow_loopback: bool = False
    
    @field_validator('ip_address')
    def validate_ip_address(cls, v) -> bool:
        if not v:
            raise ValueError("IP address cannot be empty")
        return v

class NetworkInfoResult(BaseModel):
    """Result model for network information."""
    hostname: str
    local_ip: str
    public_ip: Optional[str] = None
    interfaces: List[Dict[str, Any]]
    routing_table: Optional[List[Dict[str, Any]]] = None
    dns_servers: Optional[List[str]] = None
    is_successful: bool
    error_message: Optional[str] = None

class IPValidationResult(BaseModel):
    """Result model for IP validation."""
    ip_address: str
    is_valid: bool
    version: Optional[IPVersion] = None
    is_private: Optional[bool] = None
    is_loopback: Optional[bool] = None
    is_multicast: Optional[bool] = None
    is_link_local: Optional[bool] = None
    network: Optional[str] = None
    is_successful: bool
    error_message: Optional[str] = None

def get_network_info(input_data: NetworkInfoInput) -> NetworkInfoResult:
    """
    RORO: Receive NetworkInfoInput, return NetworkInfoResult
    
    Get comprehensive network information.
    """
    try:
        # Get basic network info
        hostname = socket.gethostname()
        local_ip = get_local_ip_address()
        
        # Get public IP
        public_ip = get_public_ip_address(input_data.timeout) if input_data.include_interfaces else None
        
        # Get network interfaces
        interfaces = []
        if input_data.include_interfaces:
            interfaces = get_network_interfaces()
        
        # Get routing table
        routing_table = None
        if input_data.include_routing:
            routing_table = get_routing_table()
        
        # Get DNS servers
        dns_servers = None
        if input_data.include_dns:
            dns_servers = get_dns_servers()
        
        return NetworkInfoResult(
            hostname=hostname,
            local_ip=local_ip,
            public_ip=public_ip,
            interfaces=interfaces,
            routing_table=routing_table,
            dns_servers=dns_servers,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Network info retrieval failed", error=str(e))
        return NetworkInfoResult(
            hostname="",
            local_ip="",
            interfaces=[],
            is_successful=False,
            error_message=str(e)
        )

def validate_ip_address(input_data: IPValidationInput) -> IPValidationResult:
    """
    RORO: Receive IPValidationInput, return IPValidationResult
    
    Validate and analyze an IP address.
    """
    try:
        # Try to parse the IP address
        ip_obj = ipaddress.ip_address(input_data.ip_address)
        
        # Determine IP version
        version = IPVersion.IPV6 if ip_obj.version == 6 else IPVersion.IPV4
        
        # Check various properties
        is_private = ip_obj.is_private
        is_loopback = ip_obj.is_loopback
        is_multicast = ip_obj.is_multicast
        is_link_local = ip_obj.is_link_local
        
        # Get network information
        network = str(ip_obj.network) if hasattr(ip_obj, 'network') else None
        
        # Determine if valid based on input criteria
        is_valid = True
        
        if not input_data.allow_private and is_private:
            is_valid = False
        
        if not input_data.allow_loopback and is_loopback:
            is_valid = False
        
        return IPValidationResult(
            ip_address=input_data.ip_address,
            is_valid=is_valid,
            version=version,
            is_private=is_private,
            is_loopback=is_loopback,
            is_multicast=is_multicast,
            is_link_local=is_link_local,
            network=network,
            is_successful=True
        )
        
    except ValueError as e:
        # Invalid IP address format
        return IPValidationResult(
            ip_address=input_data.ip_address,
            is_valid=False,
            is_successful=True,
            error_message=str(e)
        )
    except Exception as e:
        logger.error("IP validation failed", error=str(e))
        return IPValidationResult(
            ip_address=input_data.ip_address,
            is_valid=False,
            is_successful=False,
            error_message=str(e)
        )

def get_local_ip_address() -> str:
    """Get the local IP address."""
    try:
        # Create a socket to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a remote address (doesn't actually connect)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception as e:
        logger.warning("Failed to get local IP address", error=str(e))
        return "127.0.0.1"

def get_public_ip_address(timeout: float) -> Optional[str]:
    """Get the public IP address."""
    try:
        
        # Use a public IP service
        url = "https://api.ipify.org"
        
        with urllib.request.urlopen(url, timeout=timeout) as response:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            public_ip = response.read().decode('utf-8').strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return public_ip
            
    except Exception as e:
        logger.warning("Failed to get public IP address", error=str(e))
        return None

def get_network_interfaces() -> List[Dict[str, Any]]:
    """Get information about network interfaces."""
    try:
        interfaces = []
        
        # Get all network interfaces
        for interface_name, interface_addresses in socket.getaddrinfo(socket.gethostname(), None):
            try:
                # Get interface info
                interface_info = {
                    "name": interface_name[0],
                    "family": interface_name[1],
                    "type": interface_name[2],
                    "protocol": interface_name[3],
                    "address": interface_name[4][0]
                }
                
                # Get additional interface details
                if platform.system() == "Windows":
                    interface_info.update(get_windows_interface_info(interface_name[0]))
                else:
                    interface_info.update(get_unix_interface_info(interface_name[0]))
                
                interfaces.append(interface_info)
                
            except Exception as e:
                logger.warning(f"Failed to get info for interface {interface_name[0]}", error=str(e))
                continue
        
        return interfaces
        
    except Exception as e:
        logger.error("Failed to get network interfaces", error=str(e))
        return []

def get_windows_interface_info(interface_name: str) -> Dict[str, Any]:
    """Get Windows-specific interface information."""
    try:
        # Use netsh command to get interface info
        result = subprocess.run(
            ["netsh", "interface", "ip", "show", "config", interface_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        info = {}
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                line = line.strip()
                if "IP Address:" in line:
                    info["ip_address"] = line.split(":")[1].strip()
                elif "Subnet Prefix:" in line:
                    info["subnet"] = line.split(":")[1].strip()
                elif "Default Gateway:" in line:
                    info["gateway"] = line.split(":")[1].strip()
        
        return info
        
    except Exception as e:
        logger.warning(f"Failed to get Windows interface info for {interface_name}", error=str(e))
        return {}

def get_unix_interface_info(interface_name: str) -> Dict[str, Any]:
    """Get Unix-specific interface information."""
    try:
        # Use ip command to get interface info
        result = subprocess.run(
            ["ip", "addr", "show", interface_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        info = {}
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                line = line.strip()
                if "inet " in line:
                    parts = line.split()
                    info["ip_address"] = parts[1].split('/')[0]
                    info["subnet"] = parts[1].split('/')[1]
                elif "link/ether" in line:
                    parts = line.split()
                    info["mac_address"] = parts[1]
        
        return info
        
    except Exception as e:
        logger.warning(f"Failed to get Unix interface info for {interface_name}", error=str(e))
        return {}

def get_routing_table() -> List[Dict[str, Any]]:
    """Get the routing table."""
    try:
        routes = []
        
        if platform.system() == "Windows":
            # Use route print command on Windows
            result = subprocess.run(
                ["route", "print"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].replace('.', '').isdigit():
                        routes.append({
                            "destination": parts[0],
                            "netmask": parts[1],
                            "gateway": parts[2],
                            "interface": parts[3]
                        })
        else:
            # Use ip route command on Unix
            result = subprocess.run(
                ["ip", "route", "show"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        route = {"destination": parts[0]}
                        
                        for i, part in enumerate(parts[1:], 1):
                            if part == "via":
                                route["gateway"] = parts[i + 1]
                            elif part == "dev":
                                route["interface"] = parts[i + 1]
                        
                        routes.append(route)
        
        return routes
        
    except Exception as e:
        logger.error("Failed to get routing table", error=str(e))
        return []

def get_dns_servers() -> List[str]:
    """Get DNS server addresses."""
    try:
        dns_servers = []
        
        if platform.system() == "Windows":
            # Use nslookup command on Windows
            result = subprocess.run(
                ["nslookup", "localhost"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Server:" in line and "localhost" not in line:
                        server = line.split(":")[1].strip()
                        if server and server not in dns_servers:
                            dns_servers.append(server)
        else:
            # Read /etc/resolv.conf on Unix
            try:
                with open("/etc/resolv.conf", "r") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    for line in f:
                        if line.startswith("nameserver"):
                            server = line.split()[1].strip()
                            if server not in dns_servers:
                                dns_servers.append(server)
            except FileNotFoundError:
                pass
        
        return dns_servers
        
    except Exception as e:
        logger.error("Failed to get DNS servers", error=str(e))
        return []

def is_ip_in_subnet(ip_address: str, subnet: str) -> bool:
    """
    Check if an IP address is in a subnet.
    
    Args:
        ip_address: IP address to check
        subnet: Subnet in CIDR notation (e.g., "192.168.1.0/24")
        
    Returns:
        True if IP is in subnet, False otherwise
    """
    try:
        ip_obj = ipaddress.ip_address(ip_address)
        network_obj = ipaddress.ip_network(subnet, strict=False)
        return ip_obj in network_obj
        
    except Exception as e:
        logger.error("Subnet check failed", error=str(e))
        return False

def get_ip_range(subnet: str) -> List[str]:
    """
    Get all IP addresses in a subnet.
    
    Args:
        subnet: Subnet in CIDR notation
        
    Returns:
        List of IP addresses in the subnet
    """
    try:
        network_obj = ipaddress.ip_network(subnet, strict=False)
        return [str(ip) for ip in network_obj.hosts()]
        
    except Exception as e:
        logger.error("IP range generation failed", error=str(e))
        return [] 