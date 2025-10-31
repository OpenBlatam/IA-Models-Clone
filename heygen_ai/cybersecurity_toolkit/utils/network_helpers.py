from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import socket
import asyncio
import ipaddress
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from .structured_logger import get_logger, log_async_function_call, log_function_call
from typing import Any, List, Dict, Optional
"""
Network Helpers Module - Happy Path Pattern
==========================================

Network utility functions using happy path pattern:
- Early returns for error conditions
- Avoiding nested conditionals
- Keeping main success logic at the end
- Guard clauses for validation
- Clean, readable code structure
"""


# Import structured logger

# Get logger instance
logger = get_logger("network_helpers")

@log_function_call
def validate_ip_address(ip_string: str) -> Dict[str, Any]:
    """
    Validate IP address using happy path pattern with guard clauses.
    
    Args:
        ip_string: IP address string to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause 1: Check if IP address is provided
    if not ip_string:
        return {
            "is_valid": False,
            "error": "IP address is required",
            "error_type": "MissingIPAddress"
        }
    
    # Guard clause 2: Check if IP address is a string
    if not isinstance(ip_string, str):
        return {
            "is_valid": False,
            "error": "IP address must be a string",
            "error_type": "InvalidIPAddressType"
        }
    
    # Guard clause 3: Check IP address length
    if len(ip_string) > 45:  # IPv6 max length
        return {
            "is_valid": False,
            "error": "IP address too long",
            "error_type": "IPAddressTooLong"
        }
    
    # Happy path: Validate IP address
    try:
        ip_obj = ipaddress.ip_address(ip_string)
        
        validation_result = {
            "is_valid": True,
            "ip_address": str(ip_obj),
            "ip_version": ip_obj.version,
            "is_private": ip_obj.is_private,
            "is_loopback": ip_obj.is_loopback,
            "is_multicast": ip_obj.is_multicast,
            "is_reserved": ip_obj.is_reserved
        }
        
        logger.log_function_exit(
            "validate_ip_address",
            validation_result,
            context={"validation_result": "success"}
        )
        
        return validation_result
        
    except ValueError as e:
        return {
            "is_valid": False,
            "error": f"Invalid IP address: {str(e)}",
            "error_type": "InvalidIPAddress"
        }

@log_function_call
def validate_hostname(hostname: str) -> Dict[str, Any]:
    """
    Validate hostname using happy path pattern with guard clauses.
    
    Args:
        hostname: Hostname string to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause 1: Check if hostname is provided
    if not hostname:
        return {
            "is_valid": False,
            "error": "Hostname is required",
            "error_type": "MissingHostname"
        }
    
    # Guard clause 2: Check if hostname is a string
    if not isinstance(hostname, str):
        return {
            "is_valid": False,
            "error": "Hostname must be a string",
            "error_type": "InvalidHostnameType"
        }
    
    # Guard clause 3: Check hostname length
    if len(hostname) > 253:
        return {
            "is_valid": False,
            "error": "Hostname too long (max 253 characters)",
            "error_type": "HostnameTooLong"
        }
    
    # Guard clause 4: Check hostname format
    hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(hostname_pattern, hostname):
        return {
            "is_valid": False,
            "error": "Invalid hostname format",
            "error_type": "InvalidHostnameFormat"
        }
    
    # Guard clause 5: Check individual label lengths
    labels = hostname.split('.')
    for i, label in enumerate(labels):
        if len(label) > 63:
            return {
                "is_valid": False,
                "error": f"Label too long: {label} (max 63 characters)",
                "error_type": "LabelTooLong"
            }
    
    # Happy path: All validation passed
    validation_result = {
        "is_valid": True,
        "hostname": hostname.lower(),
        "labels": labels,
        "label_count": len(labels)
    }
    
    logger.log_function_exit(
        "validate_hostname",
        validation_result,
        context={"validation_result": "success"}
    )
    
    return validation_result

@log_function_call
def validate_network_target(target: str) -> Dict[str, Any]:
    """
    Validate network target using happy path pattern with guard clauses.
    
    Args:
        target: Target string to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause 1: Check if target is provided
    if not target:
        return {
            "is_valid": False,
            "error": "Target is required",
            "error_type": "MissingTarget"
        }
    
    # Guard clause 2: Check if target is a string
    if not isinstance(target, str):
        return {
            "is_valid": False,
            "error": "Target must be a string",
            "error_type": "InvalidTargetType"
        }
    
    # Guard clause 3: Try to validate as IP address first
    ip_validation = validate_ip_address(target)
    if ip_validation["is_valid"]:
        validation_result = {
            "is_valid": True,
            "target_type": "ip_address",
            "target": target,
            "ip_info": ip_validation
        }
        
        logger.log_function_exit(
            "validate_network_target",
            validation_result,
            context={"validation_result": "ip_address"}
        )
        
        return validation_result
    
    # Guard clause 4: Try to validate as hostname
    hostname_validation = validate_hostname(target)
    if hostname_validation["is_valid"]:
        validation_result = {
            "is_valid": True,
            "target_type": "hostname",
            "target": target.lower(),
            "hostname_info": hostname_validation
        }
        
        logger.log_function_exit(
            "validate_network_target",
            validation_result,
            context={"validation_result": "hostname"}
        )
        
        return validation_result
    
    # Guard clause 5: Target is neither valid IP nor hostname
    return {
        "is_valid": False,
        "error": "Target is neither a valid IP address nor hostname",
        "error_type": "InvalidTarget"
    }

@log_async_function_call
async def check_connectivity_async(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Check connectivity using happy path pattern with guard clauses.
    
    Args:
        host: Target host
        port: Target port
        timeout: Connection timeout
        
    Returns:
        Connectivity check result dictionary
    """
    # Guard clause 1: Validate host
    host_validation = validate_network_target(host)
    if not host_validation["is_valid"]:
        return {
            "success": False,
            "error": host_validation["error"],
            "error_type": host_validation["error_type"]
        }
    
    # Guard clause 2: Validate port
    if not isinstance(port, int) or port < 1 or port > 65535:
        return {
            "success": False,
            "error": "Port must be between 1 and 65535",
            "error_type": "InvalidPort"
        }
    
    # Guard clause 3: Validate timeout
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = 5.0
    
    # Happy path: Check connectivity
    check_start_time = asyncio.get_event_loop().time()
    
    try:
        # Create connection with timeout
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        
        # Calculate response time
        response_time = asyncio.get_event_loop().time() - check_start_time
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        
        result = {
            "success": True,
            "is_connectable": True,
            "host": host,
            "port": port,
            "response_time": response_time,
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "check_connectivity_async",
            result,
            context={"connectivity_result": "success"}
        )
        
        return result
        
    except asyncio.TimeoutError:
        result = {
            "success": True,
            "is_connectable": False,
            "host": host,
            "port": port,
            "error_message": "Connection timeout",
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "check_connectivity_async",
            result,
            context={"connectivity_result": "timeout"}
        )
        
        return result
        
    except ConnectionRefusedError:
        result = {
            "success": True,
            "is_connectable": False,
            "host": host,
            "port": port,
            "error_message": "Connection refused",
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "check_connectivity_async",
            result,
            context={"connectivity_result": "refused"}
        )
        
        return result
        
    except Exception as e:
        logger.log_error(
            e,
            "check_connectivity_async",
            {"host": host, "port": port, "timeout": timeout},
            {"connectivity_result": "error"}
        )
        
        return {
            "success": False,
            "is_connectable": False,
            "host": host,
            "port": port,
            "error_message": str(e),
            "check_timestamp": datetime.utcnow().isoformat()
        }

@log_function_call
def check_connectivity_sync(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Check connectivity synchronously using happy path pattern.
    
    Args:
        host: Target host
        port: Target port
        timeout: Connection timeout
        
    Returns:
        Connectivity check result dictionary
    """
    # Guard clause 1: Validate host
    host_validation = validate_network_target(host)
    if not host_validation["is_valid"]:
        return {
            "success": False,
            "error": host_validation["error"],
            "error_type": host_validation["error_type"]
        }
    
    # Guard clause 2: Validate port
    if not isinstance(port, int) or port < 1 or port > 65535:
        return {
            "success": False,
            "error": "Port must be between 1 and 65535",
            "error_type": "InvalidPort"
        }
    
    # Guard clause 3: Validate timeout
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = 5.0
    
    # Happy path: Check connectivity
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        result = sock.connect_ex((host, port))
        is_connectable = result == 0
        
        sock.close()
        
        connectivity_result = {
            "success": True,
            "is_connectable": is_connectable,
            "host": host,
            "port": port,
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "check_connectivity_sync",
            connectivity_result,
            context={"connectivity_result": "success" if is_connectable else "failed"}
        )
        
        return connectivity_result
        
    except socket.timeout:
        result = {
            "success": True,
            "is_connectable": False,
            "host": host,
            "port": port,
            "error_message": "Connection timeout",
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "check_connectivity_sync",
            result,
            context={"connectivity_result": "timeout"}
        )
        
        return result
        
    except Exception as e:
        logger.log_error(
            e,
            "check_connectivity_sync",
            {"host": host, "port": port, "timeout": timeout},
            {"connectivity_result": "error"}
        )
        
        return {
            "success": False,
            "is_connectable": False,
            "host": host,
            "port": port,
            "error_message": str(e),
            "check_timestamp": datetime.utcnow().isoformat()
        }

@log_function_call
def resolve_hostname_to_ip(hostname: str) -> Dict[str, Any]:
    """
    Resolve hostname to IP using happy path pattern.
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        Resolution result dictionary
    """
    # Guard clause 1: Validate hostname
    hostname_validation = validate_hostname(hostname)
    if not hostname_validation["is_valid"]:
        return {
            "success": False,
            "error": hostname_validation["error"],
            "error_type": hostname_validation["error_type"]
        }
    
    # Happy path: Resolve hostname
    try:
        # Resolve hostname
        ip_address = socket.gethostbyname(hostname)
        
        # Validate resolved IP
        ip_validation = validate_ip_address(ip_address)
        
        resolution_result = {
            "success": True,
            "hostname": hostname,
            "ip_address": ip_address,
            "ip_info": ip_validation,
            "resolution_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "resolve_hostname_to_ip",
            resolution_result,
            context={"resolution_result": "success"}
        )
        
        return resolution_result
        
    except socket.gaierror as e:
        result = {
            "success": False,
            "hostname": hostname,
            "error_message": f"Hostname resolution failed: {str(e)}",
            "error_type": "ResolutionError",
            "resolution_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "resolve_hostname_to_ip",
            result,
            context={"resolution_result": "gaierror"}
        )
        
        return result
        
    except Exception as e:
        logger.log_error(
            e,
            "resolve_hostname_to_ip",
            {"hostname": hostname},
            {"resolution_result": "error"}
        )
        
        return {
            "success": False,
            "hostname": hostname,
            "error_message": str(e),
            "error_type": "UnexpectedError",
            "resolution_timestamp": datetime.utcnow().isoformat()
        }

@log_function_call
def get_local_ip_address() -> Dict[str, Any]:
    """
    Get local IP address using happy path pattern.
    
    Returns:
        Local IP address result dictionary
    """
    # Happy path: Get local IP address
    try:
        # Create a socket to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a remote address (doesn't actually connect)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        
        # Validate local IP
        ip_validation = validate_ip_address(local_ip)
        
        result = {
            "success": True,
            "local_ip": local_ip,
            "ip_info": ip_validation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.log_function_exit(
            "get_local_ip_address",
            result,
            context={"local_ip_result": "success"}
        )
        
        return result
        
    except Exception as e:
        logger.log_error(
            e,
            "get_local_ip_address",
            {},
            {"local_ip_result": "error"}
        )
        
        return {
            "success": False,
            "error_message": str(e),
            "error_type": "LocalIPError",
            "timestamp": datetime.utcnow().isoformat()
        }

@log_function_call
def validate_port_range(start_port: int, end_port: int) -> Dict[str, Any]:
    """
    Validate port range using happy path pattern with guard clauses.
    
    Args:
        start_port: Start port number
        end_port: End port number
        
    Returns:
        Validation result dictionary
    """
    # Guard clause 1: Validate start port
    if not isinstance(start_port, int):
        return {
            "is_valid": False,
            "error": "Start port must be an integer",
            "error_type": "InvalidStartPortType"
        }
    
    if start_port < 1 or start_port > 65535:
        return {
            "is_valid": False,
            "error": "Start port must be between 1 and 65535",
            "error_type": "InvalidStartPortRange"
        }
    
    # Guard clause 2: Validate end port
    if not isinstance(end_port, int):
        return {
            "is_valid": False,
            "error": "End port must be an integer",
            "error_type": "InvalidEndPortType"
        }
    
    if end_port < 1 or end_port > 65535:
        return {
            "is_valid": False,
            "error": "End port must be between 1 and 65535",
            "error_type": "InvalidEndPortRange"
        }
    
    # Guard clause 3: Validate port range
    if start_port > end_port:
        return {
            "is_valid": False,
            "error": "Start port cannot be greater than end port",
            "error_type": "InvalidPortRange"
        }
    
    # Guard clause 4: Check range size
    port_count = end_port - start_port + 1
    if port_count > 1000:
        return {
            "is_valid": False,
            "error": "Port range too large (max 1000 ports)",
            "error_type": "PortRangeTooLarge"
        }
    
    # Happy path: All validation passed
    validation_result = {
        "is_valid": True,
        "start_port": start_port,
        "end_port": end_port,
        "port_count": port_count,
        "port_list": list(range(start_port, end_port + 1))
    }
    
    logger.log_function_exit(
        "validate_port_range",
        validation_result,
        context={"validation_result": "success"}
    )
    
    return validation_result

@log_function_call
def get_common_ports() -> Dict[str, Any]:
    """
    Get list of common ports using happy path pattern.
    
    Returns:
        Common ports dictionary
    """
    # Happy path: Return common ports
    common_ports = {
        "web": [80, 443, 8080, 8443],
        "email": [25, 110, 143, 465, 587, 993, 995],
        "file_transfer": [21, 22, 23, 69],
        "database": [3306, 5432, 6379, 27017],
        "remote_access": [22, 23, 3389, 5900],
        "gaming": [25565, 27015, 7777],
        "streaming": [1935, 554, 8000, 9000]
    }
    
    result = {
        "success": True,
        "common_ports": common_ports,
        "total_ports": sum(len(ports) for ports in common_ports.values()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.log_function_exit(
        "get_common_ports",
        result,
        context={"common_ports_result": "success"}
    )
    
    return result

# --- Named Exports ---

__all__ = [
    'validate_ip_address',
    'validate_hostname',
    'validate_network_target',
    'check_connectivity_async',
    'check_connectivity_sync',
    'resolve_hostname_to_ip',
    'get_local_ip_address',
    'validate_port_range',
    'get_common_ports'
] 