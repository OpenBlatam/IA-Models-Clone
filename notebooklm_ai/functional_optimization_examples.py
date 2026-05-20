from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
import secrets
from pathlib import Path
from functools import wraps, partial
from contextlib import asynccontextmanager
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
    import asyncpg
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.responses import JSONResponse
import structlog
    import structlog
        from urllib.parse import urlparse
        from urllib.parse import urlparse
        from urllib.parse import urlparse
        import socket
        import socket
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Functional Optimization Examples - Python/Cybersecurity Best Practices
🚀 Demonstrates functional programming, descriptive naming, and modular architecture
⚡ Uses guard clauses, early returns, and comprehensive error handling
🎯 Follows RORO pattern and proper async/sync function usage
"""


# Custom exception classes for cybersecurity operations
class SecurityToolError(Exception):
    """Base exception for security tool operations."""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code or "SECURITY_TOOL_ERROR"
        self.details = details or {}

class InvalidTargetError(SecurityToolError):
    """Raised when target address/hostname is invalid."""
    def __init__(self, target: str, reason: str = None):
        
    """__init__ function."""
message = f"Invalid target: {target}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "INVALID_TARGET", {"target": target, "reason": reason})

class PortScanError(SecurityToolError):
    """Raised when port scanning fails."""
    def __init__(self, target: str, port: int, reason: str = None):
        
    """__init__ function."""
message = f"Port scan failed for {target}:{port}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "PORT_SCAN_ERROR", {"target": target, "port": port, "reason": reason})

class VulnerabilityScanError(SecurityToolError):
    """Raised when vulnerability scanning fails."""
    def __init__(self, target_url: str, scan_type: str, reason: str = None):
        
    """__init__ function."""
message = f"Vulnerability scan failed for {target_url} ({scan_type})"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "VULNERABILITY_SCAN_ERROR", {"target_url": target_url, "scan_type": scan_type, "reason": reason})

class NetworkTimeoutError(SecurityToolError):
    """Raised when network operations timeout."""
    def __init__(self, operation: str, timeout_seconds: float, target: str = None):
        
    """__init__ function."""
message = f"Network timeout: {operation} exceeded {timeout_seconds}s"
        if target:
            message += f" for {target}"
        super().__init__(message, "NETWORK_TIMEOUT", {"operation": operation, "timeout_seconds": timeout_seconds, "target": target})

class AuthenticationError(SecurityToolError):
    """Raised when authentication fails."""
    def __init__(self, service: str, reason: str = None):
        
    """__init__ function."""
message = f"Authentication failed for {service}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "AUTHENTICATION_ERROR", {"service": service, "reason": reason})

class ConfigurationError(SecurityToolError):
    """Raised when configuration is invalid."""
    def __init__(self, config_key: str, reason: str = None):
        
    """__init__ function."""
message = f"Configuration error: {config_key}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "CONFIGURATION_ERROR", {"config_key": config_key, "reason": reason})

class RateLimitError(SecurityToolError):
    """Raised when rate limits are exceeded."""
    def __init__(self, service: str, retry_after: int = None):
        
    """__init__ function."""
message = f"Rate limit exceeded for {service}"
        if retry_after:
            message += f" - Retry after {retry_after} seconds"
        super().__init__(message, "RATE_LIMIT_ERROR", {"service": service, "retry_after": retry_after})

# Pydantic v2 imports for validation
try:
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Async database imports
try:
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# FastAPI imports
try:
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Configure structured logging

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Fallback to standard logging if structlog not available
try:
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR VALIDATION
# ============================================================================

if PYDANTIC_AVAILABLE:
    class ScanConfig(BaseModel):
        """Configuration for security scanning operations."""
        target_host: str = Field(..., description="Target host to scan")
        port_range: str = Field(default="1-1024", description="Port range to scan")
        scan_type: str = Field(default="tcp", description="Type of scan (tcp/udp)")
        timeout_seconds: int = Field(default=30, ge=1, le=300, description="Scan timeout")
        max_workers: int = Field(default=10, ge=1, le=100, description="Max concurrent workers")
        is_verbose: bool = Field(default=False, description="Verbose output")
        
        @field_validator('target_host')
        @classmethod
        def validate_target_host(cls, v: str) -> str:
            if not v or not v.strip(): raise ValueError("Target host cannot be empty")
            return v.strip().lower()
        
        @field_validator('scan_type')
        @classmethod
        def validate_scan_type(cls, v: str) -> str:
            valid_types = {"tcp", "udp", "syn", "connect"}
            if v not in valid_types: raise ValueError(f"Invalid scan type: {v}")
            return v.lower()

    class ScanResult(BaseModel):
        """Result of a security scan operation."""
        target_host: str
        port: int
        is_open: bool
        service_name: Optional[str] = None
        banner: Optional[str] = None
        scan_timestamp: datetime = Field(default_factory=datetime.now)
        scan_duration_ms: float = 0.0

    class VulnerabilityReport(BaseModel):
        """Vulnerability assessment report."""
        target_host: str
        vulnerability_id: str
        severity: str = Field(..., description="Critical/High/Medium/Low")
        description: str
        cve_id: Optional[str] = None
        cvss_score: Optional[float] = None
        remediation: Optional[str] = None
        discovered_at: datetime = Field(default_factory=datetime.now)

# ============================================================================
# FUNCTIONAL UTILITY FUNCTIONS
# ============================================================================

def is_valid_ip_address(ip_address: str) -> bool:
    """Validate IP address format using functional approach."""
    if not ip_address or not isinstance(ip_address, str): 
        logger.warning("Invalid IP address input", 
                      module="validation", 
                      function="is_valid_ip_address",
                      input_type=type(ip_address).__name__,
                      input_value=str(ip_address)[:50])
        return False
    
    parts = ip_address.split('.')
    if len(parts) != 4: 
        logger.warning("Invalid IP address format", 
                      module="validation", 
                      function="is_valid_ip_address",
                      input_value=ip_address,
                      parts_count=len(parts))
        return False
    
    try:
        is_valid = all(
            part.isdigit() and 0 <= int(part) <= 255 
            for part in parts
        )
        if not is_valid:
            logger.warning("Invalid IP address octets", 
                          module="validation", 
                          function="is_valid_ip_address",
                          input_value=ip_address,
                          parts=parts)
        return is_valid
    except (ValueError, TypeError) as e:
        logger.error("Exception validating IP address", 
                    module="validation", 
                    function="is_valid_ip_address",
                    input_value=ip_address,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        return False

def validate_target_with_exception(target: str) -> None:
    """Validate target and raise specific exceptions for invalid inputs."""
    if not target or not isinstance(target, str):
        raise InvalidTargetError(target, "Target must be a non-empty string")
    
    # Remove protocol prefix if present
    clean_target = target.replace('http://', '').replace('https://', '').replace('ftp://', '')
    
    # Check for IP address
    if is_valid_ip_address(clean_target):
        return
    
    # Check for IPv6 address
    if is_valid_ipv6_address(clean_target):
        return
    
    # Check for hostname
    if is_valid_hostname(clean_target):
        return
    
    raise InvalidTargetError(target, "Not a valid IP address, IPv6 address, or hostname")

def is_valid_ipv6_address(ipv6_address: str) -> bool:
    """Validate IPv6 address format."""
    if not ipv6_address or not isinstance(ipv6_address, str): return False
    
    # Basic IPv6 validation
    if '::' in ipv6_address:
        # Handle compressed format
        parts = ipv6_address.split('::')
        if len(parts) > 2: return False
    else:
        parts = ipv6_address.split(':')
        if len(parts) != 8: return False
    
    try:
        # Validate hex format
        for part in ipv6_address.split(':'):
            if part and not all(c in '0123456789abcdefABCDEF' for c in part):
                return False
        return True
    except Exception:
        return False

def is_valid_target_address(target: str) -> bool:
    """Validate target address (IP, IPv6, or hostname)."""
    if not target or not isinstance(target, str): return False
    
    # Remove protocol prefix if present
    target = target.replace('http://', '').replace('https://', '').replace('ftp://', '')
    
    # Check for IP address
    if is_valid_ip_address(target): return True
    
    # Check for IPv6 address
    if is_valid_ipv6_address(target): return True
    
    # Check for hostname
    if is_valid_hostname(target): return True
    
    return False

def is_valid_port_number(port: int) -> bool:
    """Validate port number range."""
    if not isinstance(port, int): return False
    return 1 <= port <= 65535

def is_valid_hostname(hostname: str) -> bool:
    """Validate hostname format."""
    if not hostname or not isinstance(hostname, str): return False
    if len(hostname) > 253: return False
    
    # Check for valid characters
    valid_chars = set('abcdefghijklmnopqrstuvwxyz0123456789.-')
    return all(c.lower() in valid_chars for c in hostname)

def parse_port_range(port_range: str) -> List[int]:
    """Parse port range string into list of ports."""
    if not port_range or not isinstance(port_range, str): return []
    
    try:
        if '-' in port_range:
            start, end = map(int, port_range.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(port_range)]
    except (ValueError, TypeError):
        return []

def generate_scan_id() -> str:
    """Generate unique scan identifier."""
    return f"scan_{secrets.token_hex(8)}_{int(datetime.now().timestamp())}"

def calculate_scan_duration(start_time: datetime, end_time: datetime) -> float:
    """Calculate scan duration in milliseconds."""
    if not start_time or not end_time: return 0.0
    return (end_time - start_time).total_seconds() * 1000

# ============================================================================
# GUARD CLAUSE PATTERNS
# ============================================================================

def validate_scan_parameters(target_host: str, port_range: str, scan_type: str) -> Tuple[bool, Optional[str]]:
    """Validate scan parameters with early returns."""
    # Guard clauses - all error conditions first
    if not target_host: 
        logger.warning("Missing target host", 
                      module="validation", 
                      function="validate_scan_parameters",
                      port_range=port_range,
                      scan_type=scan_type)
        return False, "Target host is required"
    
    if not is_valid_target_address(target_host):
        logger.warning("Invalid target address", 
                      module="validation", 
                      function="validate_scan_parameters",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type)
        return False, f"Invalid target address format: {target_host}"
    
    if not port_range: 
        logger.warning("Missing port range", 
                      module="validation", 
                      function="validate_scan_parameters",
                      target_host=target_host,
                      scan_type=scan_type)
        return False, "Port range is required"
    
    ports = parse_port_range(port_range)
    if not ports: 
        logger.warning("Invalid port range format", 
                      module="validation", 
                      function="validate_scan_parameters",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type)
        return False, "Invalid port range format"
    
    if not all(is_valid_port_number(port) for port in ports):
        invalid_ports = [port for port in ports if not is_valid_port_number(port)]
        logger.warning("Invalid port numbers in range", 
                      module="validation", 
                      function="validate_scan_parameters",
                      target_host=target_host,
                      port_range=port_range,
                      invalid_ports=invalid_ports)
        return False, "Invalid port numbers in range"
    
    valid_scan_types = {"tcp", "udp", "syn", "connect"}
    if scan_type not in valid_scan_types: 
        logger.warning("Invalid scan type", 
                      module="validation", 
                      function="validate_scan_parameters",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type,
                      valid_scan_types=list(valid_scan_types))
        return False, f"Invalid scan type: {scan_type}"
    
    # Happy path - all validations passed
    logger.info("Scan parameters validated successfully", 
               module="validation", 
               function="validate_scan_parameters",
               target_host=target_host,
               port_range=port_range,
               scan_type=scan_type,
               ports_count=len(ports))
    return True, None

def validate_url_parameters(target_url: str, scan_types: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate URL parameters for vulnerability scanning with early returns."""
    # Guard clauses - all error conditions first
    if not target_url or not isinstance(target_url, str): 
        return False, "Target URL is required and must be a string"
    
    if not target_url.startswith(('http://', 'https://')):
        return False, "Target URL must start with http:// or https://"
    
    # Extract hostname from URL
    try:
        parsed = urlparse(target_url)
        if not parsed.netloc:
            return False, "Invalid URL format: missing hostname"
        
        if not is_valid_hostname(parsed.netloc):
            return False, f"Invalid hostname in URL: {parsed.netloc}"
    except Exception:
        return False, "Failed to parse target URL"
    
    if not scan_types or not isinstance(scan_types, list):
        return False, "Scan types must be a non-empty list"
    
    valid_scan_types = {"sql_injection", "xss", "lfi", "rfi", "command_injection"}
    invalid_types = [st for st in scan_types if st not in valid_scan_types]
    if invalid_types:
        return False, f"Invalid scan types: {invalid_types}"
    
    # Happy path - all validations passed
    return True, None

def validate_credentials(username: str, password: str, domain: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate credentials with early returns."""
    # Guard clauses - all error conditions first
    if not username or not username.strip(): 
        return False, "Username is required"
    if len(username) < 3: 
        return False, "Username too short"
    if len(username) > 50: 
        return False, "Username too long"
    
    if not password: 
        return False, "Password is required"
    if len(password) < 8: 
        return False, "Password too short"
    if len(password) > 128: 
        return False, "Password too long"
    
    if domain and not is_valid_hostname(domain): 
        return False, "Invalid domain format"
    
    # Happy path - all validations passed
    return True, None

# ============================================================================
# FUNCTIONAL SCANNING OPERATIONS
# ============================================================================

async def scan_single_port(target_host: str, port: int, scan_type: str = "tcp", timeout: float = 5.0) -> Dict[str, Any]:
    """Scan a single port using functional approach with early returns."""
    # Guard clauses - all error conditions first
    try:
        validate_target_with_exception(target_host)
    except InvalidTargetError as e:
        logger.warning("Invalid target address for port scan", 
                      module="scanning", 
                      function="scan_single_port",
                      target_host=target_host,
                      port=port,
                      scan_type=scan_type)
        raise e
    
    if not is_valid_port_number(port):
        logger.warning("Invalid port number for scan", 
                      module="scanning", 
                      function="scan_single_port",
                      target_host=target_host,
                      port=port,
                      scan_type=scan_type)
        raise PortScanError(target_host, port, f"Invalid port number: {port}")
    
    if scan_type not in {"tcp", "udp", "syn", "connect"}:
        logger.warning("Invalid scan type", 
                      module="scanning", 
                      function="scan_single_port",
                      target_host=target_host,
                      port=port,
                      scan_type=scan_type)
        raise PortScanError(target_host, port, f"Invalid scan type: {scan_type}")
    
    if timeout <= 0:
        raise ConfigurationError("timeout", "Timeout must be positive")
    
    # Happy path - main scanning logic
    start_time = datetime.now()
    try:
        # Simulate network timeout
        await asyncio.wait_for(asyncio.sleep(0.1), timeout=timeout)
        end_time = datetime.now()
        
        # Random result for demonstration
        is_port_open = secrets.choice([True, False])
        
        logger.info("Port scan completed", 
                   module="scanning", 
                   function="scan_single_port",
                   target_host=target_host,
                   port=port,
                   scan_type=scan_type,
                   is_open=is_port_open,
                   duration_ms=calculate_scan_duration(start_time, end_time))
        
        return {
            "target_host": target_host,
            "port": port,
            "is_open": is_port_open,
            "service_name": "http" if port == 80 else "https" if port == 443 else None,
            "scan_duration_ms": calculate_scan_duration(start_time, end_time),
            "scan_timestamp": start_time.isoformat()
        }
    except asyncio.TimeoutError:
        logger.error("Port scan timeout", 
                    module="scanning", 
                    function="scan_single_port",
                    target_host=target_host,
                    port=port,
                    scan_type=scan_type,
                    timeout_seconds=timeout)
        raise NetworkTimeoutError("port_scan", timeout, target_host)
    except Exception as e:
        logger.error("Port scan failed", 
                    module="scanning", 
                    function="scan_single_port",
                    target_host=target_host,
                    port=port,
                    scan_type=scan_type,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        raise PortScanError(target_host, port, str(e))

async def scan_port_range(target_host: str, port_range: str, scan_type: str = "tcp", max_workers: int = 10) -> List[Dict[str, Any]]:
    """Scan a range of ports using functional approach with early returns."""
    logger.info("Starting port range scan", 
               module="scanning", 
               function="scan_port_range",
               target_host=target_host,
               port_range=port_range,
               scan_type=scan_type,
               max_workers=max_workers)
    
    # Guard clauses - all error conditions first
    is_valid, error_message = validate_scan_parameters(target_host, port_range, scan_type)
    if not is_valid: 
        logger.warning("Port range scan validation failed", 
                      module="scanning", 
                      function="scan_port_range",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type,
                      error_message=error_message)
        return [{"error": error_message}]
    
    if max_workers < 1 or max_workers > 100:
        logger.warning("Invalid max_workers value", 
                      module="scanning", 
                      function="scan_port_range",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type,
                      max_workers=max_workers)
        return [{"error": "Invalid max_workers value"}]
    
    # Parse port range
    ports = parse_port_range(port_range)
    if not ports: 
        logger.warning("No valid ports to scan", 
                      module="scanning", 
                      function="scan_port_range",
                      target_host=target_host,
                      port_range=port_range,
                      scan_type=scan_type)
        return [{"error": "No valid ports to scan"}]
    
    # Happy path - main scanning logic
    scan_tasks = [
        scan_single_port(target_host, port, scan_type) 
        for port in ports
    ]
    
    # Execute scans with concurrency limit
    semaphore = asyncio.Semaphore(max_workers)
    
    async def limited_scan(task) -> Any:
        async with semaphore:
            return await task
    
    try:
        # Run all scans concurrently
        results = await asyncio.gather(*[limited_scan(task) for task in scan_tasks], return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [result for result in results if not isinstance(result, Exception)]
        error_count = len(results) - len(valid_results)
        
        logger.info("Port range scan completed", 
                   module="scanning", 
                   function="scan_port_range",
                   target_host=target_host,
                   port_range=port_range,
                   scan_type=scan_type,
                   total_ports=len(ports),
                   successful_scans=len(valid_results),
                   failed_scans=error_count)
        
        return valid_results
    except Exception as e:
        logger.error("Port range scan failed", 
                    module="scanning", 
                    function="scan_port_range",
                    target_host=target_host,
                    port_range=port_range,
                    scan_type=scan_type,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        return [{"error": f"Scan failed: {str(e)}"}]

# ============================================================================
# VULNERABILITY SCANNING FUNCTIONS
# ============================================================================

def detect_sql_injection_vulnerability(target_url: str, payloads: List[str]) -> Dict[str, Any]:
    """Detect SQL injection vulnerabilities using functional approach."""
    logger.info("Starting SQL injection vulnerability scan", 
               module="vulnerability", 
               function="detect_sql_injection_vulnerability",
               target_url=target_url,
               payloads_count=len(payloads) if payloads else 0)
    
    # Guard clauses with specific exceptions
    if not target_url or not isinstance(target_url, str): 
        logger.warning("Invalid target URL for SQL injection scan", 
                      module="vulnerability", 
                      function="detect_sql_injection_vulnerability",
                      target_url=target_url,
                      target_url_type=type(target_url).__name__)
        raise VulnerabilityScanError(target_url, "sql_injection", "Target URL must be a non-empty string")
    
    # Validate URL format
    if not target_url.startswith(('http://', 'https://')):
        logger.warning("Invalid URL protocol for SQL injection scan", 
                      module="vulnerability", 
                      function="detect_sql_injection_vulnerability",
                      target_url=target_url)
        raise VulnerabilityScanError(target_url, "sql_injection", "Target URL must start with http:// or https://")
    
    # Extract and validate hostname
    try:
        parsed = urlparse(target_url)
        if not parsed.netloc:
            logger.warning("Missing hostname in URL", 
                          module="vulnerability", 
                          function="detect_sql_injection_vulnerability",
                          target_url=target_url)
            raise VulnerabilityScanError(target_url, "sql_injection", "Invalid URL format: missing hostname")
        
        if not is_valid_hostname(parsed.netloc):
            logger.warning("Invalid hostname in URL", 
                          module="vulnerability", 
                          function="detect_sql_injection_vulnerability",
                          target_url=target_url,
                          hostname=parsed.netloc)
            raise VulnerabilityScanError(target_url, "sql_injection", f"Invalid hostname in URL: {parsed.netloc}")
    except Exception as e:
        logger.error("Failed to parse target URL", 
                    module="vulnerability", 
                    function="detect_sql_injection_vulnerability",
                    target_url=target_url,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        raise VulnerabilityScanError(target_url, "sql_injection", f"Failed to parse target URL: {str(e)}")
    
    if not payloads or not isinstance(payloads, list): 
        logger.warning("Invalid payloads for SQL injection scan", 
                      module="vulnerability", 
                      function="detect_sql_injection_vulnerability",
                      target_url=target_url,
                      payloads_type=type(payloads).__name__)
        raise VulnerabilityScanError(target_url, "sql_injection", "Payloads must be a non-empty list")
    
    if not any(payload.strip() for payload in payloads): 
        logger.warning("Empty payloads for SQL injection scan", 
                      module="vulnerability", 
                      function="detect_sql_injection_vulnerability",
                      target_url=target_url)
        raise VulnerabilityScanError(target_url, "sql_injection", "Payloads list cannot be empty")
    
    # Simulate vulnerability detection
    vulnerabilities = []
    try:
        for payload in payloads:
            if payload and len(payload) > 0:
                # Simulate detection logic
                is_vulnerable = secrets.choice([True, False])
                if is_vulnerable:
                    vulnerabilities.append({
                        "type": "sql_injection",
                        "payload": payload,
                        "severity": "high",
                        "description": f"Potential SQL injection with payload: {payload[:50]}...",
                        "cve_id": "CVE-2023-1234",
                        "cvss_score": 8.5
                    })
        
        logger.info("SQL injection vulnerability scan completed", 
                   module="vulnerability", 
                   function="detect_sql_injection_vulnerability",
                   target_url=target_url,
                   vulnerabilities_found=len(vulnerabilities),
                   total_payloads_tested=len(payloads))
        
        return {
            "target_url": target_url,
            "scan_type": "sql_injection",
            "vulnerabilities": vulnerabilities,
            "total_vulnerabilities": len(vulnerabilities),
            "scan_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("SQL injection vulnerability scan failed", 
                    module="vulnerability", 
                    function="detect_sql_injection_vulnerability",
                    target_url=target_url,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        raise VulnerabilityScanError(target_url, "sql_injection", f"Scan execution failed: {str(e)}")

def detect_xss_vulnerability(target_url: str, payloads: List[str]) -> Dict[str, Any]:
    """Detect XSS vulnerabilities using functional approach with early returns."""
    # Guard clauses - all error conditions first
    if not target_url or not isinstance(target_url, str): 
        return {"error": "Invalid target URL", "vulnerabilities": []}
    
    # Validate URL format
    if not target_url.startswith(('http://', 'https://')):
        return {"error": "Target URL must start with http:// or https://", "vulnerabilities": []}
    
    # Extract and validate hostname
    try:
        parsed = urlparse(target_url)
        if not parsed.netloc:
            return {"error": "Invalid URL format: missing hostname", "vulnerabilities": []}
        
        if not is_valid_hostname(parsed.netloc):
            return {"error": f"Invalid hostname in URL: {parsed.netloc}", "vulnerabilities": []}
    except Exception as e:
        return {"error": f"Failed to parse target URL: {str(e)}", "vulnerabilities": []}
    
    if not payloads or not isinstance(payloads, list): 
        return {"error": "Invalid payloads", "vulnerabilities": []}
    
    # Happy path - main vulnerability detection logic
    vulnerabilities = []
    for payload in payloads:
        if payload and ("<script>" in payload.lower() or "javascript:" in payload.lower()):
            vulnerabilities.append({
                "type": "xss",
                "payload": payload,
                "severity": "medium",
                "description": f"Potential XSS with payload: {payload[:50]}...",
                "cve_id": "CVE-2023-5678",
                "cvss_score": 6.5
            })
    
    return {
        "target_url": target_url,
        "scan_type": "xss",
        "vulnerabilities": vulnerabilities,
        "total_vulnerabilities": len(vulnerabilities),
        "scan_timestamp": datetime.now().isoformat()
    }

# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def generate_console_report(scan_results: List[Dict[str, Any]]) -> str:
    """Generate console-formatted scan report with early returns."""
    # Guard clauses - all error conditions first
    if not scan_results or not isinstance(scan_results, list): 
        return "Error: No scan results to report"
    
    if not any(isinstance(result, dict) for result in scan_results): 
        return "Error: Invalid scan results format"
    
    # Filter valid results
    valid_results = [result for result in scan_results if isinstance(result, dict) and "error" not in result]
    
    if not valid_results: 
        return "No valid scan results found"
    
    # Happy path - main report generation logic
    report_lines = [
        "=" * 60,
        "SECURITY SCAN REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Results: {len(valid_results)}",
        ""
    ]
    
    open_ports = [result for result in valid_results if result.get("is_open", False)]
    closed_ports = [result for result in valid_results if not result.get("is_open", False)]
    
    report_lines.extend([
        f"Open Ports: {len(open_ports)}",
        f"Closed Ports: {len(closed_ports)}",
        ""
    ])
    
    if open_ports:
        report_lines.extend([
            "OPEN PORTS:",
            "-" * 20
        ])
        for result in open_ports:
            port = result.get("port", "unknown")
            service = result.get("service_name", "unknown")
            report_lines.append(f"Port {port}: {service}")
        report_lines.append("")
    
    return "\n".join(report_lines)

def generate_json_report(scan_results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
    """Generate JSON-formatted scan report with early returns."""
    # Guard clauses - all error conditions first
    if not scan_results or not isinstance(scan_results, list): 
        return json.dumps({"error": "No scan results to report"})
    
    # Happy path - main report generation logic
    valid_results = [result for result in scan_results if isinstance(result, dict) and "error" not in result]
    
    report = {
        "scan_results": valid_results,
        "summary": {
            "total_scanned": len(scan_results),
            "valid_results": len(valid_results),
            "open_ports": len([r for r in valid_results if r.get("is_open", False)]),
            "closed_ports": len([r for r in valid_results if not r.get("is_open", False)])
        }
    }
    
    if include_metadata:
        report["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "report_format": "json",
            "version": "1.0"
        }
    
    return json.dumps(report, indent=2, default=str)

# ============================================================================
# CRYPTOGRAPHY HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password using functional approach with early returns."""
    # Guard clauses - all error conditions first
    if not password or not isinstance(password, str): 
        raise ValueError("Password must be a non-empty string")
    
    if len(password) < 8: 
        raise ValueError("Password must be at least 8 characters")
    
    # Happy path - main password hashing logic
    # Generate salt if not provided
    if not salt:
        salt = secrets.token_hex(16)
    
    # Create hash
    password_with_salt = password + salt
    hashed = hashlib.sha256(password_with_salt.encode()).hexdigest()
    
    return hashed, salt

def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify password hash using functional approach with early returns."""
    # Guard clauses - all error conditions first
    if not password or not isinstance(password, str): 
        return False
    if not hashed_password or not isinstance(hashed_password, str): 
        return False
    if not salt or not isinstance(salt, str): 
        return False
    
    # Happy path - main password verification logic
    try:
        password_with_salt = password + salt
        computed_hash = hashlib.sha256(password_with_salt.encode()).hexdigest()
        return secrets.compare_digest(computed_hash, hashed_password)
    except Exception:
        return False

def encrypt_sensitive_data(data: str, key: str) -> Tuple[str, str]:
    """Encrypt sensitive data using functional approach with early returns."""
    # Guard clauses - all error conditions first
    if not data or not isinstance(data, str): 
        raise ValueError("Data must be a non-empty string")
    
    if not key or not isinstance(key, str): 
        raise ValueError("Key must be a non-empty string")
    
    if len(key) < 16: 
        raise ValueError("Key must be at least 16 characters")
    
    # Happy path - main encryption logic
    # Simple XOR encryption for demonstration (use proper encryption in production)
    encrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))
    iv = secrets.token_hex(8)
    
    return encrypted, iv

def decrypt_sensitive_data(encrypted_data: str, key: str, iv: str) -> str:
    """Decrypt sensitive data using functional approach with early returns."""
    # Guard clauses - all error conditions first
    if not encrypted_data or not isinstance(encrypted_data, str): 
        raise ValueError("Encrypted data must be a non-empty string")
    
    if not key or not isinstance(key, str): 
        raise ValueError("Key must be a non-empty string")
    
    if not iv or not isinstance(iv, str): 
        raise ValueError("IV must be a non-empty string")
    
    # Happy path - main decryption logic
    # Simple XOR decryption for demonstration
    try:
        decrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(encrypted_data, key * (len(encrypted_data) // len(key) + 1)))
        return decrypted
    except Exception:
        raise ValueError("Decryption failed")

# ============================================================================
# NETWORK HELPER FUNCTIONS
# ============================================================================

def is_port_open_sync(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if port is open using synchronous approach with early returns."""
    # Guard clauses - all error conditions first
    if not is_valid_target_address(host): 
        return False
    if not is_valid_port_number(port): 
        return False
    if timeout <= 0: 
        return False
    
    # Happy path - main sync port checking logic
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

async def is_port_open_async(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if port is open using asynchronous approach with early returns."""
    # Guard clauses - all error conditions first
    if not is_valid_target_address(host): 
        return False
    if not is_valid_port_number(port): 
        return False
    if timeout <= 0: 
        return False
    
    # Happy path - main async port checking logic
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), 
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (asyncio.TimeoutError, OSError):
        return False

def resolve_dns_hostname(hostname: str) -> Optional[str]:
    """Resolve DNS hostname to IP address with early returns."""
    # Guard clauses - all error conditions first
    if not hostname or not isinstance(hostname, str): 
        return None
    if not is_valid_hostname(hostname): 
        return None
    
    # Happy path - main DNS resolution logic
    try:
        ip_address = socket.gethostbyname(hostname)
        return ip_address if is_valid_ip_address(ip_address) else None
    except socket.gaierror:
        return None

# ============================================================================
# RORO PATTERN IMPLEMENTATION
# ============================================================================

def scan_target_roro(params: Dict[str, Any]) -> Dict[str, Any]:
    """Scan target using RORO (Receive Object, Return Object) pattern with early returns."""
    # Guard clauses - all error conditions first
    if not params or not isinstance(params, dict): 
        return {"error": "Invalid parameters object"}
    
    target_host = params.get("target_host")
    port_range = params.get("port_range", "1-1024")
    scan_type = params.get("scan_type", "tcp")
    max_workers = params.get("max_workers", 10)
    
    # Validate parameters
    is_valid, error_message = validate_scan_parameters(target_host, port_range, scan_type)
    if not is_valid: 
        return {"error": error_message}
    
    # Happy path - return scan configuration
    return {
        "target_host": target_host,
        "port_range": port_range,
        "scan_type": scan_type,
        "max_workers": max_workers,
        "scan_id": generate_scan_id(),
        "timestamp": datetime.now().isoformat()
    }

def validate_credentials_roro(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate credentials using RORO pattern with early returns."""
    # Guard clauses - all error conditions first
    if not params or not isinstance(params, dict): 
        return {"error": "Invalid parameters object"}
    
    username = params.get("username")
    password = params.get("password")
    domain = params.get("domain")
    
    # Happy path - validate credentials and return result
    is_valid, error_message = validate_credentials(username, password, domain)
    
    return {
        "is_valid": is_valid,
        "error_message": error_message,
        "username": username,
        "has_domain": bool(domain),
        "validation_timestamp": datetime.now().isoformat()
    }

# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Security Scanner API", version="1.0.0")
    
    @app.post("/scan/ports")
    async def scan_ports_endpoint(scan_config: ScanConfig) -> Dict[str, Any]:
        """Scan ports endpoint using functional approach."""
        logger.info("Port scan endpoint called", 
                   module="api", 
                   function="scan_ports_endpoint",
                   target_host=scan_config.target_host,
                   port_range=scan_config.port_range,
                   scan_type=scan_config.scan_type,
                   max_workers=scan_config.max_workers)
        
        # Guard clauses handled by Pydantic validation
        try:
            results = await scan_port_range(
                target_host=scan_config.target_host,
                port_range=scan_config.port_range,
                scan_type=scan_config.scan_type,
                max_workers=scan_config.max_workers
            )
            
            open_ports = len([r for r in results if r.get("is_open", False)])
            
            logger.info("Port scan endpoint completed successfully", 
                       module="api", 
                       function="scan_ports_endpoint",
                       target_host=scan_config.target_host,
                       total_results=len(results),
                       open_ports=open_ports)
            
            return {
                "success": True,
                "scan_id": generate_scan_id(),
                "results": results,
                "total_ports": len(results),
                "open_ports": open_ports
            }
        except Exception as e:
            logger.error("Port scan endpoint failed", 
                        module="api", 
                        function="scan_ports_endpoint",
                        target_host=scan_config.target_host,
                        port_range=scan_config.port_range,
                        scan_type=scan_config.scan_type,
                        exception_type=type(e).__name__,
                        exception_message=str(e))
            raise HTTPException(status_code=500, detail="Scan failed")
    
    @app.post("/scan/vulnerabilities")
    async def scan_vulnerabilities_endpoint(target_url: str, scan_types: List[str]) -> Dict[str, Any]:
        """Scan for vulnerabilities endpoint."""
        logger.info("Vulnerability scan endpoint called", 
                   module="api", 
                   function="scan_vulnerabilities_endpoint",
                   target_url=target_url,
                   scan_types=scan_types)
        
        try:
            # Guard clauses with comprehensive validation
            is_valid, error_message = validate_url_parameters(target_url, scan_types)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_message)
            
            results = {}
            
            if "sql_injection" in scan_types:
                payloads = ["' OR 1=1--", "'; DROP TABLE users--", "1' UNION SELECT * FROM users--"]
                results["sql_injection"] = detect_sql_injection_vulnerability(target_url, payloads)
            
            if "xss" in scan_types:
                payloads = ["<script>alert('XSS')</script>", "javascript:alert('XSS')"]
                results["xss"] = detect_xss_vulnerability(target_url, payloads)
            
            logger.info("Vulnerability scan endpoint completed successfully", 
                       module="api", 
                       function="scan_vulnerabilities_endpoint",
                       target_url=target_url,
                       scan_types=scan_types,
                       results_count=len(results))
            
            return {
                "success": True,
                "target_url": target_url,
                "scan_types": scan_types,
                "results": results,
                "scan_timestamp": datetime.now().isoformat()
            }
        except SecurityToolError as e:
            logger.error("Vulnerability scan endpoint failed with security error", 
                        module="api", 
                        function="scan_vulnerabilities_endpoint",
                        target_url=target_url,
                        scan_types=scan_types,
                        error_code=e.error_code,
                        error_message=e.message)
            error_response = format_api_error_response(e)
            raise HTTPException(status_code=400, detail=error_response)
        except Exception as e:
            logger.error("Vulnerability scan endpoint failed", 
                        module="api", 
                        function="scan_vulnerabilities_endpoint",
                        target_url=target_url,
                        scan_types=scan_types,
                        exception_type=type(e).__name__,
                        exception_message=str(e))
            error_response = format_api_error_response(e)
            raise HTTPException(status_code=500, detail=error_response)

# ============================================================================
# ERROR HANDLING AND MESSAGE MAPPING
# ============================================================================

def map_exception_to_user_message(exception: Exception) -> Dict[str, Any]:
    """Map exceptions to user-friendly CLI/API messages."""
    if isinstance(exception, SecurityToolError):
        return {
            "error": True,
            "message": exception.message,
            "error_code": exception.error_code,
            "details": exception.details,
            "user_friendly": True
        }
    
    # Map standard exceptions
    if isinstance(exception, asyncio.TimeoutError):
        return {
            "error": True,
            "message": "Operation timed out. Please try again or increase timeout.",
            "error_code": "TIMEOUT_ERROR",
            "details": {"original_exception": str(exception)},
            "user_friendly": True
        }
    
    if isinstance(exception, ConnectionError):
        return {
            "error": True,
            "message": "Network connection failed. Please check your internet connection.",
            "error_code": "CONNECTION_ERROR",
            "details": {"original_exception": str(exception)},
            "user_friendly": True
        }
    
    if isinstance(exception, ValueError):
        return {
            "error": True,
            "message": f"Invalid input: {str(exception)}",
            "error_code": "VALUE_ERROR",
            "details": {"original_exception": str(exception)},
            "user_friendly": True
        }
    
    if isinstance(exception, TypeError):
        return {
            "error": True,
            "message": f"Type error: {str(exception)}",
            "error_code": "TYPE_ERROR",
            "details": {"original_exception": str(exception)},
            "user_friendly": True
        }
    
    # Generic error for unknown exceptions
    return {
        "error": True,
        "message": "An unexpected error occurred. Please try again or contact support.",
        "error_code": "UNKNOWN_ERROR",
        "details": {
            "original_exception": str(exception),
            "exception_type": type(exception).__name__
        },
        "user_friendly": True
    }

def format_cli_error_message(exception: Exception, verbose: bool = False) -> str:
    """Format exception for CLI output."""
    error_info = map_exception_to_user_message(exception)
    
    if verbose:
        return f"""
❌ Error: {error_info['message']}
   Code: {error_info['error_code']}
   Details: {json.dumps(error_info['details'], indent=2)}
"""
    else:
        return f"❌ {error_info['message']}"

async def format_api_error_response(exception: Exception) -> Dict[str, Any]:
    """Format exception for API response."""
    error_info = map_exception_to_user_message(exception)
    
    return {
        "success": False,
        "error": error_info['message'],
        "error_code": error_info['error_code'],
        "details": error_info['details'],
        "timestamp": datetime.now().isoformat()
    }

def handle_security_operation(operation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Generic handler for security operations with error mapping."""
    try:
        result = operation_func(*args, **kwargs)
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Security operation failed", 
                    module="error_handling", 
                    function="handle_security_operation",
                    operation=operation_func.__name__,
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        return format_api_error_response(e)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function demonstrating functional patterns."""
    logger.info("Starting functional optimization examples", 
               module="main", 
               function="main")
    
    try:
        # Example 1: Port scanning with valid targets
        logger.info("Running valid port scan example", 
                   module="main", 
                   function="main",
                   example="valid_port_scan")
        scan_results = await scan_port_range("localhost", "80,443,8080", "tcp", 5)
        console_report = generate_console_report(scan_results)
        print(console_report)
        
        # Example 2: Port scanning with invalid targets (error handling)
        logger.info("Running invalid port scan example", 
                   module="main", 
                   function="main",
                   example="invalid_port_scan")
        try:
            invalid_results = await scan_port_range("invalid-host", "80,443", "tcp", 5)
            print("Invalid target results:", json.dumps(invalid_results, indent=2))
        except SecurityToolError as e:
            print("Invalid target error:", format_cli_error_message(e))
        
        # Example 3: Vulnerability scanning with valid URL
        logger.info("Running valid vulnerability scan example", 
                   module="main", 
                   function="main",
                   example="valid_vulnerability_scan")
        vuln_results = detect_sql_injection_vulnerability(
            "http://example.com/login", 
            ["' OR 1=1--", "admin'--"]
        )
        print("Valid URL results:", json.dumps(vuln_results, indent=2))
        
        # Example 4: Vulnerability scanning with invalid URL (error handling)
        logger.info("Running invalid vulnerability scan example", 
                   module="main", 
                   function="main",
                   example="invalid_vulnerability_scan")
        try:
            invalid_vuln_results = detect_sql_injection_vulnerability(
                "invalid-url", 
                ["' OR 1=1--"]
            )
            print("Invalid URL results:", json.dumps(invalid_vuln_results, indent=2))
        except SecurityToolError as e:
            print("Invalid URL error:", format_cli_error_message(e))
        
        # Example 5: RORO pattern with valid parameters
        logger.info("Running valid RORO pattern example", 
                   module="main", 
                   function="main",
                   example="valid_roro_pattern")
        scan_params = {
            "target_host": "192.168.1.1",
            "port_range": "22,80,443",
            "scan_type": "tcp",
            "max_workers": 5
        }
        roro_result = scan_target_roro(scan_params)
        print("Valid RORO results:", json.dumps(roro_result, indent=2))
        
        # Example 6: RORO pattern with invalid parameters (error handling)
        logger.info("Running invalid RORO pattern example", 
                   module="main", 
                   function="main",
                   example="invalid_roro_pattern")
        invalid_roro_params = {
            "target_host": "invalid-address",
            "port_range": "99999",
            "scan_type": "invalid_type"
        }
        try:
            invalid_roro_result = scan_target_roro(invalid_roro_params)
            print("Invalid RORO results:", json.dumps(invalid_roro_result, indent=2))
        except SecurityToolError as e:
            print("Invalid RORO error:", format_cli_error_message(e))
        
        # Example 7: Network timeout demonstration
        logger.info("Running network timeout example", 
                   module="main", 
                   function="main",
                   example="network_timeout")
        try:
            await scan_single_port("192.168.1.1", 80, "tcp", timeout=0.001)  # Very short timeout
        except NetworkTimeoutError as e:
            print("Network timeout error:", format_cli_error_message(e))
        
        logger.info("All functional optimization examples completed successfully", 
                   module="main", 
                   function="main")
        
    except Exception as e:
        logger.error("Main execution failed", 
                    module="main", 
                    function="main",
                    exception_type=type(e).__name__,
                    exception_message=str(e))
        print("Main execution error:", format_cli_error_message(e, verbose=True))
        raise

match __name__:
    case "__main__":
    asyncio.run(main()) 