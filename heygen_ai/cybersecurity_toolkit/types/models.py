from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator
import ipaddress
import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cybersecurity Toolkit - Pydantic Models
=======================================

Pydantic v2 models for request/response validation with comprehensive
error handling and guard clauses.
"""


class SeverityLevel(str, Enum):
    """Vulnerability severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ScanType(str, Enum):
    """Types of scanning operations."""
    PORT_SCAN = "port_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_SCAN = "web_scan"
    DNS_ENUMERATION = "dns_enumeration"
    SMB_ENUMERATION = "smb_enumeration"
    SSH_ENUMERATION = "ssh_enumeration"

class AttackType(str, Enum):
    """Types of attack simulations."""
    BRUTE_FORCE = "brute_force"
    EXPLOIT = "exploit"
    DOS = "dos"
    PHISHING = "phishing"

# --- Request Models ---

class BaseRequest(BaseModel):
    """Base request model with common validation."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    
    @field_validator('request_id')
    @classmethod
    async def validate_request_id(cls, v) -> bool:
        if v is not None and not re.match(r'^[a-zA-Z0-9_-]{1,50}$', v):
            raise ValueError('Request ID must be alphanumeric with underscores and hyphens only')
        return v

class ScanRequest(BaseRequest):
    """Request model for scanning operations with guard clause validation."""
    target_host: str = Field(..., description="Target hostname or IP address")
    target_ports: List[int] = Field(default=[80, 443, 22, 21], description="Ports to scan")
    scan_timeout: float = Field(default=5.0, gt=0, le=300, description="Scan timeout in seconds")
    max_concurrent_scans: int = Field(default=10, gt=0, le=1000, description="Maximum concurrent scans")
    enable_ssl_verification: bool = Field(default=True, description="Enable SSL certificate verification")
    scan_type: ScanType = Field(default=ScanType.PORT_SCAN, description="Type of scan to perform")
    
    @field_validator('target_host')
    @classmethod
    def validate_target_host(cls, v) -> Optional[Dict[str, Any]]:
        # Guard clause: Check if host is provided
        if not v or not v.strip():
            raise ValueError('Target host is required')
        
        # Guard clause: Check host length
        if len(v) > 253:
            raise ValueError('Target host name too long (max 253 characters)')
        
        # Guard clause: Validate IP address format
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            # Not an IP address, validate as hostname
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$', v):
                raise ValueError('Invalid hostname format')
            return v.lower()
    
    @field_validator('target_ports')
    @classmethod
    def validate_target_ports(cls, v) -> Optional[Dict[str, Any]]:
        # Guard clause: Check if ports list is provided
        if not v:
            raise ValueError('At least one port must be specified')
        
        # Guard clause: Check port range and uniqueness
        valid_ports = []
        for port in v:
            if not isinstance(port, int):
                raise ValueError(f'Port must be an integer, got {type(port)}')
            if port < 1 or port > 65535:
                raise ValueError(f'Port {port} is out of valid range (1-65535)')
            if port in valid_ports:
                raise ValueError(f'Duplicate port {port} found')
            valid_ports.append(port)
        
        return valid_ports
    
    @field_validator('scan_timeout')
    @classmethod
    def validate_scan_timeout(cls, v) -> bool:
        # Guard clause: Check timeout range
        if v <= 0:
            raise ValueError('Scan timeout must be positive')
        if v > 300:
            raise ValueError('Scan timeout cannot exceed 300 seconds')
        return v
    
    @field_validator('max_concurrent_scans')
    @classmethod
    def validate_max_concurrent_scans(cls, v) -> bool:
        # Guard clause: Check concurrent scans limit
        if v <= 0:
            raise ValueError('Maximum concurrent scans must be positive')
        if v > 1000:
            raise ValueError('Maximum concurrent scans cannot exceed 1000')
        return v

class VulnerabilityRequest(BaseRequest):
    """Request model for vulnerability scanning with comprehensive validation."""
    target_url: str = Field(..., description="Target URL to scan")
    scan_depth: str = Field(default="medium", description="Scan depth level")
    include_ssl_checks: bool = Field(default=True, description="Include SSL/TLS checks")
    include_header_checks: bool = Field(default=True, description="Include security header checks")
    include_content_checks: bool = Field(default=True, description="Include content vulnerability checks")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    max_redirects: int = Field(default=5, gt=0, le=20, description="Maximum redirects to follow")
    
    @field_validator('target_url')
    @classmethod
    def validate_target_url(cls, v) -> Optional[Dict[str, Any]]:
        # Guard clause: Check if URL is provided
        if not v or not v.strip():
            raise ValueError('Target URL is required')
        
        # Guard clause: Check URL format
        url_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(:[0-9]{1,5})?(/.*)?$'
        if not re.match(url_pattern, v):
            raise ValueError('Invalid URL format')
        
        # Guard clause: Check URL length
        if len(v) > 2048:
            raise ValueError('URL too long (max 2048 characters)')
        
        return v
    
    @field_validator('scan_depth')
    @classmethod
    def validate_scan_depth(cls, v) -> bool:
        # Guard clause: Check scan depth value
        valid_depths = ["low", "medium", "high"]
        if v not in valid_depths:
            raise ValueError(f'Scan depth must be one of: {valid_depths}')
        return v
    
    @field_validator('custom_headers')
    @classmethod
    def validate_custom_headers(cls, v) -> bool:
        # Guard clause: Check header format
        for header_name, header_value in v.items():
            if not header_name or not header_name.strip():
                raise ValueError('Header name cannot be empty')
            if len(header_name) > 100:
                raise ValueError('Header name too long (max 100 characters)')
            if len(header_value) > 1000:
                raise ValueError('Header value too long (max 1000 characters)')
        return v
    
    @field_validator('max_redirects')
    @classmethod
    def validate_max_redirects(cls, v) -> bool:
        # Guard clause: Check redirect limit
        if v <= 0:
            raise ValueError('Maximum redirects must be positive')
        if v > 20:
            raise ValueError('Maximum redirects cannot exceed 20')
        return v

class EnumerationRequest(BaseRequest):
    """Request model for enumeration operations with validation."""
    target_domain: str = Field(..., description="Target domain for enumeration")
    enumeration_type: str = Field(..., description="Type of enumeration")
    subdomain_list: Optional[List[str]] = Field(default=None, description="List of subdomains to check")
    username_list: Optional[List[str]] = Field(default=None, description="List of usernames to try")
    password_list: Optional[List[str]] = Field(default=None, description="List of passwords to try")
    wordlist_file: Optional[str] = Field(default=None, description="Path to wordlist file")
    recursive_enumeration: bool = Field(default=False, description="Enable recursive enumeration")
    max_recursion_depth: int = Field(default=3, gt=0, le=10, description="Maximum recursion depth")
    
    @field_validator('target_domain')
    @classmethod
    def validate_target_domain(cls, v) -> Optional[Dict[str, Any]]:
        # Guard clause: Check if domain is provided
        if not v or not v.strip():
            raise ValueError('Target domain is required')
        
        # Guard clause: Check domain format
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
        if not re.match(domain_pattern, v):
            raise ValueError('Invalid domain format')
        
        # Guard clause: Check domain length
        if len(v) > 253:
            raise ValueError('Domain name too long (max 253 characters)')
        
        return v.lower()
    
    @field_validator('enumeration_type')
    @classmethod
    def validate_enumeration_type(cls, v) -> bool:
        # Guard clause: Check enumeration type
        valid_types = ["dns", "smb", "ssh", "ftp", "http"]
        if v not in valid_types:
            raise ValueError(f'Enumeration type must be one of: {valid_types}')
        return v
    
    @field_validator('subdomain_list')
    @classmethod
    def validate_subdomain_list(cls, v) -> List[Any]:
        # Guard clause: Check subdomain list if provided
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Subdomain list must be a list')
            if len(v) > 1000:
                raise ValueError('Subdomain list too large (max 1000 items)')
            for subdomain in v:
                if not subdomain or not subdomain.strip():
                    raise ValueError('Subdomain cannot be empty')
                if len(subdomain) > 63:
                    raise ValueError('Subdomain too long (max 63 characters)')
        return v
    
    @field_validator('username_list')
    @classmethod
    def validate_username_list(cls, v) -> List[Any]:
        # Guard clause: Check username list if provided
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Username list must be a list')
            if len(v) > 10000:
                raise ValueError('Username list too large (max 10000 items)')
            for username in v:
                if not username or not username.strip():
                    raise ValueError('Username cannot be empty')
                if len(username) > 100:
                    raise ValueError('Username too long (max 100 characters)')
        return v
    
    @field_validator('password_list')
    @classmethod
    def validate_password_list(cls, v) -> List[Any]:
        # Guard clause: Check password list if provided
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Password list must be a list')
            if len(v) > 10000:
                raise ValueError('Password list too large (max 10000 items)')
            for password in v:
                if not password or not password.strip():
                    raise ValueError('Password cannot be empty')
                if len(password) > 100:
                    raise ValueError('Password too long (max 100 characters)')
        return v

class AttackRequest(BaseRequest):
    """Request model for attack simulations with validation."""
    target_host: str = Field(..., description="Target host for attack")
    target_port: int = Field(..., gt=0, le=65535, description="Target port for attack")
    attack_type: AttackType = Field(..., description="Type of attack to perform")
    username_list: List[str] = Field(default_factory=list, description="List of usernames for brute force")
    password_list: List[str] = Field(default_factory=list, description="List of passwords for brute force")
    max_attempts: int = Field(default=100, gt=0, le=100000, description="Maximum attack attempts")
    delay_between_attempts: float = Field(default=1.0, ge=0, le=60, description="Delay between attempts in seconds")
    timeout_per_attempt: float = Field(default=30.0, gt=0, le=300, description="Timeout per attempt in seconds")
    exploit_payload: Optional[str] = Field(default=None, description="Custom exploit payload")
    
    @field_validator('target_host')
    @classmethod
    def validate_target_host(cls, v) -> Optional[Dict[str, Any]]:
        # Guard clause: Check if host is provided
        if not v or not v.strip():
            raise ValueError('Target host is required')
        
        # Guard clause: Validate IP address or hostname
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            # Not an IP address, validate as hostname
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$', v):
                raise ValueError('Invalid hostname format')
            return v.lower()
    
    @field_validator('username_list')
    @classmethod
    def validate_username_list(cls, v) -> List[Any]:
        # Guard clause: Check username list for brute force attacks
        if not v:
            raise ValueError('Username list is required for brute force attacks')
        if len(v) > 10000:
            raise ValueError('Username list too large (max 10000 items)')
        for username in v:
            if not username or not username.strip():
                raise ValueError('Username cannot be empty')
            if len(username) > 100:
                raise ValueError('Username too long (max 100 characters)')
        return v
    
    @field_validator('password_list')
    @classmethod
    def validate_password_list(cls, v) -> List[Any]:
        # Guard clause: Check password list for brute force attacks
        if not v:
            raise ValueError('Password list is required for brute force attacks')
        if len(v) > 10000:
            raise ValueError('Password list too large (max 10000 items)')
        for password in v:
            if not password or not password.strip():
                raise ValueError('Password cannot be empty')
            if len(password) > 100:
                raise ValueError('Password too long (max 100 characters)')
        return v
    
    @field_validator('max_attempts')
    @classmethod
    def validate_max_attempts(cls, v) -> bool:
        # Guard clause: Check attempts limit
        if v <= 0:
            raise ValueError('Maximum attempts must be positive')
        if v > 100000:
            raise ValueError('Maximum attempts cannot exceed 100000')
        return v

# --- Response Models ---

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    success: bool = Field(..., description="Operation success status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    error_message: Optional[str] = Field(default=None, description="Error message if operation failed")
    error_type: Optional[str] = Field(default=None, description="Type of error if operation failed")

class ScanResult(BaseModel):
    """Result model for scanning operations."""
    target_host: str = Field(..., description="Target host that was scanned")
    target_port: int = Field(..., description="Target port that was scanned")
    is_port_open: bool = Field(..., description="Whether the port is open")
    service_name: Optional[str] = Field(default=None, description="Service name running on port")
    response_time: Optional[float] = Field(default=None, ge=0, description="Response time in seconds")
    ssl_info: Optional[Dict[str, Any]] = Field(default=None, description="SSL certificate information")
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the scan was performed")
    error_message: Optional[str] = Field(default=None, description="Error message if scan failed")

class VulnerabilityResult(BaseModel):
    """Result model for vulnerability scanning."""
    target_url: str = Field(..., description="Target URL that was scanned")
    vulnerability_type: str = Field(..., description="Type of vulnerability found")
    severity_level: SeverityLevel = Field(..., description="Severity level of the vulnerability")
    description: str = Field(..., description="Description of the vulnerability")
    detected_pattern: Optional[str] = Field(default=None, description="Pattern that triggered the detection")
    remediation_advice: Optional[str] = Field(default=None, description="Advice for remediation")
    cve_identifier: Optional[str] = Field(default=None, description="CVE identifier if applicable")
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the scan was performed")

class EnumerationResult(BaseModel):
    """Result model for enumeration operations."""
    target_domain: str = Field(..., description="Target domain that was enumerated")
    enumeration_type: str = Field(..., description="Type of enumeration performed")
    discovered_items: List[str] = Field(default_factory=list, description="Items discovered during enumeration")
    enumeration_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When enumeration was performed")
    success_count: int = Field(default=0, ge=0, description="Number of successful enumeration attempts")
    failure_count: int = Field(default=0, ge=0, description="Number of failed enumeration attempts")
    total_attempts: int = Field(default=0, ge=0, description="Total number of attempts made")

class AttackResult(BaseModel):
    """Result model for attack simulations."""
    target_host: str = Field(..., description="Target host that was attacked")
    attack_type: AttackType = Field(..., description="Type of attack performed")
    is_successful: bool = Field(..., description="Whether the attack was successful")
    discovered_credentials: Optional[Dict[str, str]] = Field(default=None, description="Credentials discovered during attack")
    attack_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the attack was performed")
    attempts_made: int = Field(default=0, ge=0, description="Number of attempts made")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate of the attack")
    error_message: Optional[str] = Field(default=None, description="Error message if attack failed")

class ScanResponse(BaseResponse):
    """Response model for scanning operations."""
    data: List[ScanResult] = Field(default_factory=list, description="Scan results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VulnerabilityResponse(BaseResponse):
    """Response model for vulnerability scanning."""
    data: List[VulnerabilityResult] = Field(default_factory=list, description="Vulnerability results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class EnumerationResponse(BaseResponse):
    """Response model for enumeration operations."""
    data: EnumerationResult = Field(..., description="Enumeration results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AttackResponse(BaseResponse):
    """Response model for attack simulations."""
    data: AttackResult = Field(..., description="Attack results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# --- Named Exports ---

__all__ = [
    # Enums
    'SeverityLevel',
    'ScanType',
    'AttackType',
    
    # Base models
    'BaseRequest',
    'BaseResponse',
    
    # Request models
    'ScanRequest',
    'VulnerabilityRequest',
    'EnumerationRequest',
    'AttackRequest',
    
    # Result models
    'ScanResult',
    'VulnerabilityResult',
    'EnumerationResult',
    'AttackResult',
    
    # Response models
    'ScanResponse',
    'VulnerabilityResponse',
    'EnumerationResponse',
    'AttackResponse'
] 