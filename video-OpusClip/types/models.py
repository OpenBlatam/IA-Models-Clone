#!/usr/bin/env python3
"""
Models Module for Video-OpusClip
Data models and type definitions
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from typing_extensions import Annotated

# ============================================================================
# ENUMS
# ============================================================================

class ScanType(str, Enum):
    """Types of security scans"""
    PORT_SCAN = "port_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_SCAN = "web_scan"
    NETWORK_SCAN = "network_scan"
    COMPREHENSIVE_SCAN = "comprehensive_scan"

class EnumerationType(str, Enum):
    """Types of enumeration"""
    DNS_ENUMERATION = "dns_enumeration"
    SMB_ENUMERATION = "smb_enumeration"
    SSH_ENUMERATION = "ssh_enumeration"
    USER_ENUMERATION = "user_enumeration"
    SERVICE_ENUMERATION = "service_enumeration"

class AttackType(str, Enum):
    """Types of attacks"""
    BRUTE_FORCE = "brute_force"
    EXPLOITATION = "exploitation"
    SOCIAL_ENGINEERING = "social_engineering"
    PHISHING = "phishing"
    DOS = "denial_of_service"

class SeverityLevel(str, Enum):
    """Severity levels for findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class StatusType(str, Enum):
    """Status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProtocolType(str, Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SSH = "ssh"
    SMTP = "smtp"
    DNS = "dns"

class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms"""
    AES = "aes"
    RSA = "rsa"
    FERNET = "fernet"
    CHACHA20 = "chacha20"

class HashAlgorithm(str, Enum):
    """Hash algorithms"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

# ============================================================================
# BASE MODELS
# ============================================================================

class BaseVideoOpusClipModel(BaseModel):
    """Base model for Video-OpusClip with common configuration"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True
    )

class TimestampedModel(BaseVideoOpusClipModel):
    """Base model with timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return datetime.utcnow()

# ============================================================================
# SCANNING MODELS
# ============================================================================

class ScanTarget(BaseVideoOpusClipModel):
    """Target for scanning operations"""
    host: str = Field(..., description="Target hostname or IP address")
    port: Optional[int] = Field(None, description="Target port")
    protocol: ProtocolType = Field(ProtocolType.TCP, description="Target protocol")
    description: Optional[str] = Field(None, description="Target description")
    
    @validator('host')
    def validate_host(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Host cannot be empty")
        return v.strip()
    
    @validator('port')
    def validate_port(cls, v):
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

class ScanConfiguration(BaseVideoOpusClipModel):
    """Configuration for scanning operations"""
    scan_type: ScanType = Field(..., description="Type of scan to perform")
    targets: List[ScanTarget] = Field(..., min_items=1, description="Targets to scan")
    timeout: float = Field(30.0, gt=0, description="Timeout in seconds")
    max_concurrent: int = Field(10, gt=0, le=100, description="Maximum concurrent operations")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retry attempts")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    user_agent: str = Field("Video-OpusClip-Scanner/1.0", description="User agent string")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    
    @validator('targets')
    def validate_targets(cls, v):
        if not v:
            raise ValueError("At least one target must be specified")
        return v

class PortResult(BaseVideoOpusClipModel):
    """Result of port scanning"""
    port: int = Field(..., ge=1, le=65535, description="Port number")
    status: Literal["open", "closed", "filtered", "unfiltered"] = Field(..., description="Port status")
    service: Optional[str] = Field(None, description="Detected service")
    version: Optional[str] = Field(None, description="Service version")
    banner: Optional[str] = Field(None, description="Service banner")
    response_time: Optional[float] = Field(None, gt=0, description="Response time in seconds")

class VulnerabilityResult(BaseVideoOpusClipModel):
    """Result of vulnerability scanning"""
    vulnerability_type: str = Field(..., description="Type of vulnerability")
    severity: SeverityLevel = Field(..., description="Vulnerability severity")
    title: str = Field(..., description="Vulnerability title")
    description: str = Field(..., description="Vulnerability description")
    affected_url: Optional[str] = Field(None, description="Affected URL")
    affected_parameter: Optional[str] = Field(None, description="Affected parameter")
    payload: Optional[str] = Field(None, description="Exploit payload")
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    cvss_score: Optional[float] = Field(None, ge=0, le=10, description="CVSS score")
    remediation: Optional[str] = Field(None, description="Remediation steps")
    references: List[str] = Field(default_factory=list, description="Reference links")

class ScanResult(TimestampedModel):
    """Result of scanning operations"""
    scan_id: str = Field(..., description="Unique scan identifier")
    scan_type: ScanType = Field(..., description="Type of scan performed")
    target: ScanTarget = Field(..., description="Scanned target")
    status: StatusType = Field(..., description="Scan status")
    start_time: datetime = Field(..., description="Scan start time")
    end_time: Optional[datetime] = Field(None, description="Scan end time")
    duration: Optional[float] = Field(None, gt=0, description="Scan duration in seconds")
    ports_scanned: int = Field(0, ge=0, description="Number of ports scanned")
    ports_open: int = Field(0, ge=0, description="Number of open ports")
    vulnerabilities_found: int = Field(0, ge=0, description="Number of vulnerabilities found")
    port_results: List[PortResult] = Field(default_factory=list, description="Port scan results")
    vulnerability_results: List[VulnerabilityResult] = Field(default_factory=list, description="Vulnerability scan results")
    error_message: Optional[str] = Field(None, description="Error message if scan failed")
    
    @root_validator
    def validate_scan_times(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        
        if end_time and start_time and end_time < start_time:
            raise ValueError("End time cannot be before start time")
        
        return values

# ============================================================================
# ENUMERATION MODELS
# ============================================================================

class DNSRecord(BaseVideoOpusClipModel):
    """DNS record information"""
    record_type: str = Field(..., description="DNS record type")
    name: str = Field(..., description="Record name")
    value: str = Field(..., description="Record value")
    ttl: Optional[int] = Field(None, description="Time to live")
    priority: Optional[int] = Field(None, description="Priority (for MX records)")

class SubdomainResult(BaseVideoOpusClipModel):
    """Subdomain enumeration result"""
    subdomain: str = Field(..., description="Subdomain name")
    ip_addresses: List[str] = Field(default_factory=list, description="IP addresses")
    status: Literal["active", "inactive", "unknown"] = Field(..., description="Subdomain status")
    http_status: Optional[int] = Field(None, description="HTTP status code")
    title: Optional[str] = Field(None, description="Page title")
    technologies: List[str] = Field(default_factory=list, description="Detected technologies")

class SMBShare(BaseVideoOpusClipModel):
    """SMB share information"""
    share_name: str = Field(..., description="Share name")
    share_type: str = Field(..., description="Share type")
    comment: Optional[str] = Field(None, description="Share comment")
    permissions: List[str] = Field(default_factory=list, description="Share permissions")
    accessible: bool = Field(False, description="Whether share is accessible")
    files: List[str] = Field(default_factory=list, description="List of files in share")

class SSHHostKey(BaseVideoOpusClipModel):
    """SSH host key information"""
    key_type: str = Field(..., description="Key type")
    key_fingerprint: str = Field(..., description="Key fingerprint")
    key_length: Optional[int] = Field(None, description="Key length in bits")
    algorithm: str = Field(..., description="Encryption algorithm")

class EnumerationResult(TimestampedModel):
    """Result of enumeration operations"""
    enumeration_id: str = Field(..., description="Unique enumeration identifier")
    enumeration_type: EnumerationType = Field(..., description="Type of enumeration")
    target: ScanTarget = Field(..., description="Enumerated target")
    status: StatusType = Field(..., description="Enumeration status")
    start_time: datetime = Field(..., description="Enumeration start time")
    end_time: Optional[datetime] = Field(None, description="Enumeration end time")
    duration: Optional[float] = Field(None, gt=0, description="Enumeration duration in seconds")
    
    # DNS enumeration results
    dns_records: List[DNSRecord] = Field(default_factory=list, description="DNS records found")
    subdomains: List[SubdomainResult] = Field(default_factory=list, description="Subdomains found")
    zone_transfers: List[str] = Field(default_factory=list, description="Zone transfer results")
    
    # SMB enumeration results
    smb_shares: List[SMBShare] = Field(default_factory=list, description="SMB shares found")
    smb_users: List[str] = Field(default_factory=list, description="SMB users found")
    smb_groups: List[str] = Field(default_factory=list, description="SMB groups found")
    
    # SSH enumeration results
    ssh_host_keys: List[SSHHostKey] = Field(default_factory=list, description="SSH host keys")
    ssh_algorithms: Dict[str, List[str]] = Field(default_factory=dict, description="SSH algorithms")
    ssh_users: List[str] = Field(default_factory=list, description="SSH users found")
    
    error_message: Optional[str] = Field(None, description="Error message if enumeration failed")

# ============================================================================
# ATTACK MODELS
# ============================================================================

class Credential(BaseVideoOpusClipModel):
    """Credential information"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    service: str = Field(..., description="Service name")
    target: str = Field(..., description="Target host")
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    source: str = Field(..., description="Source of credential")
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Username cannot be empty")
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Password cannot be empty")
        return v.strip()

class ExploitResult(BaseVideoOpusClipModel):
    """Result of exploitation attempts"""
    exploit_type: str = Field(..., description="Type of exploit")
    target: str = Field(..., description="Target host")
    payload: str = Field(..., description="Exploit payload")
    success: bool = Field(..., description="Whether exploit was successful")
    response: Optional[str] = Field(None, description="Exploit response")
    shell_access: bool = Field(False, description="Whether shell access was obtained")
    data_extracted: Optional[Dict[str, Any]] = Field(None, description="Extracted data")
    error_message: Optional[str] = Field(None, description="Error message if exploit failed")

class AttackResult(TimestampedModel):
    """Result of attack operations"""
    attack_id: str = Field(..., description="Unique attack identifier")
    attack_type: AttackType = Field(..., description="Type of attack")
    target: ScanTarget = Field(..., description="Attack target")
    status: StatusType = Field(..., description="Attack status")
    start_time: datetime = Field(..., description="Attack start time")
    end_time: Optional[datetime] = Field(None, description="Attack end time")
    duration: Optional[float] = Field(None, gt=0, description="Attack duration in seconds")
    
    # Brute force results
    credentials_found: List[Credential] = Field(default_factory=list, description="Credentials discovered")
    attempts_made: int = Field(0, ge=0, description="Number of attempts made")
    total_combinations: int = Field(0, ge=0, description="Total combinations tested")
    
    # Exploitation results
    exploits_attempted: int = Field(0, ge=0, description="Number of exploits attempted")
    exploits_successful: int = Field(0, ge=0, description="Number of successful exploits")
    exploit_results: List[ExploitResult] = Field(default_factory=list, description="Exploit results")
    
    error_message: Optional[str] = Field(None, description="Error message if attack failed")

# ============================================================================
# SECURITY MODELS
# ============================================================================

class SecurityFinding(BaseVideoOpusClipModel):
    """Security finding information"""
    finding_id: str = Field(..., description="Unique finding identifier")
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Finding description")
    severity: SeverityLevel = Field(..., description="Finding severity")
    category: str = Field(..., description="Finding category")
    affected_component: str = Field(..., description="Affected component")
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    cvss_score: Optional[float] = Field(None, ge=0, le=10, description="CVSS score")
    remediation: str = Field(..., description="Remediation steps")
    references: List[str] = Field(default_factory=list, description="Reference links")
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    status: Literal["open", "in_progress", "resolved", "false_positive"] = Field("open", description="Finding status")

class SecurityAssessment(BaseVideoOpusClipModel):
    """Security assessment information"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    target: ScanTarget = Field(..., description="Assessment target")
    assessment_date: datetime = Field(default_factory=datetime.utcnow, description="Assessment date")
    security_score: float = Field(..., ge=0, le=100, description="Overall security score")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Risk level")
    
    # Findings summary
    total_findings: int = Field(0, ge=0, description="Total findings")
    critical_findings: int = Field(0, ge=0, description="Critical findings")
    high_findings: int = Field(0, ge=0, description="High findings")
    medium_findings: int = Field(0, ge=0, description="Medium findings")
    low_findings: int = Field(0, ge=0, description="Low findings")
    
    # Detailed findings
    findings: List[SecurityFinding] = Field(default_factory=list, description="Detailed findings")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")
    
    @validator('security_score')
    def validate_security_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Security score must be between 0 and 100")
        return v

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class DatabaseConfig(BaseVideoOpusClipModel):
    """Database configuration"""
    host: str = Field(..., description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    ssl_mode: str = Field("prefer", description="SSL mode")
    max_connections: int = Field(10, gt=0, description="Maximum connections")
    connection_timeout: float = Field(30.0, gt=0, description="Connection timeout")

class SecurityConfig(BaseVideoOpusClipModel):
    """Security configuration"""
    encryption_algorithm: EncryptionAlgorithm = Field(EncryptionAlgorithm.AES, description="Encryption algorithm")
    hash_algorithm: HashAlgorithm = Field(HashAlgorithm.SHA256, description="Hash algorithm")
    key_size: int = Field(256, description="Key size in bits")
    salt_size: int = Field(32, description="Salt size in bytes")
    iterations: int = Field(100000, description="PBKDF2 iterations")
    session_timeout: int = Field(3600, description="Session timeout in seconds")
    max_login_attempts: int = Field(5, description="Maximum login attempts")
    password_min_length: int = Field(8, description="Minimum password length")
    require_special_chars: bool = Field(True, description="Require special characters in passwords")
    
    @validator('key_size')
    def validate_key_size(cls, v):
        if v not in [128, 192, 256]:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        return v

class NetworkConfig(BaseVideoOpusClipModel):
    """Network configuration"""
    timeout: float = Field(30.0, gt=0, description="Network timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    user_agent: str = Field("Video-OpusClip/1.0", description="User agent string")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    follow_redirects: bool = Field(True, description="Follow HTTP redirects")
    max_redirects: int = Field(5, ge=0, description="Maximum redirects to follow")
    proxy_url: Optional[str] = Field(None, description="Proxy URL")
    proxy_username: Optional[str] = Field(None, description="Proxy username")
    proxy_password: Optional[str] = Field(None, description="Proxy password")

class ApplicationConfig(BaseVideoOpusClipModel):
    """Application configuration"""
    app_name: str = Field("Video-OpusClip", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")
    max_file_size: int = Field(10485760, description="Maximum file size in bytes")
    allowed_file_types: List[str] = Field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"], description="Allowed file types")
    upload_directory: str = Field("./uploads", description="Upload directory")
    temp_directory: str = Field("./temp", description="Temporary directory")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class BaseResponse(BaseVideoOpusClipModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: Literal[False] = False
    error_code: str = Field(..., description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")

class SuccessResponse(BaseResponse):
    """Success response model"""
    success: Literal[True] = True
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

class PaginatedResponse(BaseVideoOpusClipModel):
    """Paginated response model"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., gt=0, description="Items per page")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

# ============================================================================
# UTILITY MODELS
# ============================================================================

class FileInfo(BaseVideoOpusClipModel):
    """File information"""
    filename: str = Field(..., description="File name")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_type: str = Field(..., description="File type")
    checksum: Optional[str] = Field(None, description="File checksum")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="File creation time")
    modified_at: datetime = Field(default_factory=datetime.utcnow, description="File modification time")
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v < 0:
            raise ValueError("File size cannot be negative")
        return v

class SystemInfo(BaseVideoOpusClipModel):
    """System information"""
    hostname: str = Field(..., description="System hostname")
    platform: str = Field(..., description="Operating system platform")
    version: str = Field(..., description="System version")
    architecture: str = Field(..., description="System architecture")
    cpu_count: int = Field(..., gt=0, description="Number of CPU cores")
    memory_total: int = Field(..., gt=0, description="Total memory in bytes")
    memory_available: int = Field(..., ge=0, description="Available memory in bytes")
    disk_total: int = Field(..., gt=0, description="Total disk space in bytes")
    disk_available: int = Field(..., ge=0, description="Available disk space in bytes")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")

# ============================================================================
# TYPE ALIASES
# ============================================================================

# Common type aliases for better readability
HostPort = Tuple[str, int]
IPAddress = str
DomainName = str
URL = str
FilePath = str
JSONData = Dict[str, Any]
ListOfStrings = List[str]
OptionalString = Optional[str]
OptionalInt = Optional[int]
OptionalFloat = Optional[float]
OptionalBool = Optional[bool]
OptionalDateTime = Optional[datetime]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    import ipaddress
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def validate_domain_name(domain: str) -> bool:
    """Validate domain name format"""
    import re
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    return bool(re.match(pattern, domain))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def validate_file_path(path: str) -> bool:
    """Validate file path format"""
    import os
    try:
        os.path.normpath(path)
        return True
    except Exception:
        return False

# ============================================================================
# MODEL FACTORIES
# ============================================================================

def create_scan_target(host: str, port: Optional[int] = None, protocol: ProtocolType = ProtocolType.TCP) -> ScanTarget:
    """Create a scan target with validation"""
    return ScanTarget(host=host, port=port, protocol=protocol)

def create_security_finding(
    title: str,
    description: str,
    severity: SeverityLevel,
    category: str,
    affected_component: str,
    remediation: str
) -> SecurityFinding:
    """Create a security finding with auto-generated ID"""
    import uuid
    finding_id = str(uuid.uuid4())
    return SecurityFinding(
        finding_id=finding_id,
        title=title,
        description=description,
        severity=severity,
        category=category,
        affected_component=affected_component,
        remediation=remediation
    )

def create_credential(username: str, password: str, service: str, target: str, source: str) -> Credential:
    """Create a credential with validation"""
    return Credential(
        username=username,
        password=password,
        service=service,
        target=target,
        source=source
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of models
    print("📋 Video-OpusClip Models Example")
    
    # Create scan target
    target = create_scan_target("192.168.1.100", 80, ProtocolType.TCP)
    print(f"Scan target: {target}")
    
    # Create security finding
    finding = create_security_finding(
        title="SQL Injection Vulnerability",
        description="SQL injection vulnerability found in login form",
        severity=SeverityLevel.HIGH,
        category="Web Security",
        affected_component="Login Form",
        remediation="Use parameterized queries and input validation"
    )
    print(f"Security finding: {finding}")
    
    # Create credential
    credential = create_credential(
        username="admin",
        password="password123",
        service="SSH",
        target="192.168.1.100",
        source="Brute Force Attack"
    )
    print(f"Credential: {credential}")
    
    # Create scan configuration
    config = ScanConfiguration(
        scan_type=ScanType.PORT_SCAN,
        targets=[target],
        timeout=30.0,
        max_concurrent=5
    )
    print(f"Scan configuration: {config}")
    
    # Validate IP address
    print(f"IP validation: {validate_ip_address('192.168.1.1')}")
    print(f"Domain validation: {validate_domain_name('example.com')}")
    print(f"URL validation: {validate_url('https://example.com')}") 