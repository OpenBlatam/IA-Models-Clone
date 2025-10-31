from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
            import ipaddress
            import ipaddress
            import re
            import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Network target API schemas for cybersecurity tools.
"""

class TargetType(str, Enum):
    """Types of network targets."""
    HOST = "host"
    NETWORK = "network"
    SUBNET = "subnet"
    DOMAIN = "domain"
    URL = "url"
    SERVICE = "service"
    APPLICATION = "application"

class TargetStatus(str, Enum):
    """Target status."""
    PENDING = "pending"
    ACTIVE = "active"
    SCANNED = "scanned"
    COMPROMISED = "compromised"
    SECURE = "secure"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class OperatingSystem(str, Enum):
    """Operating system types."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    BSD = "bsd"
    SOLARIS = "solaris"
    ANDROID = "android"
    IOS = "ios"
    UNKNOWN = "unknown"

# Request Schemas
class CreateNetworkTargetRequest(BaseModel):
    """Request schema for creating a network target."""
    name: str = Field(..., min_length=1, max_length=200, description="Target name")
    target_type: TargetType = Field(..., description="Type of target")
    ip_address: Optional[str] = Field(None, description="IP address")
    ip_addresses: List[str] = Field(default_factory=list, description="Multiple IP addresses")
    hostname: Optional[str] = Field(None, description="Hostname")
    domain: Optional[str] = Field(None, description="Domain name")
    url: Optional[str] = Field(None, description="URL")
    mac_address: Optional[str] = Field(None, description="MAC address")
    network_interface: Optional[str] = Field(None, description="Network interface")
    subnet: Optional[str] = Field(None, description="Subnet")
    gateway: Optional[str] = Field(None, description="Gateway address")
    dns_servers: List[str] = Field(default_factory=list, description="DNS servers")
    operating_system: OperatingSystem = Field(default=OperatingSystem.UNKNOWN, description="Operating system")
    os_version: Optional[str] = Field(None, description="OS version")
    os_details: Optional[str] = Field(None, description="OS details")
    architecture: Optional[str] = Field(None, description="System architecture")
    owner: Optional[str] = Field(None, description="Target owner")
    department: Optional[str] = Field(None, description="Department")
    location: Optional[str] = Field(None, description="Physical location")
    environment: Optional[str] = Field(None, description="Environment (prod, dev, test)")
    criticality: Optional[str] = Field(None, description="Business criticality")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    security_policies: List[str] = Field(default_factory=list, description="Security policies")
    is_monitored: bool = Field(default=False, description="Is under monitoring")
    alert_threshold: Optional[float] = Field(None, description="Alert threshold")
    alert_contacts: List[str] = Field(default_factory=list, description="Alert contacts")
    scan_frequency: Optional[str] = Field(None, description="Scan frequency")
    tags: List[str] = Field(default_factory=list, description="Target tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('name')
    def validate_name(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('ip_address', 'ip_addresses')
    def validate_ip_addresses(cls, v) -> bool:
        if isinstance(v, str):
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f"Invalid IP address: {v}")
        elif isinstance(v, list):
            for ip in v:
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    raise ValueError(f"Invalid IP address: {ip}")
        return v
    
    @field_validator('mac_address')
    def validate_mac_address(cls, v) -> bool:
        if v is not None:
            mac_pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
            if not re.match(mac_pattern, v):
                raise ValueError(f"Invalid MAC address format: {v}")
        return v
    
    @field_validator('url')
    def validate_url(cls, v) -> bool:
        if v is not None:
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, v):
                raise ValueError(f"Invalid URL format: {v}")
        return v

class UpdateNetworkTargetRequest(BaseModel):
    """Request schema for updating a network target."""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Target name")
    target_type: Optional[TargetType] = Field(None, description="Type of target")
    ip_address: Optional[str] = Field(None, description="IP address")
    ip_addresses: Optional[List[str]] = Field(None, description="Multiple IP addresses")
    hostname: Optional[str] = Field(None, description="Hostname")
    domain: Optional[str] = Field(None, description="Domain name")
    url: Optional[str] = Field(None, description="URL")
    mac_address: Optional[str] = Field(None, description="MAC address")
    network_interface: Optional[str] = Field(None, description="Network interface")
    subnet: Optional[str] = Field(None, description="Subnet")
    gateway: Optional[str] = Field(None, description="Gateway address")
    dns_servers: Optional[List[str]] = Field(None, description="DNS servers")
    operating_system: Optional[OperatingSystem] = Field(None, description="Operating system")
    os_version: Optional[str] = Field(None, description="OS version")
    os_details: Optional[str] = Field(None, description="OS details")
    architecture: Optional[str] = Field(None, description="System architecture")
    status: Optional[TargetStatus] = Field(None, description="Target status")
    owner: Optional[str] = Field(None, description="Target owner")
    department: Optional[str] = Field(None, description="Department")
    location: Optional[str] = Field(None, description="Physical location")
    environment: Optional[str] = Field(None, description="Environment")
    criticality: Optional[str] = Field(None, description="Business criticality")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Compliance frameworks")
    security_policies: Optional[List[str]] = Field(None, description="Security policies")
    is_monitored: Optional[bool] = Field(None, description="Is under monitoring")
    alert_threshold: Optional[float] = Field(None, description="Alert threshold")
    alert_contacts: Optional[List[str]] = Field(None, description="Alert contacts")
    scan_frequency: Optional[str] = Field(None, description="Scan frequency")
    tags: Optional[List[str]] = Field(None, description="Target tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class NetworkTargetFilterRequest(BaseModel):
    """Request schema for filtering network targets."""
    target_type: Optional[TargetType] = Field(None, description="Filter by target type")
    status: Optional[TargetStatus] = Field(None, description="Filter by status")
    operating_system: Optional[OperatingSystem] = Field(None, description="Filter by operating system")
    owner: Optional[str] = Field(None, description="Filter by owner")
    department: Optional[str] = Field(None, description="Filter by department")
    location: Optional[str] = Field(None, description="Filter by location")
    environment: Optional[str] = Field(None, description="Filter by environment")
    criticality: Optional[str] = Field(None, description="Filter by criticality")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Filter by compliance frameworks")
    is_monitored: Optional[bool] = Field(None, description="Filter by monitoring status")
    subnet: Optional[str] = Field(None, description="Filter by subnet")
    discovered_after: Optional[datetime] = Field(None, description="Filter by discovery date (after)")
    discovered_before: Optional[datetime] = Field(None, description="Filter by discovery date (before)")
    last_scanned_after: Optional[datetime] = Field(None, description="Filter by last scan date (after)")
    last_scanned_before: Optional[datetime] = Field(None, description="Filter by last scan date (before)")
    search: Optional[str] = Field(None, description="Search in name, hostname, and IP")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")

# Response Schemas
class ServiceInfoResponse(BaseModel):
    """Response schema for service information."""
    name: str = Field(..., description="Service name")
    port: int = Field(..., description="Service port")
    protocol: str = Field(..., description="Protocol")
    version: Optional[str] = Field(None, description="Service version")
    banner: Optional[str] = Field(None, description="Service banner")
    is_secure: bool = Field(..., description="Whether service uses encryption")

class VulnerabilityInfoResponse(BaseModel):
    """Response schema for vulnerability information."""
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    title: str = Field(..., description="Vulnerability title")
    severity: str = Field(..., description="Vulnerability severity")
    cvss_score: Optional[float] = Field(None, description="CVSS score")
    description: str = Field(..., description="Vulnerability description")
    affected_service: Optional[str] = Field(None, description="Affected service")
    remediation: Optional[str] = Field(None, description="Remediation steps")

class NetworkTargetResponse(BaseModel):
    """Response schema for network target data."""
    id: str = Field(..., description="Unique target identifier")
    name: str = Field(..., description="Target name")
    target_type: TargetType = Field(..., description="Type of target")
    ip_address: Optional[str] = Field(None, description="IP address")
    ip_addresses: List[str] = Field(..., description="Multiple IP addresses")
    hostname: Optional[str] = Field(None, description="Hostname")
    domain: Optional[str] = Field(None, description="Domain name")
    url: Optional[str] = Field(None, description="URL")
    mac_address: Optional[str] = Field(None, description="MAC address")
    network_interface: Optional[str] = Field(None, description="Network interface")
    subnet: Optional[str] = Field(None, description="Subnet")
    gateway: Optional[str] = Field(None, description="Gateway address")
    dns_servers: List[str] = Field(..., description="DNS servers")
    operating_system: OperatingSystem = Field(..., description="Operating system")
    os_version: Optional[str] = Field(None, description="OS version")
    os_details: Optional[str] = Field(None, description="OS details")
    architecture: Optional[str] = Field(None, description="System architecture")
    services: List[ServiceInfoResponse] = Field(..., description="Running services")
    applications: List[str] = Field(..., description="Installed applications")
    open_ports: List[int] = Field(..., description="Open ports")
    closed_ports: List[int] = Field(..., description="Closed ports")
    filtered_ports: List[int] = Field(..., description="Filtered ports")
    status: TargetStatus = Field(..., description="Target status")
    risk_score: Optional[float] = Field(None, description="Risk score")
    security_level: Optional[str] = Field(None, description="Security level")
    vulnerabilities: List[VulnerabilityInfoResponse] = Field(..., description="Vulnerabilities found")
    total_vulnerabilities: int = Field(..., description="Total vulnerabilities")
    critical_vulnerabilities: int = Field(..., description="Critical vulnerabilities")
    high_vulnerabilities: int = Field(..., description="High vulnerabilities")
    medium_vulnerabilities: int = Field(..., description="Medium vulnerabilities")
    low_vulnerabilities: int = Field(..., description="Low vulnerabilities")
    has_remote_access: bool = Field(..., description="Has remote access enabled")
    ssh_enabled: bool = Field(..., description="SSH enabled")
    telnet_enabled: bool = Field(..., description="Telnet enabled")
    rdp_enabled: bool = Field(..., description="RDP enabled")
    vnc_enabled: bool = Field(..., description="VNC enabled")
    web_server: Optional[str] = Field(None, description="Web server type")
    web_server_version: Optional[str] = Field(None, description="Web server version")
    ssl_enabled: bool = Field(..., description="SSL/TLS enabled")
    ssl_certificate: Optional[Dict[str, Any]] = Field(None, description="SSL certificate info")
    ssl_issues: List[str] = Field(..., description="SSL/TLS issues")
    database_services: List[str] = Field(..., description="Database services")
    database_versions: Dict[str, str] = Field(..., description="Database versions")
    supported_protocols: List[str] = Field(..., description="Supported protocols")
    active_connections: List[Dict[str, Any]] = Field(..., description="Active connections")
    discovered_at: datetime = Field(..., description="Discovery timestamp")
    last_scanned: Optional[datetime] = Field(None, description="Last scan timestamp")
    scan_frequency: Optional[str] = Field(None, description="Scan frequency")
    next_scan: Optional[datetime] = Field(None, description="Next scheduled scan")
    owner: Optional[str] = Field(None, description="Target owner")
    department: Optional[str] = Field(None, description="Department")
    location: Optional[str] = Field(None, description="Physical location")
    environment: Optional[str] = Field(None, description="Environment")
    criticality: Optional[str] = Field(None, description="Business criticality")
    compliance_frameworks: List[str] = Field(..., description="Compliance frameworks")
    compliance_status: Dict[str, str] = Field(..., description="Compliance status by framework")
    gaps_identified: List[str] = Field(..., description="Compliance gaps identified")
    security_policies: List[str] = Field(..., description="Security policies")
    patch_level: Optional[str] = Field(None, description="Patch level")
    last_patched: Optional[datetime] = Field(None, description="Last patch date")
    is_monitored: bool = Field(..., description="Is under monitoring")
    alert_threshold: Optional[float] = Field(None, description="Alert threshold")
    alert_contacts: List[str] = Field(..., description="Alert contacts")
    first_seen: datetime = Field(..., description="First seen timestamp")
    last_seen: datetime = Field(..., description="Last seen timestamp")
    uptime: Optional[float] = Field(None, description="Uptime percentage")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    tags: List[str] = Field(..., description="Target tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    is_online: bool = Field(..., description="Whether target is online")
    is_compromised: bool = Field(..., description="Whether target is compromised")
    is_secure: bool = Field(..., description="Whether target is secure")
    risk_level: str = Field(..., description="Risk level")

class NetworkTargetListResponse(BaseModel):
    """Response schema for network target list."""
    targets: List[NetworkTargetResponse] = Field(..., description="List of network targets")
    total_count: int = Field(..., description="Total number of targets")
    filtered_count: int = Field(..., description="Number of targets after filtering")
    page: Optional[int] = Field(None, description="Current page number")
    page_size: Optional[int] = Field(None, description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")

class NetworkTargetStatsResponse(BaseModel):
    """Response schema for network target statistics."""
    total_targets: int = Field(..., description="Total number of targets")
    online_targets: int = Field(..., description="Number of online targets")
    offline_targets: int = Field(..., description="Number of offline targets")
    compromised_targets: int = Field(..., description="Number of compromised targets")
    secure_targets: int = Field(..., description="Number of secure targets")
    targets_by_type: Dict[str, int] = Field(..., description="Targets count by type")
    targets_by_status: Dict[str, int] = Field(..., description="Targets count by status")
    targets_by_os: Dict[str, int] = Field(..., description="Targets count by operating system")
    targets_by_environment: Dict[str, int] = Field(..., description="Targets count by environment")
    total_vulnerabilities: int = Field(..., description="Total vulnerabilities across all targets")
    average_risk_score: float = Field(..., description="Average risk score")
    monitored_targets: int = Field(..., description="Number of monitored targets")
    recent_targets: List[NetworkTargetResponse] = Field(..., description="Recent targets")

class NetworkTargetCreateResponse(BaseModel):
    """Response schema for network target creation."""
    id: str = Field(..., description="Created target identifier")
    message: str = Field(..., description="Success message")
    created_at: datetime = Field(..., description="Creation timestamp")

class NetworkTargetUpdateResponse(BaseModel):
    """Response schema for network target update."""
    id: str = Field(..., description="Updated target identifier")
    message: str = Field(..., description="Success message")
    updated_at: datetime = Field(..., description="Update timestamp")
    changes: Dict[str, Any] = Field(..., description="Changes made")

class NetworkTargetDeleteResponse(BaseModel):
    """Response schema for network target deletion."""
    id: str = Field(..., description="Deleted target identifier")
    message: str = Field(..., description="Success message")
    deleted_at: datetime = Field(..., description="Deletion timestamp")

# Error Schemas
class NetworkTargetErrorResponse(BaseModel):
    """Error response schema for network target operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Bulk Operation Schemas
class BulkNetworkTargetRequest(BaseModel):
    """Request schema for bulk network target operations."""
    operation: str = Field(..., description="Bulk operation type")
    target_ids: List[str] = Field(..., description="List of target IDs")
    updates: Optional[Dict[str, Any]] = Field(None, description="Updates to apply")

class BulkNetworkTargetResponse(BaseModel):
    """Response schema for bulk network target operations."""
    operation: str = Field(..., description="Bulk operation type")
    total_targets: int = Field(..., description="Total number of targets")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Operation errors")
    completed_at: datetime = Field(..., description="Completion timestamp")

# Export Schemas
class NetworkTargetExportRequest(BaseModel):
    """Request schema for network target export."""
    format: str = Field(..., description="Export format (json, csv, xml)")
    filters: Optional[NetworkTargetFilterRequest] = Field(None, description="Export filters")
    include_services: bool = Field(default=True, description="Include service details")
    include_vulnerabilities: bool = Field(default=True, description="Include vulnerability details")
    include_metadata: bool = Field(default=True, description="Include metadata")

class NetworkTargetExportResponse(BaseModel):
    """Response schema for network target export."""
    download_url: str = Field(..., description="Download URL for exported file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Download link expiration")
    target_count: int = Field(..., description="Number of targets exported") 