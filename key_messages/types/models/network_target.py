from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, List, Dict, Any, Union
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
Network target data model for cybersecurity tools.
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

class ServiceInfo(BaseModel):
    """Service information."""
    name: str = Field(..., description="Service name")
    port: int = Field(..., ge=1, le=65535, description="Service port")
    protocol: str = Field(default="tcp", description="Protocol")
    version: Optional[str] = Field(None, description="Service version")
    banner: Optional[str] = Field(None, description="Service banner")
    is_secure: bool = Field(default=False, description="Whether service uses encryption")

class VulnerabilityInfo(BaseModel):
    """Vulnerability information."""
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    title: str = Field(..., description="Vulnerability title")
    severity: str = Field(..., description="Vulnerability severity")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="CVSS score")
    description: str = Field(..., description="Vulnerability description")
    affected_service: Optional[str] = Field(None, description="Affected service")
    remediation: Optional[str] = Field(None, description="Remediation steps")

class NetworkTargetModel(BaseModel):
    """Network target data model."""
    
    # Core fields
    id: str = Field(..., description="Unique target identifier")
    name: str = Field(..., description="Target name")
    target_type: TargetType = Field(..., description="Type of target")
    
    # Network information
    ip_address: Optional[str] = Field(None, description="IP address")
    ip_addresses: List[str] = Field(default_factory=list, description="Multiple IP addresses")
    hostname: Optional[str] = Field(None, description="Hostname")
    domain: Optional[str] = Field(None, description="Domain name")
    url: Optional[str] = Field(None, description="URL")
    
    # Network details
    mac_address: Optional[str] = Field(None, description="MAC address")
    network_interface: Optional[str] = Field(None, description="Network interface")
    subnet: Optional[str] = Field(None, description="Subnet")
    gateway: Optional[str] = Field(None, description="Gateway address")
    dns_servers: List[str] = Field(default_factory=list, description="DNS servers")
    
    # System information
    operating_system: OperatingSystem = Field(default=OperatingSystem.UNKNOWN, description="Operating system")
    os_version: Optional[str] = Field(None, description="OS version")
    os_details: Optional[str] = Field(None, description="OS details")
    architecture: Optional[str] = Field(None, description="System architecture")
    
    # Services and applications
    services: List[ServiceInfo] = Field(default_factory=list, description="Running services")
    applications: List[str] = Field(default_factory=list, description="Installed applications")
    open_ports: List[int] = Field(default_factory=list, description="Open ports")
    closed_ports: List[int] = Field(default_factory=list, description="Closed ports")
    filtered_ports: List[int] = Field(default_factory=list, description="Filtered ports")
    
    # Security assessment
    status: TargetStatus = Field(default=TargetStatus.PENDING, description="Target status")
    risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Risk score")
    security_level: Optional[str] = Field(None, description="Security level")
    vulnerabilities: List[VulnerabilityInfo] = Field(default_factory=list, description="Vulnerabilities found")
    total_vulnerabilities: int = Field(default=0, description="Total vulnerabilities")
    critical_vulnerabilities: int = Field(default=0, description="Critical vulnerabilities")
    high_vulnerabilities: int = Field(default=0, description="High vulnerabilities")
    medium_vulnerabilities: int = Field(default=0, description="Medium vulnerabilities")
    low_vulnerabilities: int = Field(default=0, description="Low vulnerabilities")
    
    # Access and authentication
    has_remote_access: bool = Field(default=False, description="Has remote access enabled")
    ssh_enabled: bool = Field(default=False, description="SSH enabled")
    telnet_enabled: bool = Field(default=False, description="Telnet enabled")
    rdp_enabled: bool = Field(default=False, description="RDP enabled")
    vnc_enabled: bool = Field(default=False, description="VNC enabled")
    
    # Web services
    web_server: Optional[str] = Field(None, description="Web server type")
    web_server_version: Optional[str] = Field(None, description="Web server version")
    ssl_enabled: bool = Field(default=False, description="SSL/TLS enabled")
    ssl_certificate: Optional[Dict[str, Any]] = Field(None, description="SSL certificate info")
    ssl_issues: List[str] = Field(default_factory=list, description="SSL/TLS issues")
    
    # Database services
    database_services: List[str] = Field(default_factory=list, description="Database services")
    database_versions: Dict[str, str] = Field(default_factory=dict, description="Database versions")
    
    # Network protocols
    supported_protocols: List[str] = Field(default_factory=list, description="Supported protocols")
    active_connections: List[Dict[str, Any]] = Field(default_factory=list, description="Active connections")
    
    # Discovery and scanning
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    last_scanned: Optional[datetime] = Field(None, description="Last scan timestamp")
    scan_frequency: Optional[str] = Field(None, description="Scan frequency")
    next_scan: Optional[datetime] = Field(None, description="Next scheduled scan")
    
    # Ownership and classification
    owner: Optional[str] = Field(None, description="Target owner")
    department: Optional[str] = Field(None, description="Department")
    location: Optional[str] = Field(None, description="Physical location")
    environment: Optional[str] = Field(None, description="Environment (prod, dev, test)")
    criticality: Optional[str] = Field(None, description="Business criticality")
    
    # Compliance and policies
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    security_policies: List[str] = Field(default_factory=list, description="Security policies")
    patch_level: Optional[str] = Field(None, description="Patch level")
    last_patched: Optional[datetime] = Field(None, description="Last patch date")
    
    # Monitoring and alerts
    is_monitored: bool = Field(default=False, description="Is under monitoring")
    alert_threshold: Optional[float] = Field(None, description="Alert threshold")
    alert_contacts: List[str] = Field(default_factory=list, description="Alert contacts")
    
    # History and tracking
    first_seen: datetime = Field(default_factory=datetime.utcnow, description="First seen timestamp")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last seen timestamp")
    uptime: Optional[float] = Field(None, description="Uptime percentage")
    
    # Custom fields
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Target tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    
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
    
    def is_online(self) -> bool:
        """Check if target is online."""
        return self.status not in [TargetStatus.OFFLINE, TargetStatus.UNKNOWN]
    
    def is_compromised(self) -> bool:
        """Check if target is compromised."""
        return self.status == TargetStatus.COMPROMISED
    
    def is_secure(self) -> bool:
        """Check if target is secure."""
        return self.status == TargetStatus.SECURE
    
    def get_risk_level(self) -> str:
        """Get risk level based on risk score."""
        if self.risk_score is None:
            return "unknown"
        elif self.risk_score >= 8.0:
            return "critical"
        elif self.risk_score >= 6.0:
            return "high"
        elif self.risk_score >= 4.0:
            return "medium"
        elif self.risk_score >= 2.0:
            return "low"
        else:
            return "minimal"
    
    def calculate_risk_score(self) -> float:
        """Calculate risk score based on vulnerabilities and configuration."""
        score = 0.0
        
        # Vulnerability-based scoring
        score += self.critical_vulnerabilities * 10.0
        score += self.high_vulnerabilities * 7.0
        score += self.medium_vulnerabilities * 4.0
        score += self.low_vulnerabilities * 1.0
        
        # Configuration-based scoring
        if self.telnet_enabled:
            score += 5.0  # Telnet is insecure
        if not self.ssl_enabled and any('http' in service.name.lower() for service in self.services):
            score += 3.0  # HTTP without SSL
        if self.ssh_enabled and not any('ssh' in service.name.lower() for service in self.services):
            score += 2.0  # SSH not properly configured
        
        # Normalize to 0-10 scale
        return min(score / 10.0, 10.0)
    
    def add_service(self, service: ServiceInfo) -> None:
        """Add a service to the target."""
        self.services.append(service)
        if service.port not in self.open_ports:
            self.open_ports.append(service.port)
    
    def add_vulnerability(self, vulnerability: VulnerabilityInfo) -> None:
        """Add a vulnerability to the target."""
        self.vulnerabilities.append(vulnerability)
        self.total_vulnerabilities += 1
        
        # Update severity counts
        if vulnerability.severity.lower() == "critical":
            self.critical_vulnerabilities += 1
        elif vulnerability.severity.lower() == "high":
            self.high_vulnerabilities += 1
        elif vulnerability.severity.lower() == "medium":
            self.medium_vulnerabilities += 1
        elif vulnerability.severity.lower() == "low":
            self.low_vulnerabilities += 1
    
    def update_status(self, status: TargetStatus) -> None:
        """Update target status."""
        self.status = status
        self.last_seen = datetime.utcnow()
    
    def mark_scanned(self) -> None:
        """Mark target as scanned."""
        self.last_scanned = datetime.utcnow()
        if self.status == TargetStatus.PENDING:
            self.status = TargetStatus.SCANNED
    
    def get_service_by_port(self, port: int) -> Optional[ServiceInfo]:
        """Get service information by port."""
        for service in self.services:
            if service.port == port:
                return service
        return None
    
    def get_vulnerabilities_by_severity(self, severity: str) -> List[VulnerabilityInfo]:
        """Get vulnerabilities by severity level."""
        return [v for v in self.vulnerabilities if v.severity.lower() == severity.lower()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkTargetModel':
        """Create model from dictionary."""
        return cls(**data)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 