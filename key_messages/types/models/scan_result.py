from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
        import ipaddress
            import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Scan result data model for cybersecurity tools.
"""

class ScanType(str, Enum):
    """Types of security scans."""
    PORT_SCAN = "port_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_SCAN = "web_scan"
    NETWORK_SCAN = "network_scan"
    SSL_SCAN = "ssl_scan"
    DNS_SCAN = "dns_scan"
    SERVICE_SCAN = "service_scan"

class ScanStatus(str, Enum):
    """Scan status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class PortStatus(str, Enum):
    """Port status."""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNKNOWN = "unknown"

class ServiceInfo(BaseModel):
    """Service information model."""
    name: str = Field(..., description="Service name")
    version: Optional[str] = Field(None, description="Service version")
    banner: Optional[str] = Field(None, description="Service banner")
    product: Optional[str] = Field(None, description="Product name")
    extra_info: Optional[str] = Field(None, description="Additional information")

class PortInfo(BaseModel):
    """Port information model."""
    port: int = Field(..., ge=1, le=65535, description="Port number")
    protocol: str = Field(default="tcp", description="Protocol (tcp/udp)")
    status: PortStatus = Field(..., description="Port status")
    service: Optional[ServiceInfo] = Field(None, description="Service information")
    response_time: Optional[float] = Field(None, ge=0.0, description="Response time in seconds")
    is_common_port: bool = Field(default=False, description="Whether this is a common port")

class VulnerabilityInfo(BaseModel):
    """Vulnerability information model."""
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    title: str = Field(..., description="Vulnerability title")
    description: str = Field(..., description="Vulnerability description")
    severity: str = Field(..., description="Severity level")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="CVSS score")
    references: List[str] = Field(default_factory=list, description="Reference links")
    affected_services: List[str] = Field(default_factory=list, description="Affected services")

class ScanResultModel(BaseModel):
    """Scan result data model."""
    
    # Core fields
    id: str = Field(..., description="Unique scan result identifier")
    scan_type: ScanType = Field(..., description="Type of scan performed")
    target: str = Field(..., description="Target hostname or IP address")
    
    # Status and timing
    status: ScanStatus = Field(default=ScanStatus.PENDING, description="Scan status")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Scan start time")
    completed_at: Optional[datetime] = Field(None, description="Scan completion time")
    duration: Optional[float] = Field(None, ge=0.0, description="Scan duration in seconds")
    
    # Results
    open_ports: List[PortInfo] = Field(default_factory=list, description="Open ports found")
    closed_ports: List[PortInfo] = Field(default_factory=list, description="Closed ports found")
    filtered_ports: List[PortInfo] = Field(default_factory=list, description="Filtered ports found")
    
    # Vulnerabilities
    vulnerabilities: List[VulnerabilityInfo] = Field(default_factory=list, description="Vulnerabilities found")
    total_vulnerabilities: int = Field(default=0, description="Total number of vulnerabilities")
    critical_vulnerabilities: int = Field(default=0, description="Number of critical vulnerabilities")
    high_vulnerabilities: int = Field(default=0, description="Number of high vulnerabilities")
    medium_vulnerabilities: int = Field(default=0, description="Number of medium vulnerabilities")
    low_vulnerabilities: int = Field(default=0, description="Number of low vulnerabilities")
    
    # Services
    services_found: List[str] = Field(default_factory=list, description="Services discovered")
    unique_services: int = Field(default=0, description="Number of unique services")
    
    # Network information
    hostname: Optional[str] = Field(None, description="Resolved hostname")
    ip_addresses: List[str] = Field(default_factory=list, description="IP addresses")
    mac_address: Optional[str] = Field(None, description="MAC address")
    os_detection: Optional[str] = Field(None, description="Operating system detection")
    
    # SSL/TLS information
    ssl_certificate: Optional[Dict[str, Any]] = Field(None, description="SSL certificate information")
    ssl_issues: List[str] = Field(default_factory=list, description="SSL/TLS issues found")
    
    # DNS information
    dns_records: Dict[str, List[str]] = Field(default_factory=dict, description="DNS records")
    reverse_dns: Optional[str] = Field(None, description="Reverse DNS lookup result")
    
    # Statistics
    total_ports_scanned: int = Field(default=0, description="Total ports scanned")
    ports_per_second: Optional[float] = Field(None, ge=0.0, description="Scan speed in ports per second")
    
    # Configuration
    scan_config: Dict[str, Any] = Field(default_factory=dict, description="Scan configuration used")
    scan_range: Optional[str] = Field(None, description="Port range scanned")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Errors encountered during scan")
    warnings: List[str] = Field(default_factory=list, description="Warnings during scan")
    
    # Metadata
    scanner_version: Optional[str] = Field(None, description="Scanner version used")
    scan_profile: Optional[str] = Field(None, description="Scan profile used")
    tags: List[str] = Field(default_factory=list, description="Scan tags")
    
    # Custom fields
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('target')
    def validate_target(cls, v) -> Optional[Dict[str, Any]]:
        if not v.strip():
            raise ValueError("Target cannot be empty")
        return v.strip()
    
    @field_validator('ip_addresses')
    def validate_ip_addresses(cls, v) -> bool:
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
    
    def is_completed(self) -> bool:
        """Check if scan is completed."""
        return self.status in [ScanStatus.COMPLETED, ScanStatus.FAILED, ScanStatus.CANCELLED, ScanStatus.TIMEOUT]
    
    def is_successful(self) -> bool:
        """Check if scan completed successfully."""
        return self.status == ScanStatus.COMPLETED
    
    def get_risk_score(self) -> float:
        """Calculate overall risk score based on vulnerabilities."""
        score = 0.0
        
        # Weight vulnerabilities by severity
        score += self.critical_vulnerabilities * 10.0
        score += self.high_vulnerabilities * 7.0
        score += self.medium_vulnerabilities * 4.0
        score += self.low_vulnerabilities * 1.0
        
        # Normalize to 0-10 scale
        return min(score / 10.0, 10.0)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary of the scan."""
        return {
            "total_ports": self.total_ports_scanned,
            "open_ports": len(self.open_ports),
            "services_found": self.unique_services,
            "vulnerabilities": {
                "total": self.total_vulnerabilities,
                "critical": self.critical_vulnerabilities,
                "high": self.high_vulnerabilities,
                "medium": self.medium_vulnerabilities,
                "low": self.low_vulnerabilities
            },
            "risk_score": self.get_risk_score(),
            "ssl_issues": len(self.ssl_issues),
            "scan_duration": self.duration
        }
    
    def add_vulnerability(self, vulnerability: VulnerabilityInfo) -> None:
        """Add a vulnerability to the scan result."""
        self.vulnerabilities.append(vulnerability)
        self.total_vulnerabilities += 1
        
        # Update severity counts
        severity = vulnerability.severity.lower()
        if severity == "critical":
            self.critical_vulnerabilities += 1
        elif severity == "high":
            self.high_vulnerabilities += 1
        elif severity == "medium":
            self.medium_vulnerabilities += 1
        elif severity == "low":
            self.low_vulnerabilities += 1
    
    def add_port(self, port_info: PortInfo) -> None:
        """Add a port to the appropriate list."""
        if port_info.status == PortStatus.OPEN:
            self.open_ports.append(port_info)
        elif port_info.status == PortStatus.CLOSED:
            self.closed_ports.append(port_info)
        elif port_info.status == PortStatus.FILTERED:
            self.filtered_ports.append(port_info)
        
        # Update service count
        if port_info.service and port_info.service.name not in self.services_found:
            self.services_found.append(port_info.service.name)
            self.unique_services += 1
    
    def complete_scan(self, duration: float) -> None:
        """Mark scan as completed."""
        self.status = ScanStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.duration = duration
        
        # Calculate scan speed
        if duration > 0:
            self.ports_per_second = self.total_ports_scanned / duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScanResultModel':
        """Create model from dictionary."""
        return cls(**data)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 