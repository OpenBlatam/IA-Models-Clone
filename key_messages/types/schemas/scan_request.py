from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
            import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Scan request API schemas for cybersecurity tools.
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

# Request Schemas
class CreateScanRequest(BaseModel):
    """Request schema for creating a scan."""
    scan_type: ScanType = Field(..., description="Type of scan to perform")
    target: str = Field(..., description="Target hostname, IP, or URL")
    ports: Optional[List[int]] = Field(None, description="Ports to scan")
    port_range: Optional[str] = Field(None, description="Port range (e.g., '1-1000')")
    timeout: Optional[float] = Field(None, ge=1.0, le=3600.0, description="Scan timeout in seconds")
    max_workers: Optional[int] = Field(None, ge=1, le=100, description="Maximum concurrent workers")
    check_services: bool = Field(default=True, description="Check for running services")
    get_banners: bool = Field(default=False, description="Get service banners")
    ssl_check: bool = Field(default=True, description="Check SSL/TLS certificates")
    dns_check: bool = Field(default=True, description="Perform DNS lookups")
    vulnerability_check: bool = Field(default=False, description="Check for vulnerabilities")
    scan_profile: Optional[str] = Field(None, description="Scan profile to use")
    custom_options: Dict[str, Any] = Field(default_factory=dict, description="Custom scan options")
    priority: Optional[str] = Field(None, description="Scan priority")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled scan time")
    
    @field_validator('target')
    def validate_target(cls, v) -> Optional[Dict[str, Any]]:
        if not v.strip():
            raise ValueError("Target cannot be empty")
        return v.strip()
    
    @field_validator('ports')
    def validate_ports(cls, v) -> bool:
        if v is not None:
            for port in v:
                if port < 1 or port > 65535:
                    raise ValueError(f"Port {port} must be between 1 and 65535")
        return v
    
    @field_validator('port_range')
    def validate_port_range(cls, v) -> bool:
        if v is not None:
            range_pattern = r'^\d+-\d+$'
            if not re.match(range_pattern, v):
                raise ValueError("Port range must be in format 'start-end'")
            start, end = map(int, v.split('-'))
            if start < 1 or end > 65535 or start > end:
                raise ValueError("Invalid port range")
        return v

class UpdateScanRequest(BaseModel):
    """Request schema for updating a scan."""
    status: Optional[ScanStatus] = Field(None, description="Scan status")
    priority: Optional[str] = Field(None, description="Scan priority")
    timeout: Optional[float] = Field(None, ge=1.0, le=3600.0, description="Scan timeout")
    custom_options: Optional[Dict[str, Any]] = Field(None, description="Custom scan options")
    notes: Optional[str] = Field(None, description="Scan notes")

class ScanFilterRequest(BaseModel):
    """Request schema for filtering scans."""
    scan_type: Optional[ScanType] = Field(None, description="Filter by scan type")
    status: Optional[ScanStatus] = Field(None, description="Filter by status")
    target: Optional[str] = Field(None, description="Filter by target")
    started_after: Optional[datetime] = Field(None, description="Filter by start date (after)")
    started_before: Optional[datetime] = Field(None, description="Filter by start date (before)")
    completed_after: Optional[datetime] = Field(None, description="Filter by completion date (after)")
    completed_before: Optional[datetime] = Field(None, description="Filter by completion date (before)")
    priority: Optional[str] = Field(None, description="Filter by priority")
    scan_profile: Optional[str] = Field(None, description="Filter by scan profile")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")

# Response Schemas
class ServiceInfoResponse(BaseModel):
    """Response schema for service information."""
    name: str = Field(..., description="Service name")
    version: Optional[str] = Field(None, description="Service version")
    banner: Optional[str] = Field(None, description="Service banner")
    product: Optional[str] = Field(None, description="Product name")
    extra_info: Optional[str] = Field(None, description="Additional information")

class PortInfoResponse(BaseModel):
    """Response schema for port information."""
    port: int = Field(..., description="Port number")
    protocol: str = Field(..., description="Protocol (tcp/udp)")
    status: PortStatus = Field(..., description="Port status")
    service: Optional[ServiceInfoResponse] = Field(None, description="Service information")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    is_common_port: bool = Field(..., description="Whether this is a common port")

class VulnerabilityInfoResponse(BaseModel):
    """Response schema for vulnerability information."""
    cve_id: Optional[str] = Field(None, description="CVE identifier")
    title: str = Field(..., description="Vulnerability title")
    description: str = Field(..., description="Vulnerability description")
    severity: str = Field(..., description="Severity level")
    cvss_score: Optional[float] = Field(None, description="CVSS score")
    references: List[str] = Field(..., description="Reference links")
    affected_services: List[str] = Field(..., description="Affected services")

class ScanResultResponse(BaseModel):
    """Response schema for scan result data."""
    id: str = Field(..., description="Unique scan result identifier")
    scan_type: ScanType = Field(..., description="Type of scan performed")
    target: str = Field(..., description="Target hostname or IP address")
    status: ScanStatus = Field(..., description="Scan status")
    started_at: datetime = Field(..., description="Scan start time")
    completed_at: Optional[datetime] = Field(None, description="Scan completion time")
    duration: Optional[float] = Field(None, description="Scan duration in seconds")
    open_ports: List[PortInfoResponse] = Field(..., description="Open ports found")
    closed_ports: List[PortInfoResponse] = Field(..., description="Closed ports found")
    filtered_ports: List[PortInfoResponse] = Field(..., description="Filtered ports found")
    vulnerabilities: List[VulnerabilityInfoResponse] = Field(..., description="Vulnerabilities found")
    total_vulnerabilities: int = Field(..., description="Total number of vulnerabilities")
    critical_vulnerabilities: int = Field(..., description="Number of critical vulnerabilities")
    high_vulnerabilities: int = Field(..., description="Number of high vulnerabilities")
    medium_vulnerabilities: int = Field(..., description="Number of medium vulnerabilities")
    low_vulnerabilities: int = Field(..., description="Number of low vulnerabilities")
    services_found: List[str] = Field(..., description="Services discovered")
    unique_services: int = Field(..., description="Number of unique services")
    hostname: Optional[str] = Field(None, description="Resolved hostname")
    ip_addresses: List[str] = Field(..., description="IP addresses")
    mac_address: Optional[str] = Field(None, description="MAC address")
    os_detection: Optional[str] = Field(None, description="Operating system detection")
    ssl_certificate: Optional[Dict[str, Any]] = Field(None, description="SSL certificate information")
    ssl_issues: List[str] = Field(..., description="SSL/TLS issues found")
    dns_records: Dict[str, List[str]] = Field(..., description="DNS records")
    reverse_dns: Optional[str] = Field(None, description="Reverse DNS lookup result")
    total_ports_scanned: int = Field(..., description="Total ports scanned")
    ports_per_second: Optional[float] = Field(None, description="Scan speed in ports per second")
    scan_config: Dict[str, Any] = Field(..., description="Scan configuration used")
    errors: List[str] = Field(..., description="Errors encountered during scan")
    warnings: List[str] = Field(..., description="Warnings during scan")
    scanner_version: Optional[str] = Field(None, description="Scanner version used")
    scan_profile: Optional[str] = Field(None, description="Scan profile used")
    tags: List[str] = Field(..., description="Scan tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    is_completed: bool = Field(..., description="Whether scan is completed")
    is_successful: bool = Field(..., description="Whether scan completed successfully")
    risk_score: float = Field(..., description="Overall risk score")
    security_summary: Dict[str, Any] = Field(..., description="Security summary")

class ScanListResponse(BaseModel):
    """Response schema for scan list."""
    scans: List[ScanResultResponse] = Field(..., description="List of scans")
    total_count: int = Field(..., description="Total number of scans")
    filtered_count: int = Field(..., description="Number of scans after filtering")
    page: Optional[int] = Field(None, description="Current page number")
    page_size: Optional[int] = Field(None, description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")

class ScanStatsResponse(BaseModel):
    """Response schema for scan statistics."""
    total_scans: int = Field(..., description="Total number of scans")
    pending_scans: int = Field(..., description="Number of pending scans")
    running_scans: int = Field(..., description="Number of running scans")
    completed_scans: int = Field(..., description="Number of completed scans")
    failed_scans: int = Field(..., description="Number of failed scans")
    cancelled_scans: int = Field(..., description="Number of cancelled scans")
    scans_by_type: Dict[str, int] = Field(..., description="Scans count by type")
    scans_by_status: Dict[str, int] = Field(..., description="Scans count by status")
    average_duration: Optional[float] = Field(None, description="Average scan duration")
    total_vulnerabilities_found: int = Field(..., description="Total vulnerabilities found")
    recent_scans: List[ScanResultResponse] = Field(..., description="Recent scans")

class ScanCreateResponse(BaseModel):
    """Response schema for scan creation."""
    id: str = Field(..., description="Created scan identifier")
    message: str = Field(..., description="Success message")
    status: ScanStatus = Field(..., description="Initial scan status")
    estimated_duration: Optional[float] = Field(None, description="Estimated scan duration")
    created_at: datetime = Field(..., description="Creation timestamp")

class ScanUpdateResponse(BaseModel):
    """Response schema for scan update."""
    id: str = Field(..., description="Updated scan identifier")
    message: str = Field(..., description="Success message")
    updated_at: datetime = Field(..., description="Update timestamp")
    changes: Dict[str, Any] = Field(..., description="Changes made")

class ScanCancelResponse(BaseModel):
    """Response schema for scan cancellation."""
    id: str = Field(..., description="Cancelled scan identifier")
    message: str = Field(..., description="Success message")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")

# Error Schemas
class ScanErrorResponse(BaseModel):
    """Error response schema for scan operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Bulk Operation Schemas
class BulkScanRequest(BaseModel):
    """Request schema for bulk scan operations."""
    operation: str = Field(..., description="Bulk operation type")
    scan_ids: List[str] = Field(..., description="List of scan IDs")
    updates: Optional[Dict[str, Any]] = Field(None, description="Updates to apply")

class BulkScanResponse(BaseModel):
    """Response schema for bulk scan operations."""
    operation: str = Field(..., description="Bulk operation type")
    total_scans: int = Field(..., description="Total number of scans")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Operation errors")
    completed_at: datetime = Field(..., description="Completion timestamp")

# Export Schemas
class ScanExportRequest(BaseModel):
    """Request schema for scan export."""
    format: str = Field(..., description="Export format (json, csv, xml)")
    filters: Optional[ScanFilterRequest] = Field(None, description="Export filters")
    include_details: bool = Field(default=True, description="Include detailed scan results")
    include_vulnerabilities: bool = Field(default=True, description="Include vulnerability details")
    include_services: bool = Field(default=True, description="Include service details")

class ScanExportResponse(BaseModel):
    """Response schema for scan export."""
    download_url: str = Field(..., description="Download URL for exported file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Download link expiration")
    scan_count: int = Field(..., description="Number of scans exported") 