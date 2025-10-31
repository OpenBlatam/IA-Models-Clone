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
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Attack request API schemas for cybersecurity tools.
"""

class AttackType(str, Enum):
    """Types of security attacks."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DOS = "dos"
    DDOS = "ddos"
    MAN_IN_THE_MIDDLE = "man_in_the_middle"
    PHISHING = "phishing"
    SOCIAL_ENGINEERING = "social_engineering"
    EXPLOIT = "exploit"

class AttackStatus(str, Enum):
    """Attack status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class AttackPhase(str, Enum):
    """Attack phases."""
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    GAINING_ACCESS = "gaining_access"
    MAINTAINING_ACCESS = "maintaining_access"
    COVERING_TRACKS = "covering_tracks"

# Request Schemas
class CreateAttackRequest(BaseModel):
    """Request schema for creating an attack."""
    attack_type: AttackType = Field(..., description="Type of attack to perform")
    target: str = Field(..., description="Target hostname, IP, or URL")
    payload: Optional[str] = Field(None, description="Attack payload")
    payload_type: Optional[str] = Field(None, description="Payload type")
    credentials: Optional[Dict[str, str]] = Field(None, description="Credentials to use")
    wordlist: Optional[List[str]] = Field(None, description="Wordlist for brute force")
    timeout: Optional[float] = Field(None, ge=1.0, le=3600.0, description="Attack timeout in seconds")
    max_attempts: Optional[int] = Field(None, ge=1, le=10000, description="Maximum attack attempts")
    threads: Optional[int] = Field(None, ge=1, le=100, description="Number of threads")
    delay: Optional[float] = Field(None, ge=0.0, description="Delay between attempts")
    exploit_module: Optional[str] = Field(None, description="Exploit module to use")
    exploit_options: Dict[str, Any] = Field(default_factory=dict, description="Exploit options")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    user_agent: Optional[str] = Field(None, description="Custom user agent")
    proxy: Optional[str] = Field(None, description="Proxy configuration")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    attack_profile: Optional[str] = Field(None, description="Attack profile to use")
    custom_options: Dict[str, Any] = Field(default_factory=dict, description="Custom attack options")
    priority: Optional[str] = Field(None, description="Attack priority")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled attack time")
    
    @field_validator('target')
    def validate_target(cls, v) -> Optional[Dict[str, Any]]:
        if not v.strip():
            raise ValueError("Target cannot be empty")
        return v.strip()
    
    @field_validator('max_attempts')
    def validate_max_attempts(cls, v) -> bool:
        if v is not None and v <= 0:
            raise ValueError("Max attempts must be positive")
        return v
    
    @field_validator('threads')
    def validate_threads(cls, v) -> bool:
        if v is not None and v <= 0:
            raise ValueError("Threads must be positive")
        return v

class UpdateAttackRequest(BaseModel):
    """Request schema for updating an attack."""
    status: Optional[AttackStatus] = Field(None, description="Attack status")
    phase: Optional[AttackPhase] = Field(None, description="Attack phase")
    priority: Optional[str] = Field(None, description="Attack priority")
    timeout: Optional[float] = Field(None, ge=1.0, le=3600.0, description="Attack timeout")
    custom_options: Optional[Dict[str, Any]] = Field(None, description="Custom attack options")
    notes: Optional[str] = Field(None, description="Attack notes")

class AttackFilterRequest(BaseModel):
    """Request schema for filtering attacks."""
    attack_type: Optional[AttackType] = Field(None, description="Filter by attack type")
    status: Optional[AttackStatus] = Field(None, description="Filter by status")
    phase: Optional[AttackPhase] = Field(None, description="Filter by phase")
    target: Optional[str] = Field(None, description="Filter by target")
    started_after: Optional[datetime] = Field(None, description="Filter by start date (after)")
    started_before: Optional[datetime] = Field(None, description="Filter by start date (before)")
    completed_after: Optional[datetime] = Field(None, description="Filter by completion date (after)")
    completed_before: Optional[datetime] = Field(None, description="Filter by completion date (before)")
    priority: Optional[str] = Field(None, description="Filter by priority")
    attack_profile: Optional[str] = Field(None, description="Filter by attack profile")
    is_successful: Optional[bool] = Field(None, description="Filter by success status")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")

# Response Schemas
class PayloadInfoResponse(BaseModel):
    """Response schema for payload information."""
    type: str = Field(..., description="Payload type")
    content: str = Field(..., description="Payload content")
    encoding: Optional[str] = Field(None, description="Payload encoding")
    size: Optional[int] = Field(None, description="Payload size in bytes")
    checksum: Optional[str] = Field(None, description="Payload checksum")

class ExploitInfoResponse(BaseModel):
    """Response schema for exploit information."""
    name: str = Field(..., description="Exploit name")
    description: str = Field(..., description="Exploit description")
    cve_id: Optional[str] = Field(None, description="Related CVE ID")
    module: Optional[str] = Field(None, description="Exploit module used")
    options: Dict[str, Any] = Field(..., description="Exploit options")

class CredentialInfoResponse(BaseModel):
    """Response schema for credential information."""
    username: str = Field(..., description="Username")
    password: Optional[str] = Field(None, description="Password")
    hash: Optional[str] = Field(None, description="Password hash")
    salt: Optional[str] = Field(None, description="Password salt")
    is_valid: bool = Field(..., description="Whether credential is valid")

class AttackResultResponse(BaseModel):
    """Response schema for attack result data."""
    id: str = Field(..., description="Unique attack result identifier")
    attack_type: AttackType = Field(..., description="Type of attack performed")
    target: str = Field(..., description="Target hostname, IP, or URL")
    status: AttackStatus = Field(..., description="Attack status")
    phase: AttackPhase = Field(..., description="Current attack phase")
    started_at: datetime = Field(..., description="Attack start time")
    completed_at: Optional[datetime] = Field(None, description="Attack completion time")
    duration: Optional[float] = Field(None, description="Attack duration in seconds")
    is_successful: bool = Field(..., description="Whether attack was successful")
    success_rate: Optional[float] = Field(None, description="Attack success rate")
    attempts_made: int = Field(..., description="Number of attempts made")
    successful_attempts: int = Field(..., description="Number of successful attempts")
    payloads_used: List[PayloadInfoResponse] = Field(..., description="Payloads used in attack")
    exploits_used: List[ExploitInfoResponse] = Field(..., description="Exploits used in attack")
    credentials_found: List[CredentialInfoResponse] = Field(..., description="Credentials discovered")
    valid_credentials: int = Field(..., description="Number of valid credentials found")
    access_gained: bool = Field(..., description="Whether access was gained")
    access_level: Optional[str] = Field(None, description="Level of access gained")
    shell_access: bool = Field(..., description="Whether shell access was obtained")
    web_access: bool = Field(..., description="Whether web access was obtained")
    database_access: bool = Field(..., description="Whether database access was obtained")
    data_exfiltrated: bool = Field(..., description="Whether data was exfiltrated")
    data_size: Optional[int] = Field(None, description="Size of exfiltrated data in bytes")
    data_types: List[str] = Field(..., description="Types of data exfiltrated")
    persistence_established: bool = Field(..., description="Whether persistence was established")
    backdoor_installed: bool = Field(..., description="Whether backdoor was installed")
    rootkit_installed: bool = Field(..., description="Whether rootkit was installed")
    was_detected: bool = Field(..., description="Whether attack was detected")
    detection_time: Optional[float] = Field(None, description="Time to detection in seconds")
    response_triggered: bool = Field(..., description="Whether response was triggered")
    response_type: Optional[str] = Field(None, description="Type of response triggered")
    packets_sent: int = Field(..., description="Number of packets sent")
    packets_received: int = Field(..., description="Number of packets received")
    bandwidth_used: Optional[float] = Field(None, description="Bandwidth used in MB")
    errors: List[str] = Field(..., description="Errors encountered during attack")
    warnings: List[str] = Field(..., description="Warnings during attack")
    exceptions: List[str] = Field(..., description="Exceptions raised during attack")
    evidence_collected: List[str] = Field(..., description="Evidence collected")
    artifacts_created: List[str] = Field(..., description="Artifacts created")
    logs_generated: List[str] = Field(..., description="Logs generated")
    attack_config: Dict[str, Any] = Field(..., description="Attack configuration used")
    target_config: Dict[str, Any] = Field(..., description="Target configuration")
    risk_level: Optional[str] = Field(None, description="Risk level of the attack")
    impact_score: Optional[float] = Field(None, description="Impact score")
    complexity_score: Optional[float] = Field(None, description="Complexity score")
    attacker_ip: Optional[str] = Field(None, description="Attacker IP address")
    tool_used: Optional[str] = Field(None, description="Tool used for attack")
    version: Optional[str] = Field(None, description="Tool version")
    tags: List[str] = Field(..., description="Attack tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    is_completed: bool = Field(..., description="Whether attack is completed")
    risk_score: float = Field(..., description="Overall risk score")

class AttackListResponse(BaseModel):
    """Response schema for attack list."""
    attacks: List[AttackResultResponse] = Field(..., description="List of attacks")
    total_count: int = Field(..., description="Total number of attacks")
    filtered_count: int = Field(..., description="Number of attacks after filtering")
    page: Optional[int] = Field(None, description="Current page number")
    page_size: Optional[int] = Field(None, description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")

class AttackStatsResponse(BaseModel):
    """Response schema for attack statistics."""
    total_attacks: int = Field(..., description="Total number of attacks")
    pending_attacks: int = Field(..., description="Number of pending attacks")
    running_attacks: int = Field(..., description="Number of running attacks")
    successful_attacks: int = Field(..., description="Number of successful attacks")
    failed_attacks: int = Field(..., description="Number of failed attacks")
    cancelled_attacks: int = Field(..., description="Number of cancelled attacks")
    attacks_by_type: Dict[str, int] = Field(..., description="Attacks count by type")
    attacks_by_status: Dict[str, int] = Field(..., description="Attacks count by status")
    attacks_by_phase: Dict[str, int] = Field(..., description="Attacks count by phase")
    average_duration: Optional[float] = Field(None, description="Average attack duration")
    success_rate: float = Field(..., description="Overall success rate")
    total_credentials_found: int = Field(..., description="Total credentials found")
    total_access_gained: int = Field(..., description="Total access gained")
    recent_attacks: List[AttackResultResponse] = Field(..., description="Recent attacks")

class AttackCreateResponse(BaseModel):
    """Response schema for attack creation."""
    id: str = Field(..., description="Created attack identifier")
    message: str = Field(..., description="Success message")
    status: AttackStatus = Field(..., description="Initial attack status")
    phase: AttackPhase = Field(..., description="Initial attack phase")
    estimated_duration: Optional[float] = Field(None, description="Estimated attack duration")
    created_at: datetime = Field(..., description="Creation timestamp")

class AttackUpdateResponse(BaseModel):
    """Response schema for attack update."""
    id: str = Field(..., description="Updated attack identifier")
    message: str = Field(..., description="Success message")
    updated_at: datetime = Field(..., description="Update timestamp")
    changes: Dict[str, Any] = Field(..., description="Changes made")

class AttackCancelResponse(BaseModel):
    """Response schema for attack cancellation."""
    id: str = Field(..., description="Cancelled attack identifier")
    message: str = Field(..., description="Success message")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")

# Error Schemas
class AttackErrorResponse(BaseModel):
    """Error response schema for attack operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Bulk Operation Schemas
class BulkAttackRequest(BaseModel):
    """Request schema for bulk attack operations."""
    operation: str = Field(..., description="Bulk operation type")
    attack_ids: List[str] = Field(..., description="List of attack IDs")
    updates: Optional[Dict[str, Any]] = Field(None, description="Updates to apply")

class BulkAttackResponse(BaseModel):
    """Response schema for bulk attack operations."""
    operation: str = Field(..., description="Bulk operation type")
    total_attacks: int = Field(..., description="Total number of attacks")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Operation errors")
    completed_at: datetime = Field(..., description="Completion timestamp")

# Export Schemas
class AttackExportRequest(BaseModel):
    """Request schema for attack export."""
    format: str = Field(..., description="Export format (json, csv, xml)")
    filters: Optional[AttackFilterRequest] = Field(None, description="Export filters")
    include_details: bool = Field(default=True, description="Include detailed attack results")
    include_payloads: bool = Field(default=True, description="Include payload details")
    include_credentials: bool = Field(default=True, description="Include credential details")

class AttackExportResponse(BaseModel):
    """Response schema for attack export."""
    download_url: str = Field(..., description="Download URL for exported file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Download link expiration")
    attack_count: int = Field(..., description="Number of attacks exported") 