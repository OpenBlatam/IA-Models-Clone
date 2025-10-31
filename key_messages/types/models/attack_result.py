from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
            import ipaddress
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Attack result data model for cybersecurity tools.
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

class PayloadInfo(BaseModel):
    """Attack payload information."""
    type: str = Field(..., description="Payload type")
    content: str = Field(..., description="Payload content")
    encoding: Optional[str] = Field(None, description="Payload encoding")
    size: Optional[int] = Field(None, ge=0, description="Payload size in bytes")
    checksum: Optional[str] = Field(None, description="Payload checksum")

class ExploitInfo(BaseModel):
    """Exploit information."""
    name: str = Field(..., description="Exploit name")
    description: str = Field(..., description="Exploit description")
    cve_id: Optional[str] = Field(None, description="Related CVE ID")
    module: Optional[str] = Field(None, description="Exploit module used")
    options: Dict[str, Any] = Field(default_factory=dict, description="Exploit options")

class CredentialInfo(BaseModel):
    """Credential information."""
    username: str = Field(..., description="Username")
    password: Optional[str] = Field(None, description="Password")
    hash: Optional[str] = Field(None, description="Password hash")
    salt: Optional[str] = Field(None, description="Password salt")
    is_valid: bool = Field(default=False, description="Whether credential is valid")

class AttackResultModel(BaseModel):
    """Attack result data model."""
    
    # Core fields
    id: str = Field(..., description="Unique attack result identifier")
    attack_type: AttackType = Field(..., description="Type of attack performed")
    target: str = Field(..., description="Target hostname, IP, or URL")
    
    # Status and timing
    status: AttackStatus = Field(default=AttackStatus.PENDING, description="Attack status")
    phase: AttackPhase = Field(default=AttackPhase.RECONNAISSANCE, description="Current attack phase")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Attack start time")
    completed_at: Optional[datetime] = Field(None, description="Attack completion time")
    duration: Optional[float] = Field(None, ge=0.0, description="Attack duration in seconds")
    
    # Success metrics
    is_successful: bool = Field(default=False, description="Whether attack was successful")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Attack success rate")
    attempts_made: int = Field(default=0, description="Number of attempts made")
    successful_attempts: int = Field(default=0, description="Number of successful attempts")
    
    # Payloads and exploits
    payloads_used: List[PayloadInfo] = Field(default_factory=list, description="Payloads used in attack")
    exploits_used: List[ExploitInfo] = Field(default_factory=list, description="Exploits used in attack")
    
    # Credentials
    credentials_found: List[CredentialInfo] = Field(default_factory=list, description="Credentials discovered")
    valid_credentials: int = Field(default=0, description="Number of valid credentials found")
    
    # Access gained
    access_gained: bool = Field(default=False, description="Whether access was gained")
    access_level: Optional[str] = Field(None, description="Level of access gained")
    shell_access: bool = Field(default=False, description="Whether shell access was obtained")
    web_access: bool = Field(default=False, description="Whether web access was obtained")
    database_access: bool = Field(default=False, description="Whether database access was obtained")
    
    # Data exfiltration
    data_exfiltrated: bool = Field(default=False, description="Whether data was exfiltrated")
    data_size: Optional[int] = Field(None, ge=0, description="Size of exfiltrated data in bytes")
    data_types: List[str] = Field(default_factory=list, description="Types of data exfiltrated")
    
    # Persistence
    persistence_established: bool = Field(default=False, description="Whether persistence was established")
    backdoor_installed: bool = Field(default=False, description="Whether backdoor was installed")
    rootkit_installed: bool = Field(default=False, description="Whether rootkit was installed")
    
    # Detection and response
    was_detected: bool = Field(default=False, description="Whether attack was detected")
    detection_time: Optional[float] = Field(None, ge=0.0, description="Time to detection in seconds")
    response_triggered: bool = Field(default=False, description="Whether response was triggered")
    response_type: Optional[str] = Field(None, description="Type of response triggered")
    
    # Network activity
    packets_sent: int = Field(default=0, description="Number of packets sent")
    packets_received: int = Field(default=0, description="Number of packets received")
    bandwidth_used: Optional[float] = Field(None, ge=0.0, description="Bandwidth used in MB")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Errors encountered during attack")
    warnings: List[str] = Field(default_factory=list, description="Warnings during attack")
    exceptions: List[str] = Field(default_factory=list, description="Exceptions raised during attack")
    
    # Evidence and artifacts
    evidence_collected: List[str] = Field(default_factory=list, description="Evidence collected")
    artifacts_created: List[str] = Field(default_factory=list, description="Artifacts created")
    logs_generated: List[str] = Field(default_factory=list, description="Logs generated")
    
    # Configuration
    attack_config: Dict[str, Any] = Field(default_factory=dict, description="Attack configuration used")
    target_config: Dict[str, Any] = Field(default_factory=dict, description="Target configuration")
    
    # Risk assessment
    risk_level: Optional[str] = Field(None, description="Risk level of the attack")
    impact_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Impact score")
    complexity_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Complexity score")
    
    # Metadata
    attacker_ip: Optional[str] = Field(None, description="Attacker IP address")
    tool_used: Optional[str] = Field(None, description="Tool used for attack")
    version: Optional[str] = Field(None, description="Tool version")
    tags: List[str] = Field(default_factory=list, description="Attack tags")
    
    # Custom fields
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('target')
    def validate_target(cls, v) -> Optional[Dict[str, Any]]:
        if not v.strip():
            raise ValueError("Target cannot be empty")
        return v.strip()
    
    @field_validator('attacker_ip')
    def validate_attacker_ip(cls, v) -> bool:
        if v is not None:
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f"Invalid IP address: {v}")
        return v
    
    @field_validator('cve_id')
    def validate_cve_id(cls, v) -> bool:
        if v is not None and not v.startswith('CVE-'):
            raise ValueError(f"Invalid CVE ID format: {v}")
        return v
    
    def is_completed(self) -> bool:
        """Check if attack is completed."""
        return self.status in [AttackStatus.SUCCESSFUL, AttackStatus.FAILED, AttackStatus.CANCELLED, AttackStatus.TIMEOUT]
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.attempts_made == 0:
            return 0.0
        return self.successful_attempts / self.attempts_made
    
    def get_impact_score(self) -> float:
        """Calculate impact score based on attack results."""
        score = 0.0
        
        # Access gained
        if self.access_gained:
            score += 3.0
        if self.shell_access:
            score += 2.0
        if self.web_access:
            score += 1.0
        if self.database_access:
            score += 2.0
        
        # Data exfiltration
        if self.data_exfiltrated:
            score += 2.0
            if self.data_size and self.data_size > 1000000:  # > 1MB
                score += 1.0
        
        # Persistence
        if self.persistence_established:
            score += 2.0
        if self.backdoor_installed:
            score += 1.0
        if self.rootkit_installed:
            score += 1.0
        
        # Credentials
        score += min(self.valid_credentials * 0.5, 2.0)
        
        return min(score, 10.0)
    
    def add_credential(self, credential: CredentialInfo) -> None:
        """Add a credential to the attack result."""
        self.credentials_found.append(credential)
        if credential.is_valid:
            self.valid_credentials += 1
    
    def add_payload(self, payload: PayloadInfo) -> None:
        """Add a payload to the attack result."""
        self.payloads_used.append(payload)
    
    def add_exploit(self, exploit: ExploitInfo) -> None:
        """Add an exploit to the attack result."""
        self.exploits_used.append(exploit)
    
    def mark_successful(self) -> None:
        """Mark attack as successful."""
        self.is_successful = True
        self.status = AttackStatus.SUCCESSFUL
        self.successful_attempts += 1
    
    def complete_attack(self, duration: float) -> None:
        """Mark attack as completed."""
        self.completed_at = datetime.utcnow()
        self.duration = duration
        self.success_rate = self.get_success_rate()
        self.impact_score = self.get_impact_score()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttackResultModel':
        """Create model from dictionary."""
        return cls(**data)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 