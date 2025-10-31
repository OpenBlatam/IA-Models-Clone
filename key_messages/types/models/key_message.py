from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
            import ipaddress
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Key message data model for cybersecurity tools.
"""

class MessagePriority(str, Enum):
    """Message priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MessageCategory(str, Enum):
    """Message categories for cybersecurity."""
    VULNERABILITY = "vulnerability"
    ATTACK = "attack"
    SCAN_RESULT = "scan_result"
    THREAT_INTEL = "threat_intel"
    INCIDENT = "incident"
    ALERT = "alert"
    REPORT = "report"

class MessageStatus(str, Enum):
    """Message status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class KeyMessageModel(BaseModel):
    """Key message data model."""
    
    # Core fields
    id: str = Field(..., description="Unique message identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Message title")
    content: str = Field(..., min_length=1, description="Message content")
    summary: Optional[str] = Field(None, max_length=500, description="Message summary")
    
    # Classification
    priority: MessagePriority = Field(default=MessagePriority.MEDIUM, description="Message priority")
    category: MessageCategory = Field(..., description="Message category")
    tags: List[str] = Field(default_factory=list, description="Message tags")
    
    # Metadata
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Message status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    
    # Security context
    source_ip: Optional[str] = Field(None, description="Source IP address")
    target_ip: Optional[str] = Field(None, description="Target IP address")
    port: Optional[int] = Field(None, ge=1, le=65535, description="Port number")
    protocol: Optional[str] = Field(None, description="Network protocol")
    
    # Threat intelligence
    threat_level: Optional[str] = Field(None, description="Threat level assessment")
    cve_ids: List[str] = Field(default_factory=list, description="Related CVE identifiers")
    ioc_indicators: List[str] = Field(default_factory=list, description="Indicators of compromise")
    
    # Analysis results
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Risk score")
    false_positive_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="False positive probability")
    
    # References and links
    references: List[str] = Field(default_factory=list, description="Reference links")
    related_messages: List[str] = Field(default_factory=list, description="Related message IDs")
    
    # Custom fields
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Audit fields
    created_by: Optional[str] = Field(None, description="Creator identifier")
    assigned_to: Optional[str] = Field(None, description="Assignee identifier")
    reviewed_by: Optional[str] = Field(None, description="Reviewer identifier")
    review_notes: Optional[str] = Field(None, description="Review notes")
    
    @field_validator('title')
    def validate_title(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('content')
    def validate_content(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('source_ip', 'target_ip')
    def validate_ip_address(cls, v) -> bool:
        if v is not None:
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f"Invalid IP address format: {v}")
        return v
    
    @field_validator('cve_ids')
    def validate_cve_ids(cls, v) -> bool:
        for cve_id in v:
            if not cve_id.startswith('CVE-'):
                raise ValueError(f"Invalid CVE ID format: {cve_id}")
        return v
    
    @field_validator('tags')
    def validate_tags(cls, v) -> bool:
        # Remove duplicates and empty tags
        unique_tags = list(set(tag.strip() for tag in v if tag.strip()))
        return unique_tags
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_high_priority(self) -> bool:
        """Check if the message is high priority."""
        return self.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]
    
    def is_urgent(self) -> bool:
        """Check if the message requires immediate attention."""
        return (
            self.is_high_priority() and 
            self.status == MessageStatus.PENDING and
            not self.is_expired()
        )
    
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
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the message."""
        if tag.strip() and tag.strip() not in self.tags:
            self.tags.append(tag.strip())
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the message."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_status(self, status: MessageStatus, notes: Optional[str] = None) -> None:
        """Update message status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if notes:
            self.review_notes = notes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyMessageModel':
        """Create model from dictionary."""
        return cls(**data)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 