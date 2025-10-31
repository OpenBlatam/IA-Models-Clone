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
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Key message API schemas for cybersecurity tools.
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

# Request Schemas
class CreateKeyMessageRequest(BaseModel):
    """Request schema for creating a key message."""
    title: str = Field(..., min_length=1, max_length=200, description="Message title")
    content: str = Field(..., min_length=1, description="Message content")
    summary: Optional[str] = Field(None, max_length=500, description="Message summary")
    priority: MessagePriority = Field(default=MessagePriority.MEDIUM, description="Message priority")
    category: MessageCategory = Field(..., description="Message category")
    tags: List[str] = Field(default_factory=list, description="Message tags")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    target_ip: Optional[str] = Field(None, description="Target IP address")
    port: Optional[int] = Field(None, ge=1, le=65535, description="Port number")
    protocol: Optional[str] = Field(None, description="Network protocol")
    threat_level: Optional[str] = Field(None, description="Threat level assessment")
    cve_ids: List[str] = Field(default_factory=list, description="Related CVE identifiers")
    ioc_indicators: List[str] = Field(default_factory=list, description="Indicators of compromise")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Risk score")
    references: List[str] = Field(default_factory=list, description="Reference links")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
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

class UpdateKeyMessageRequest(BaseModel):
    """Request schema for updating a key message."""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Message title")
    content: Optional[str] = Field(None, min_length=1, description="Message content")
    summary: Optional[str] = Field(None, max_length=500, description="Message summary")
    priority: Optional[MessagePriority] = Field(None, description="Message priority")
    category: Optional[MessageCategory] = Field(None, description="Message category")
    tags: Optional[List[str]] = Field(None, description="Message tags")
    status: Optional[MessageStatus] = Field(None, description="Message status")
    threat_level: Optional[str] = Field(None, description="Threat level assessment")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Risk score")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    assigned_to: Optional[str] = Field(None, description="Assignee identifier")
    review_notes: Optional[str] = Field(None, description="Review notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class KeyMessageFilterRequest(BaseModel):
    """Request schema for filtering key messages."""
    priority: Optional[MessagePriority] = Field(None, description="Filter by priority")
    category: Optional[MessageCategory] = Field(None, description="Filter by category")
    status: Optional[MessageStatus] = Field(None, description="Filter by status")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    source_ip: Optional[str] = Field(None, description="Filter by source IP")
    target_ip: Optional[str] = Field(None, description="Filter by target IP")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date (after)")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date (before)")
    assigned_to: Optional[str] = Field(None, description="Filter by assignee")
    search: Optional[str] = Field(None, description="Search in title and content")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")

# Response Schemas
class KeyMessageResponse(BaseModel):
    """Response schema for key message data."""
    id: str = Field(..., description="Unique message identifier")
    title: str = Field(..., description="Message title")
    content: str = Field(..., description="Message content")
    summary: Optional[str] = Field(None, description="Message summary")
    priority: MessagePriority = Field(..., description="Message priority")
    category: MessageCategory = Field(..., description="Message category")
    tags: List[str] = Field(..., description="Message tags")
    status: MessageStatus = Field(..., description="Message status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    target_ip: Optional[str] = Field(None, description="Target IP address")
    port: Optional[int] = Field(None, description="Port number")
    protocol: Optional[str] = Field(None, description="Network protocol")
    threat_level: Optional[str] = Field(None, description="Threat level assessment")
    cve_ids: List[str] = Field(..., description="Related CVE identifiers")
    ioc_indicators: List[str] = Field(..., description="Indicators of compromise")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    risk_score: Optional[float] = Field(None, description="Risk score")
    false_positive_probability: Optional[float] = Field(None, description="False positive probability")
    references: List[str] = Field(..., description="Reference links")
    related_messages: List[str] = Field(..., description="Related message IDs")
    created_by: Optional[str] = Field(None, description="Creator identifier")
    assigned_to: Optional[str] = Field(None, description="Assignee identifier")
    reviewed_by: Optional[str] = Field(None, description="Reviewer identifier")
    review_notes: Optional[str] = Field(None, description="Review notes")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    is_expired: bool = Field(..., description="Whether message has expired")
    is_high_priority: bool = Field(..., description="Whether message is high priority")
    is_urgent: bool = Field(..., description="Whether message requires immediate attention")
    risk_level: str = Field(..., description="Risk level")

class KeyMessageListResponse(BaseModel):
    """Response schema for key message list."""
    messages: List[KeyMessageResponse] = Field(..., description="List of key messages")
    total_count: int = Field(..., description="Total number of messages")
    filtered_count: int = Field(..., description="Number of messages after filtering")
    page: Optional[int] = Field(None, description="Current page number")
    page_size: Optional[int] = Field(None, description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")

class KeyMessageStatsResponse(BaseModel):
    """Response schema for key message statistics."""
    total_messages: int = Field(..., description="Total number of messages")
    pending_messages: int = Field(..., description="Number of pending messages")
    processing_messages: int = Field(..., description="Number of processing messages")
    completed_messages: int = Field(..., description="Number of completed messages")
    failed_messages: int = Field(..., description="Number of failed messages")
    critical_messages: int = Field(..., description="Number of critical messages")
    high_messages: int = Field(..., description="Number of high priority messages")
    medium_messages: int = Field(..., description="Number of medium priority messages")
    low_messages: int = Field(..., description="Number of low priority messages")
    info_messages: int = Field(..., description="Number of info messages")
    messages_by_category: Dict[str, int] = Field(..., description="Messages count by category")
    messages_by_status: Dict[str, int] = Field(..., description="Messages count by status")
    recent_messages: List[KeyMessageResponse] = Field(..., description="Recent messages")

class KeyMessageCreateResponse(BaseModel):
    """Response schema for key message creation."""
    id: str = Field(..., description="Created message identifier")
    message: str = Field(..., description="Success message")
    created_at: datetime = Field(..., description="Creation timestamp")

class KeyMessageUpdateResponse(BaseModel):
    """Response schema for key message update."""
    id: str = Field(..., description="Updated message identifier")
    message: str = Field(..., description="Success message")
    updated_at: datetime = Field(..., description="Update timestamp")
    changes: Dict[str, Any] = Field(..., description="Changes made")

class KeyMessageDeleteResponse(BaseModel):
    """Response schema for key message deletion."""
    id: str = Field(..., description="Deleted message identifier")
    message: str = Field(..., description="Success message")
    deleted_at: datetime = Field(..., description="Deletion timestamp")

# Error Schemas
class KeyMessageErrorResponse(BaseModel):
    """Error response schema for key message operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""
    error: str = Field(..., description="Validation error message")
    field_errors: Dict[str, List[str]] = Field(..., description="Field-specific errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Bulk Operation Schemas
class BulkKeyMessageRequest(BaseModel):
    """Request schema for bulk key message operations."""
    operation: str = Field(..., description="Bulk operation type")
    message_ids: List[str] = Field(..., description="List of message IDs")
    updates: Optional[Dict[str, Any]] = Field(None, description="Updates to apply")

class BulkKeyMessageResponse(BaseModel):
    """Response schema for bulk key message operations."""
    operation: str = Field(..., description="Bulk operation type")
    total_messages: int = Field(..., description="Total number of messages")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Operation errors")
    completed_at: datetime = Field(..., description="Completion timestamp")

# Export Schemas
class KeyMessageExportRequest(BaseModel):
    """Request schema for key message export."""
    format: str = Field(..., description="Export format (json, csv, xml)")
    filters: Optional[KeyMessageFilterRequest] = Field(None, description="Export filters")
    include_metadata: bool = Field(default=True, description="Include metadata in export")
    include_related: bool = Field(default=False, description="Include related messages")

class KeyMessageExportResponse(BaseModel):
    """Response schema for key message export."""
    download_url: str = Field(..., description="Download URL for exported file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Download link expiration")
    message_count: int = Field(..., description="Number of messages exported") 