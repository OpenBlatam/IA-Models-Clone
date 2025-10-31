"""
Pydantic v2 Schemas for Email Sequence API

This module contains all request and response schemas for the email sequence API,
following FastAPI best practices with proper validation and type hints.
"""

from typing import Any, List, Dict, Optional, Union, Literal
from datetime import datetime, timedelta
from uuid import UUID
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


# Enums
class SequenceStatus(str, Enum):
    """Email sequence status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class StepType(str, Enum):
    """Sequence step types"""
    EMAIL = "email"
    DELAY = "delay"
    CONDITION = "condition"
    ACTION = "action"
    WEBHOOK = "webhook"


class TriggerType(str, Enum):
    """Sequence trigger types"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    EVENT_BASED = "event_based"
    CONDITIONAL = "conditional"
    SCHEDULED = "scheduled"


class TemplateStatus(str, Enum):
    """Email template status"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class SubscriberStatus(str, Enum):
    """Subscriber status"""
    ACTIVE = "active"
    UNSUBSCRIBED = "unsubscribed"
    BOUNCED = "bounced"
    COMPLAINED = "complained"


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Sequence Trigger Schema
class SequenceTriggerSchema(BaseModel):
    """Sequence trigger schema"""
    trigger_type: TriggerType
    delay_hours: Optional[int] = Field(default=0, ge=0, le=8760)  # Max 1 year
    delay_days: Optional[int] = Field(default=0, ge=0, le=365)   # Max 1 year
    event_name: Optional[str] = Field(None, max_length=100)
    conditions: Optional[Dict[str, Any]] = None
    scheduled_time: Optional[datetime] = None

    @validator('delay_hours', 'delay_days')
    def validate_delay(cls, v):
        if v is not None and v < 0:
            raise ValueError("Delay must be non-negative")
        return v


# Sequence Step Schema
class SequenceStepSchema(BaseModel):
    """Sequence step schema"""
    step_type: StepType
    order: int = Field(ge=1, le=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Email-specific fields
    template_id: Optional[UUID] = None
    subject: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = None
    
    # Delay-specific fields
    delay_hours: Optional[int] = Field(default=0, ge=0, le=8760)
    delay_days: Optional[int] = Field(default=0, ge=0, le=365)
    
    # Condition-specific fields
    condition_expression: Optional[str] = None
    condition_variables: Optional[Dict[str, Any]] = None
    
    # Action-specific fields
    action_type: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    
    # Webhook-specific fields
    webhook_url: Optional[str] = None
    webhook_method: Optional[str] = Field(default="POST", regex="^(GET|POST|PUT|DELETE|PATCH)$")
    webhook_headers: Optional[Dict[str, str]] = None
    
    # Common fields
    is_active: bool = True

    @validator('order')
    def validate_order(cls, v):
        if v < 1:
            raise ValueError("Step order must be at least 1")
        return v


# Sequence Request/Response Schemas
class SequenceCreateRequest(BaseModel):
    """Request schema for creating email sequences"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    target_audience: str = Field(..., min_length=1, max_length=500)
    goals: List[str] = Field(..., min_items=1, max_items=10)
    tone: str = Field(default="professional", max_length=50)
    
    # Sequence configuration
    steps: List[SequenceStepSchema] = Field(default_factory=list)
    triggers: List[SequenceTriggerSchema] = Field(default_factory=list)
    
    # Personalization settings
    personalization_enabled: bool = True
    personalization_variables: Optional[Dict[str, Any]] = None
    
    # A/B testing settings
    ab_testing_enabled: bool = False
    ab_test_variants: Optional[List[Dict[str, Any]]] = None
    
    # Analytics settings
    tracking_enabled: bool = True
    conversion_tracking: bool = True
    
    # Timing settings
    max_duration_days: Optional[int] = Field(None, ge=1, le=365)
    timezone: str = Field(default="UTC", max_length=50)
    
    # Metadata
    tags: List[str] = Field(default_factory=list, max_items=20)
    category: Optional[str] = Field(None, max_length=100)
    priority: int = Field(default=1, ge=1, le=10)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @validator('steps')
    def validate_steps_order(cls, v):
        if not v:
            return v
        
        orders = [step.order for step in v]
        if len(orders) != len(set(orders)):
            raise ValueError("Step orders must be unique")
        
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("Step orders must be sequential starting from 1")
        
        return v


class SequenceUpdateRequest(BaseModel):
    """Request schema for updating email sequences"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[SequenceStatus] = None
    
    # Sequence configuration
    steps: Optional[List[SequenceStepSchema]] = None
    triggers: Optional[List[SequenceTriggerSchema]] = None
    
    # Personalization settings
    personalization_enabled: Optional[bool] = None
    personalization_variables: Optional[Dict[str, Any]] = None
    
    # A/B testing settings
    ab_testing_enabled: Optional[bool] = None
    ab_test_variants: Optional[List[Dict[str, Any]]] = None
    
    # Analytics settings
    tracking_enabled: Optional[bool] = None
    conversion_tracking: Optional[bool] = None
    
    # Timing settings
    max_duration_days: Optional[int] = Field(None, ge=1, le=365)
    timezone: Optional[str] = Field(None, max_length=50)
    
    # Metadata
    tags: Optional[List[str]] = Field(None, max_items=20)
    category: Optional[str] = Field(None, max_length=100)
    priority: Optional[int] = Field(None, ge=1, le=10)


class SequenceResponse(BaseResponse):
    """Response schema for email sequences"""
    id: UUID
    name: str
    description: Optional[str]
    status: SequenceStatus
    
    # Sequence configuration
    steps: List[SequenceStepSchema]
    triggers: List[SequenceTriggerSchema]
    
    # Personalization settings
    personalization_enabled: bool
    personalization_variables: Optional[Dict[str, Any]]
    
    # A/B testing settings
    ab_testing_enabled: bool
    ab_test_variants: Optional[List[Dict[str, Any]]]
    
    # Analytics settings
    tracking_enabled: bool
    conversion_tracking: bool
    
    # Timing settings
    max_duration_days: Optional[int]
    timezone: str
    
    # Metadata
    tags: List[str]
    category: Optional[str]
    priority: int
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    activated_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Statistics
    total_subscribers: int
    active_subscribers: int
    completed_subscribers: int


# Template Schemas
class TemplateVariableSchema(BaseModel):
    """Template variable schema"""
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., regex="^(string|number|boolean|date|email)$")
    default_value: Optional[str] = None
    required: bool = False
    description: Optional[str] = Field(None, max_length=500)


class TemplateCreateRequest(BaseModel):
    """Request schema for creating email templates"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    subject: str = Field(..., min_length=1, max_length=255)
    html_content: str = Field(..., min_length=1)
    text_content: Optional[str] = None
    variables: List[TemplateVariableSchema] = Field(default_factory=list)
    category: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list, max_items=20)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class TemplateResponse(BaseResponse):
    """Response schema for email templates"""
    id: UUID
    name: str
    description: Optional[str]
    subject: str
    html_content: str
    text_content: Optional[str]
    variables: List[TemplateVariableSchema]
    status: TemplateStatus
    category: Optional[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime


# Campaign Schemas
class CampaignCreateRequest(BaseModel):
    """Request schema for creating email campaigns"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    sequence_id: UUID
    target_segments: List[str] = Field(..., min_items=1)
    scheduled_time: Optional[datetime] = None
    timezone: str = Field(default="UTC", max_length=50)
    tags: List[str] = Field(default_factory=list, max_items=20)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class CampaignResponse(BaseResponse):
    """Response schema for email campaigns"""
    id: UUID
    name: str
    description: Optional[str]
    sequence_id: UUID
    target_segments: List[str]
    scheduled_time: Optional[datetime]
    timezone: str
    status: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    launched_at: Optional[datetime]
    completed_at: Optional[datetime]


# Subscriber Schemas
class SubscriberCreateRequest(BaseModel):
    """Request schema for creating subscribers"""
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    company: Optional[str] = Field(None, max_length=200)
    job_title: Optional[str] = Field(None, max_length=200)
    custom_fields: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list, max_items=20)
    source: Optional[str] = Field(None, max_length=100)

    @validator('email')
    def validate_email(cls, v):
        return v.lower().strip()


class SubscriberResponse(BaseResponse):
    """Response schema for subscribers"""
    id: UUID
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    job_title: Optional[str]
    custom_fields: Optional[Dict[str, Any]]
    status: SubscriberStatus
    tags: List[str]
    source: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_activity_at: Optional[datetime]


# Analytics Schemas
class AnalyticsResponse(BaseResponse):
    """Response schema for analytics data"""
    sequence_id: UUID
    metrics: Dict[str, Any]
    time_range: Dict[str, datetime]
    generated_at: datetime


# Bulk Operation Schemas
class BulkSubscriberCreateRequest(BaseModel):
    """Request schema for bulk subscriber creation"""
    subscribers: List[SubscriberCreateRequest] = Field(..., min_items=1, max_items=1000)
    sequence_id: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list, max_items=20)


class BulkOperationResponse(BaseResponse):
    """Response schema for bulk operations"""
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# Search and Filter Schemas
class SequenceSearchRequest(BaseModel):
    """Request schema for searching sequences"""
    query: Optional[str] = Field(None, max_length=255)
    status: Optional[SequenceStatus] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SequenceListResponse(BaseResponse):
    """Response schema for sequence lists"""
    sequences: List[SequenceResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool


# Health Check Schema
class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    version: str
    services: Dict[str, str]






