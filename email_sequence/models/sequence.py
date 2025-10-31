from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Email Sequence Models

This module contains the core models for email sequences, steps, and triggers.
"""



class TriggerType(str, Enum):
    """Types of sequence triggers"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    EVENT_BASED = "event_based"
    CONDITIONAL = "conditional"
    SCHEDULED = "scheduled"


class StepType(str, Enum):
    """Types of sequence steps"""
    EMAIL = "email"
    DELAY = "delay"
    CONDITION = "condition"
    ACTION = "action"
    WEBHOOK = "webhook"


class SequenceStatus(str, Enum):
    """Status of email sequences"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class SequenceTrigger(BaseModel):
    """Model for sequence triggers"""
    id: UUID = Field(default_factory=uuid4)
    trigger_type: TriggerType
    delay_hours: Optional[int] = Field(default=0, ge=0)
    delay_days: Optional[int] = Field(default=0, ge=0)
    event_name: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    scheduled_time: Optional[datetime] = None
    
    @validator('delay_hours', 'delay_days')
    def validate_delay(cls, v) -> bool:
        if v is not None and v < 0:
            raise ValueError("Delay must be non-negative")
        return v


class SequenceStep(BaseModel):
    """Model for individual sequence steps"""
    id: UUID = Field(default_factory=uuid4)
    step_type: StepType
    order: int = Field(ge=1)
    name: str
    description: Optional[str] = None
    
    # Email-specific fields
    template_id: Optional[UUID] = None
    subject: Optional[str] = None
    content: Optional[str] = None
    
    # Delay-specific fields
    delay_hours: Optional[int] = Field(default=0, ge=0)
    delay_days: Optional[int] = Field(default=0, ge=0)
    
    # Condition-specific fields
    condition_expression: Optional[str] = None
    condition_variables: Optional[Dict[str, Any]] = None
    
    # Action-specific fields
    action_type: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    
    # Webhook-specific fields
    webhook_url: Optional[str] = None
    webhook_method: Optional[str] = "POST"
    webhook_headers: Optional[Dict[str, str]] = None
    
    # Common fields
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('order')
    def validate_order(cls, v) -> bool:
        if v < 1:
            raise ValueError("Step order must be at least 1")
        return v
    
    @validator('delay_hours', 'delay_days')
    def validate_delay(cls, v) -> bool:
        if v is not None and v < 0:
            raise ValueError("Delay must be non-negative")
        return v


class EmailSequence(BaseModel):
    """Main model for email sequences"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    status: SequenceStatus = SequenceStatus.DRAFT
    
    # Sequence configuration
    steps: List[SequenceStep] = Field(default_factory=list)
    triggers: List[SequenceTrigger] = Field(default_factory=list)
    
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
    max_duration_days: Optional[int] = Field(default=None, ge=1)
    timezone: str = "UTC"
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Statistics
    total_subscribers: int = 0
    active_subscribers: int = 0
    completed_subscribers: int = 0
    
    @validator('steps')
    def validate_steps_order(cls, v) -> bool:
        """Validate that steps have unique and sequential order"""
        if not v:
            return v
        
        orders = [step.order for step in v]
        if len(orders) != len(set(orders)):
            raise ValueError("Step orders must be unique")
        
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("Step orders must be sequential starting from 1")
        
        return v
    
    @validator('name')
    def validate_name(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
    def get_total_duration(self) -> timedelta:
        """Calculate total sequence duration"""
        total_hours = 0
        total_days = 0
        
        for step in self.steps:
            if step.step_type == StepType.DELAY:
                total_hours += step.delay_hours or 0
                total_days += step.delay_days or 0
        
        return timedelta(days=total_days, hours=total_hours)
    
    def get_step_by_order(self, order: int) -> Optional[SequenceStep]:
        """Get step by order number"""
        for step in self.steps:
            if step.order == order:
                return step
        return None
    
    def add_step(self, step: SequenceStep) -> None:
        """Add a new step to the sequence"""
        # Auto-assign order if not specified
        if not step.order:
            step.order = len(self.steps) + 1
        
        self.steps.append(step)
        self.steps.sort(key=lambda x: x.order)
    
    def remove_step(self, step_id: UUID) -> bool:
        """Remove a step from the sequence"""
        initial_length = len(self.steps)
        self.steps = [step for step in self.steps if step.id != step_id]
        
        # Reorder remaining steps
        for i, step in enumerate(self.steps, 1):
            step.order = i
        
        return len(self.steps) < initial_length
    
    def activate(self) -> None:
        """Activate the sequence"""
        if self.status == SequenceStatus.DRAFT:
            self.status = SequenceStatus.ACTIVE
            self.activated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the sequence"""
        if self.status == SequenceStatus.ACTIVE:
            self.status = SequenceStatus.PAUSED
    
    def complete(self) -> None:
        """Mark sequence as completed"""
        self.status = SequenceStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the sequence"""
        self.status = SequenceStatus.ARCHIVED
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 