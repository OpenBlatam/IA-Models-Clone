"""
Email Campaign Models

This module contains models for email campaigns and campaign metrics.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class CampaignStatus(str, Enum):
    """Status of email campaigns"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CampaignType(str, Enum):
    """Types of email campaigns"""
    SEQUENCE = "sequence"
    BROADCAST = "broadcast"
    AUTOMATED = "automated"
    TRIGGERED = "triggered"
    A_B_TEST = "ab_test"


class CampaignMetrics(BaseModel):
    """Model for campaign performance metrics"""
    campaign_id: UUID
    
    # Delivery metrics
    total_sent: int = 0
    total_delivered: int = 0
    total_bounced: int = 0
    total_failed: int = 0
    
    # Engagement metrics
    total_opened: int = 0
    total_clicked: int = 0
    total_unsubscribed: int = 0
    total_spam_complaints: int = 0
    
    # Conversion metrics
    total_conversions: int = 0
    conversion_value: float = 0.0
    
    # Calculated rates
    delivery_rate: float = 0.0
    open_rate: float = 0.0
    click_rate: float = 0.0
    unsubscribe_rate: float = 0.0
    spam_rate: float = 0.0
    conversion_rate: float = 0.0
    
    # Timing metrics
    avg_time_to_open: Optional[float] = None  # in minutes
    avg_time_to_click: Optional[float] = None  # in minutes
    avg_time_to_convert: Optional[float] = None  # in minutes
    
    # Device and location metrics
    device_breakdown: Dict[str, int] = Field(default_factory=dict)
    location_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def calculate_rates(self) -> None:
        """Calculate all rates based on current metrics"""
        if self.total_sent > 0:
            self.delivery_rate = (self.total_delivered / self.total_sent) * 100
            self.open_rate = (self.total_opened / self.total_delivered) * 100 if self.total_delivered > 0 else 0
            self.click_rate = (self.total_clicked / self.total_delivered) * 100 if self.total_delivered > 0 else 0
            self.unsubscribe_rate = (self.total_unsubscribed / self.total_delivered) * 100 if self.total_delivered > 0 else 0
            self.spam_rate = (self.total_spam_complaints / self.total_delivered) * 100 if self.total_delivered > 0 else 0
            self.conversion_rate = (self.total_conversions / self.total_delivered) * 100 if self.total_delivered > 0 else 0
    
    def add_delivery(self, delivered: bool = True, bounced: bool = False, failed: bool = False) -> None:
        """Add delivery metrics"""
        self.total_sent += 1
        if delivered:
            self.total_delivered += 1
        if bounced:
            self.total_bounced += 1
        if failed:
            self.total_failed += 1
        self.calculate_rates()
        self.last_updated = datetime.utcnow()
    
    def add_engagement(self, opened: bool = False, clicked: bool = False, 
                      unsubscribed: bool = False, spam: bool = False) -> None:
        """Add engagement metrics"""
        if opened:
            self.total_opened += 1
        if clicked:
            self.total_clicked += 1
        if unsubscribed:
            self.total_unsubscribed += 1
        if spam:
            self.total_spam_complaints += 1
        self.calculate_rates()
        self.last_updated = datetime.utcnow()
    
    def add_conversion(self, value: float = 0.0) -> None:
        """Add conversion metrics"""
        self.total_conversions += 1
        self.conversion_value += value
        self.calculate_rates()
        self.last_updated = datetime.utcnow()


class EmailCampaign(BaseModel):
    """Model for email campaigns"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    campaign_type: CampaignType = CampaignType.SEQUENCE
    status: CampaignStatus = CampaignStatus.DRAFT
    
    # Campaign configuration
    sequence_id: Optional[UUID] = None  # For sequence-based campaigns
    target_segments: List[UUID] = Field(default_factory=list)
    target_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    # Scheduling
    scheduled_time: Optional[datetime] = None
    timezone: str = "UTC"
    send_immediately: bool = False
    
    # A/B Testing
    ab_test_enabled: bool = False
    ab_test_variants: List[Dict[str, Any]] = Field(default_factory=list)
    ab_test_winner: Optional[str] = None
    ab_test_end_time: Optional[datetime] = None
    
    # Content
    subject_line: Optional[str] = None
    preview_text: Optional[str] = None
    template_id: Optional[UUID] = None
    custom_content: Optional[Dict[str, Any]] = None
    
    # Targeting and personalization
    personalization_enabled: bool = True
    personalization_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Delivery settings
    batch_size: int = 100
    delay_between_batches: float = 1.0  # seconds
    max_retries: int = 3
    retry_delay: float = 300.0  # seconds
    
    # Tracking and analytics
    tracking_enabled: bool = True
    conversion_tracking: bool = True
    conversion_goals: List[str] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    
    # Statistics
    estimated_recipients: int = 0
    actual_recipients: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    launched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    metrics: Optional[CampaignMetrics] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Campaign name cannot be empty")
        return v.strip()
    
    @validator('scheduled_time')
    def validate_scheduled_time(cls, v):
        if v is not None and v < datetime.utcnow():
            raise ValueError("Scheduled time cannot be in the past")
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 10000:
            raise ValueError("Batch size must be between 1 and 10000")
        return v
    
    @validator('delay_between_batches')
    def validate_delay(cls, v):
        if v < 0 or v > 3600:
            raise ValueError("Delay between batches must be between 0 and 3600 seconds")
        return v
    
    def initialize_metrics(self) -> None:
        """Initialize campaign metrics"""
        if not self.metrics:
            self.metrics = CampaignMetrics(campaign_id=self.id)
    
    def add_target_segment(self, segment_id: UUID) -> None:
        """Add target segment"""
        if segment_id not in self.target_segments:
            self.target_segments.append(segment_id)
            self.updated_at = datetime.utcnow()
    
    def remove_target_segment(self, segment_id: UUID) -> bool:
        """Remove target segment"""
        if segment_id in self.target_segments:
            self.target_segments.remove(segment_id)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def add_ab_test_variant(self, variant: Dict[str, Any]) -> None:
        """Add A/B test variant"""
        if not self.ab_test_enabled:
            self.ab_test_enabled = True
        
        self.ab_test_variants.append(variant)
        self.updated_at = datetime.utcnow()
    
    def set_ab_test_winner(self, winner_variant: str) -> None:
        """Set A/B test winner"""
        self.ab_test_winner = winner_variant
        self.ab_test_end_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def schedule(self, scheduled_time: datetime) -> None:
        """Schedule the campaign"""
        if self.status != CampaignStatus.DRAFT:
            raise ValueError("Only draft campaigns can be scheduled")
        
        self.scheduled_time = scheduled_time
        self.status = CampaignStatus.SCHEDULED
        self.updated_at = datetime.utcnow()
    
    def launch(self) -> None:
        """Launch the campaign"""
        if self.status not in [CampaignStatus.DRAFT, CampaignStatus.SCHEDULED]:
            raise ValueError("Campaign must be in draft or scheduled status to launch")
        
        self.status = CampaignStatus.RUNNING
        self.launched_at = datetime.utcnow()
        self.initialize_metrics()
        self.updated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the campaign"""
        if self.status != CampaignStatus.RUNNING:
            raise ValueError("Only running campaigns can be paused")
        
        self.status = CampaignStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume(self) -> None:
        """Resume the campaign"""
        if self.status != CampaignStatus.PAUSED:
            raise ValueError("Only paused campaigns can be resumed")
        
        self.status = CampaignStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark campaign as completed"""
        if self.status not in [CampaignStatus.RUNNING, CampaignStatus.PAUSED]:
            raise ValueError("Campaign must be running or paused to complete")
        
        self.status = CampaignStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def cancel(self) -> None:
        """Cancel the campaign"""
        if self.status in [CampaignStatus.COMPLETED, CampaignStatus.CANCELLED]:
            raise ValueError("Campaign is already completed or cancelled")
        
        self.status = CampaignStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def fail(self, error_message: str) -> None:
        """Mark campaign as failed"""
        self.status = CampaignStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Store error message in custom content
        if not self.custom_content:
            self.custom_content = {}
        self.custom_content["error_message"] = error_message
    
    def is_ready_to_launch(self) -> bool:
        """Check if campaign is ready to launch"""
        if self.status != CampaignStatus.DRAFT:
            return False
        
        if not self.target_segments and not self.target_criteria:
            return False
        
        if not self.sequence_id and not self.template_id:
            return False
        
        if self.scheduled_time and self.scheduled_time > datetime.utcnow():
            return False
        
        return True
    
    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Get estimated completion time"""
        if not self.launched_at or not self.estimated_recipients:
            return None
        
        # Calculate based on batch size and delay
        total_batches = (self.estimated_recipients + self.batch_size - 1) // self.batch_size
        total_delay = (total_batches - 1) * self.delay_between_batches
        
        return self.launched_at + timedelta(seconds=total_delay)
    
    def get_progress_percentage(self) -> float:
        """Get campaign progress percentage"""
        if not self.metrics or self.estimated_recipients == 0:
            return 0.0
        
        return (self.metrics.total_sent / self.estimated_recipients) * 100
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get campaign performance summary"""
        if not self.metrics:
            return {}
        
        return {
            "campaign_id": str(self.id),
            "status": self.status.value,
            "progress_percentage": self.get_progress_percentage(),
            "delivery_rate": self.metrics.delivery_rate,
            "open_rate": self.metrics.open_rate,
            "click_rate": self.metrics.click_rate,
            "conversion_rate": self.metrics.conversion_rate,
            "total_sent": self.metrics.total_sent,
            "total_delivered": self.metrics.total_delivered,
            "total_opened": self.metrics.total_opened,
            "total_clicked": self.metrics.total_clicked,
            "total_conversions": self.metrics.total_conversions,
            "conversion_value": self.metrics.conversion_value,
            "estimated_completion": self.get_estimated_completion_time().isoformat() if self.get_estimated_completion_time() else None
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }






























