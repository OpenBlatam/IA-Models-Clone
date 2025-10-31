from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
from uuid import UUID, uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Subscriber Models

This module contains models for email subscribers and subscriber segments.
"""



class SubscriberStatus(str, Enum):
    """Status of email subscribers"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNSUBSCRIBED = "unsubscribed"
    BOUNCED = "bounced"
    SPAM = "spam"


class SubscriptionSource(str, Enum):
    """Source of subscription"""
    WEBSITE = "website"
    LANDING_PAGE = "landing_page"
    SOCIAL_MEDIA = "social_media"
    REFERRAL = "referral"
    MANUAL = "manual"
    API = "api"
    IMPORT = "import"


class SubscriberSegment(BaseModel):
    """Model for subscriber segments"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    
    # Segment criteria
    criteria: Dict[str, Any] = Field(default_factory=dict)
    
    # Segment settings
    is_active: bool = True
    is_dynamic: bool = True  # Dynamic segments update automatically
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    
    # Statistics
    subscriber_count: int = 0
    last_updated: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('name')
    def validate_name(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Segment name cannot be empty")
        return v.strip()
    
    def add_criteria(self, field: str, operator: str, value: Any) -> None:
        """Add criteria to the segment"""
        self.criteria[field] = {
            "operator": operator,
            "value": value
        }
        self.updated_at = datetime.utcnow()
    
    def remove_criteria(self, field: str) -> bool:
        """Remove criteria from the segment"""
        if field in self.criteria:
            del self.criteria[field]
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def update_subscriber_count(self, count: int) -> None:
        """Update subscriber count"""
        self.subscriber_count = count
        self.last_updated = datetime.utcnow()
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class Subscriber(BaseModel):
    """Model for email subscribers"""
    id: UUID = Field(default_factory=uuid4)
    email: EmailStr = Field(..., description="Subscriber email address")
    
    # Personal information
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = None
    company: Optional[str] = Field(None, max_length=255)
    job_title: Optional[str] = Field(None, max_length=255)
    
    # Subscription information
    status: SubscriberStatus = SubscriberStatus.ACTIVE
    source: SubscriptionSource = SubscriptionSource.WEBSITE
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)
    unsubscribed_at: Optional[datetime] = None
    
    # Preferences
    preferences: Dict[str, Any] = Field(default_factory=dict)
    interests: List[str] = Field(default_factory=list)
    frequency: str = "weekly"  # daily, weekly, monthly
    
    # Location and demographics
    country: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    
    # Custom fields
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Segments
    segments: List[UUID] = Field(default_factory=list)
    
    # Engagement metrics
    total_emails_sent: int = 0
    total_emails_opened: int = 0
    total_emails_clicked: int = 0
    last_email_sent: Optional[datetime] = None
    last_email_opened: Optional[datetime] = None
    last_email_clicked: Optional[datetime] = None
    
    # Bounce and spam information
    bounce_count: int = 0
    spam_count: int = 0
    last_bounce: Optional[datetime] = None
    last_spam: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('first_name', 'last_name')
    def validate_names(cls, v) -> bool:
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None
    
    @validator('interests')
    def validate_interests(cls, v) -> bool:
        return [interest.strip() for interest in v if interest.strip()]
    
    @property
    def full_name(self) -> str:
        """Get full name of subscriber"""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else ""
    
    @property
    def open_rate(self) -> float:
        """Calculate email open rate"""
        if self.total_emails_sent == 0:
            return 0.0
        return (self.total_emails_opened / self.total_emails_sent) * 100
    
    @property
    def click_rate(self) -> float:
        """Calculate email click rate"""
        if self.total_emails_sent == 0:
            return 0.0
        return (self.total_emails_clicked / self.total_emails_sent) * 100
    
    @property
    def is_engaged(self) -> bool:
        """Check if subscriber is engaged (opened email in last 30 days)"""
        if not self.last_email_opened:
            return False
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        return self.last_email_opened > thirty_days_ago
    
    def add_to_segment(self, segment_id: UUID) -> None:
        """Add subscriber to a segment"""
        if segment_id not in self.segments:
            self.segments.append(segment_id)
            self.updated_at = datetime.utcnow()
    
    def remove_from_segment(self, segment_id: UUID) -> bool:
        """Remove subscriber from a segment"""
        if segment_id in self.segments:
            self.segments.remove(segment_id)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update subscriber preference"""
        self.preferences[key] = value
        self.updated_at = datetime.utcnow()
    
    def add_interest(self, interest: str) -> None:
        """Add interest to subscriber"""
        if interest.strip() and interest.strip() not in self.interests:
            self.interests.append(interest.strip())
            self.updated_at = datetime.utcnow()
    
    def remove_interest(self, interest: str) -> bool:
        """Remove interest from subscriber"""
        if interest in self.interests:
            self.interests.remove(interest)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def unsubscribe(self, reason: Optional[str] = None) -> None:
        """Unsubscribe the subscriber"""
        self.status = SubscriberStatus.UNSUBSCRIBED
        self.unsubscribed_at = datetime.utcnow()
        if reason:
            self.preferences["unsubscribe_reason"] = reason
        self.updated_at = datetime.utcnow()
    
    def resubscribe(self) -> None:
        """Resubscribe the subscriber"""
        self.status = SubscriberStatus.ACTIVE
        self.unsubscribed_at = None
        if "unsubscribe_reason" in self.preferences:
            del self.preferences["unsubscribe_reason"]
        self.updated_at = datetime.utcnow()
    
    def record_email_sent(self) -> None:
        """Record that an email was sent"""
        self.total_emails_sent += 1
        self.last_email_sent = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def record_email_opened(self) -> None:
        """Record that an email was opened"""
        self.total_emails_opened += 1
        self.last_email_opened = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def record_email_clicked(self) -> None:
        """Record that an email was clicked"""
        self.total_emails_clicked += 1
        self.last_email_clicked = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def record_bounce(self) -> None:
        """Record a bounce"""
        self.bounce_count += 1
        self.last_bounce = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Auto-unsubscribe after multiple bounces
        if self.bounce_count >= 3:
            self.status = SubscriberStatus.BOUNCED
    
    def record_spam(self) -> None:
        """Record a spam complaint"""
        self.spam_count += 1
        self.last_spam = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Auto-unsubscribe after spam complaint
        if self.spam_count >= 1:
            self.status = SubscriberStatus.SPAM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subscriber to dictionary for personalization"""
        return {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "company": self.company,
            "job_title": self.job_title,
            "country": self.country,
            "city": self.city,
            "timezone": self.timezone,
            "language": self.language,
            "interests": self.interests,
            "preferences": self.preferences,
            "custom_fields": self.custom_fields,
            "open_rate": self.open_rate,
            "click_rate": self.click_rate,
            "is_engaged": self.is_engaged,
            "subscribed_at": self.subscribed_at.isoformat() if self.subscribed_at else None,
            "total_emails_sent": self.total_emails_sent,
            "total_emails_opened": self.total_emails_opened,
            "total_emails_clicked": self.total_emails_clicked
        }
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 