"""
🎯 ADS Domain - Value Objects

Value objects represent immutable concepts in the domain that are defined
by their attributes rather than their identity.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID


class AdStatus(Enum):
    """Status of an advertisement."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class AdType(Enum):
    """Type of advertisement."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    CAROUSEL = "carousel"
    STORY = "story"
    REELS = "reels"
    SHOPPING = "shopping"
    DYNAMIC = "dynamic"


class Platform(Enum):
    """Advertising platform."""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    YOUTUBE = "youtube"


@dataclass(frozen=True)
class Budget:
    """Budget value object for advertising campaigns.

    Accepts legacy alias `total_limit` used in tests; if provided, it is mapped
    to `lifetime_limit`.
    """
    amount: Decimal = Decimal('0')
    currency: str = "USD"
    daily_limit: Optional[Decimal] = None
    lifetime_limit: Optional[Decimal] = None
    # Back-compat/test alias
    total_limit: Optional[Decimal] = None
    
    def __post_init__(self):
        # Map alias total_limit -> lifetime_limit if needed
        if self.total_limit is not None and self.lifetime_limit is None:
            object.__setattr__(self, 'lifetime_limit', self.total_limit)

        if self.amount < 0:
            raise ValueError("Budget amount must be non-negative")
        if self.daily_limit is not None and self.daily_limit <= 0:
            raise ValueError("Daily limit must be positive")
        if self.lifetime_limit is not None and self.lifetime_limit <= 0:
            raise ValueError("Lifetime limit must be positive")
        if self.daily_limit is not None and self.lifetime_limit is not None and self.daily_limit > self.lifetime_limit:
            raise ValueError("Daily limit cannot exceed lifetime limit")
    
    def is_within_daily_limit(self, spent_today: Decimal) -> bool:
        """Check if spending is within daily limit."""
        if not self.daily_limit:
            return True
        return spent_today < self.daily_limit
    
    def is_within_lifetime_limit(self, total_spent: Decimal) -> bool:
        """Check if spending is within lifetime limit."""
        if not self.lifetime_limit:
            return True
        return total_spent < self.lifetime_limit
    
    def remaining_daily_budget(self, spent_today: Decimal) -> Decimal:
        """Calculate remaining daily budget."""
        if not self.daily_limit:
            return Decimal('inf')
        return max(Decimal('0'), self.daily_limit - spent_today)
    
    def remaining_lifetime_budget(self, total_spent: Decimal) -> Decimal:
        """Calculate remaining lifetime budget."""
        if not self.lifetime_limit:
            return Decimal('inf')
        return max(Decimal('0'), self.lifetime_limit - total_spent)


@dataclass(frozen=True)
class TargetingCriteria:
    """Targeting criteria for advertisements.

    Supports both the newer structured fields and legacy/test aliases used in
    unit tests (e.g., demographics, location, behavior).
    """
    # Structured fields
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    genders: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    behaviors: Optional[List[str]] = None
    custom_audiences: Optional[List[str]] = None
    lookalike_audiences: Optional[List[str]] = None

    # Back-compat/test aliases
    demographics: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, Any]] = None
    behavior: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.age_min and self.age_max and self.age_min > self.age_max:
            raise ValueError("Age minimum cannot be greater than maximum")
        if self.age_min and (self.age_min < 13 or self.age_min > 65):
            raise ValueError("Age minimum must be between 13 and 65")
        if self.age_max and (self.age_max < 13 or self.age_max > 65):
            raise ValueError("Age maximum must be between 13 and 65")
    
    def is_valid(self) -> bool:
        """Check if targeting criteria is valid."""
        return any([
            self.age_min or self.age_max,
            self.genders,
            self.locations,
            self.interests,
            self.behaviors or self.behavior,
            self.custom_audiences,
            self.lookalike_audiences,
            self.demographics,
            self.location,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'age_min': self.age_min,
            'age_max': self.age_max,
            'genders': self.genders,
            'locations': self.locations,
            'interests': self.interests,
            'behaviors': self.behaviors or self.behavior,
            'custom_audiences': self.custom_audiences,
            'lookalike_audiences': self.lookalike_audiences,
            'demographics': self.demographics,
            'location': self.location,
        }


@dataclass(frozen=True)
class AdMetrics:
    """Performance metrics for advertisements."""
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: Decimal = Decimal('0')
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpm: Optional[Decimal] = None
    conversion_rate: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.impressions < 0:
            raise ValueError("Impressions cannot be negative")
        if self.clicks < 0:
            raise ValueError("Clicks cannot be negative")
        if self.conversions < 0:
            raise ValueError("Conversions cannot be negative")
        if self.spend < 0:
            raise ValueError("Spend cannot be negative")
        if self.clicks > self.impressions:
            raise ValueError("Clicks cannot exceed impressions")
        if self.conversions > self.clicks:
            raise ValueError("Conversions cannot exceed clicks")
    
    @property
    def calculated_ctr(self) -> Decimal:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return Decimal('0')
        return (self.clicks / self.impressions) * 100
    
    @property
    def calculated_cpc(self) -> Decimal:
        """Calculate cost per click."""
        if self.clicks == 0:
            return Decimal('0')
        return self.spend / self.clicks
    
    @property
    def calculated_cpm(self) -> Decimal:
        """Calculate cost per thousand impressions."""
        if self.impressions == 0:
            return Decimal('0')
        return (self.spend / self.impressions) * 1000
    
    @property
    def calculated_conversion_rate(self) -> Decimal:
        """Calculate conversion rate."""
        if self.clicks == 0:
            return Decimal('0')
        return (self.conversions / self.clicks) * 100
    
    def update_metrics(self, **kwargs) -> 'AdMetrics':
        """Create new metrics with updated values."""
        current_data = {
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'spend': self.spend
        }
        current_data.update(kwargs)
        return AdMetrics(**current_data)


@dataclass(frozen=True)
class AdSchedule:
    """Schedule for advertisement display."""
    start_date: datetime
    end_date: Optional[datetime] = None
    start_time: Optional[str] = None  # HH:MM format
    end_time: Optional[str] = None    # HH:MM format
    days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    timezone: str = "UTC"
    
    def __post_init__(self):
        if self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.start_time and self.end_time:
            if self.start_time >= self.end_time:
                raise ValueError("Start time must be before end time")
        if self.days_of_week:
            if not all(0 <= day <= 6 for day in self.days_of_week):
                raise ValueError("Days of week must be between 0 and 6")
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Check if the schedule is currently active."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Check date range
        if current_time < self.start_date:
            return False
        if self.end_date and current_time > self.end_date:
            return False
        
        # Check time range
        if self.start_time and self.end_time:
            current_time_str = current_time.strftime("%H:%M")
            if not (self.start_time <= current_time_str <= self.end_time):
                return False
        
        # Check day of week
        if self.days_of_week:
            current_day = current_time.weekday()
            if current_day not in self.days_of_week:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'days_of_week': self.days_of_week,
            'timezone': self.timezone
        }
