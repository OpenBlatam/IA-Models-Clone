"""
🎯 ADS Domain - Core Business Entities

Core business entities that represent the main concepts in the advertising domain.
These entities have identity and lifecycle, and contain business logic.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from .value_objects import (
    AdStatus, AdType, Platform, Budget, TargetingCriteria, 
    AdMetrics, AdSchedule
)


@dataclass
class Ad:
    """Core advertisement entity."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    ad_type: AdType = AdType.TEXT
    platform: Platform = Platform.FACEBOOK
    status: AdStatus = AdStatus.DRAFT
    
    # Content
    headline: str = ""
    body_text: str = ""
    # Back-compat/test alias for content
    content: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    call_to_action: Optional[str] = None
    
    # Targeting and Budget
    targeting: TargetingCriteria = field(default_factory=TargetingCriteria)
    # Back-compat/test alias
    targeting_criteria: Optional[TargetingCriteria] = None
    budget: Budget = field(default_factory=lambda: Budget(Decimal('100')))
    schedule: Optional[AdSchedule] = None
    
    # Performance
    metrics: AdMetrics = field(default_factory=AdMetrics)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[UUID] = None
    campaign_id: Optional[UUID] = None
    ad_group_id: Optional[UUID] = None
    # Back-compat/test alias
    group_id: Optional[UUID] = None
    
    def __post_init__(self):
        """Validate entity after initialization."""
        # Map legacy alias group_id -> ad_group_id if provided
        if self.ad_group_id is None and self.group_id is not None:
            self.ad_group_id = self.group_id
        # Map legacy alias targeting_criteria -> targeting if provided
        if self.targeting_criteria is not None and not self.targeting.is_valid():
            self.targeting = self.targeting_criteria
        # If content is provided and body_text empty, use content
        if (self.content and not self.body_text.strip()):
            self.body_text = self.content
        # If headline empty, default to name
        if not self.headline.strip():
            self.headline = self.name or self.headline
        # Basic name validation
        if not (self.name and self.name.strip()):
            raise ValueError("Ad name cannot be empty")
    
    def approve(self) -> None:
        """Approve the advertisement."""
        if self.status != AdStatus.PENDING_REVIEW:
            raise ValueError("Only pending review ads can be approved")
        self.status = AdStatus.APPROVED
        self.updated_at = datetime.now(timezone.utc)
    
    def reject(self, reason: str) -> None:
        """Reject the advertisement."""
        if self.status != AdStatus.PENDING_REVIEW:
            raise ValueError("Only pending review ads can be rejected")
        self.status = AdStatus.REJECTED
        self.updated_at = datetime.now(timezone.utc)
    
    def activate(self) -> None:
        """Activate the advertisement."""
        if self.status not in [AdStatus.APPROVED, AdStatus.PAUSED]:
            raise ValueError("Only approved or paused ads can be activated")
        if not self.schedule or not self.schedule.is_active():
            raise ValueError("Ad schedule must be active to activate ad")
        self.status = AdStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
    
    def pause(self) -> None:
        """Pause the advertisement."""
        if self.status != AdStatus.ACTIVE:
            raise ValueError("Only active ads can be paused")
        self.status = AdStatus.PAUSED
        self.updated_at = datetime.now(timezone.utc)
    
    def archive(self) -> None:
        """Archive the advertisement."""
        if self.status in [AdStatus.ACTIVE, AdStatus.PAUSED]:
            raise ValueError("Active or paused ads cannot be archived")
        self.status = AdStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc)
    
    def update_metrics(self, **kwargs) -> None:
        """Update performance metrics."""
        self.metrics = self.metrics.update_metrics(**kwargs)
        self.updated_at = datetime.now(timezone.utc)
    
    def is_active(self) -> bool:
        """Check if the ad is currently active."""
        return (
            self.status == AdStatus.ACTIVE and
            (not self.schedule or self.schedule.is_active())
        )
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if the ad can spend the specified amount."""
        return (
            self.is_active() and
            self.budget.is_within_daily_limit(self.metrics.spend + amount) and
            self.budget.is_within_lifetime_limit(self.metrics.spend + amount)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'ad_type': self.ad_type.value,
            'platform': self.platform.value,
            'status': self.status.value,
            'headline': self.headline,
            'body_text': self.body_text,
            'image_url': self.image_url,
            'video_url': self.video_url,
            'call_to_action': self.call_to_action,
            'targeting': self.targeting.to_dict(),
            'budget': {
                'amount': str(self.budget.amount),
                'currency': self.budget.currency,
                'daily_limit': str(self.budget.daily_limit) if self.budget.daily_limit else None,
                'lifetime_limit': str(self.budget.lifetime_limit) if self.budget.lifetime_limit else None
            },
            'schedule': self.schedule.to_dict() if self.schedule else None,
            'metrics': {
                'impressions': self.metrics.impressions,
                'clicks': self.metrics.clicks,
                'conversions': self.metrics.conversions,
                'spend': str(self.metrics.spend)
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': str(self.created_by) if self.created_by else None,
            'campaign_id': str(self.campaign_id) if self.campaign_id else None,
            'ad_group_id': str(self.ad_group_id) if self.ad_group_id else None
        }


@dataclass
class AdCampaign:
    """Advertising campaign entity."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    objective: str = ""
    status: AdStatus = AdStatus.DRAFT
    
    # Budget and Schedule
    budget: Budget = field(default_factory=lambda: Budget(Decimal('1000')))
    schedule: Optional[AdSchedule] = None
    
    # Targeting
    targeting: TargetingCriteria = field(default_factory=TargetingCriteria)
    
    # Performance
    total_spend: Decimal = Decimal('0')
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[UUID] = None
    platform: Platform = Platform.FACEBOOK
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.name.strip():
            raise ValueError("Campaign name cannot be empty")
        if not self.objective.strip():
            raise ValueError("Campaign objective cannot be empty")
    
    def add_ad(self, ad: Ad) -> None:
        """Add an advertisement to the campaign."""
        ad.campaign_id = self.id
        ad.platform = self.platform
        ad.targeting = self.targeting
        ad.budget = self.budget
    
    def remove_ad(self, ad: Ad) -> None:
        """Remove an advertisement from the campaign."""
        if ad.campaign_id == self.id:
            ad.campaign_id = None
    
    def update_performance(self, impressions: int, clicks: int, conversions: int, spend: Decimal) -> None:
        """Update campaign performance metrics."""
        self.total_impressions += impressions
        self.total_clicks += clicks
        self.total_conversions += conversions
        self.total_spend += spend
        self.updated_at = datetime.now(timezone.utc)
    
    def is_active(self) -> bool:
        """Check if the campaign is currently active."""
        return (
            self.status == AdStatus.ACTIVE and
            (not self.schedule or self.schedule.is_active())
        )
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if the campaign can spend the specified amount."""
        return (
            self.is_active() and
            self.budget.is_within_daily_limit(self.total_spend + amount) and
            self.budget.is_within_lifetime_limit(self.total_spend + amount)
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get campaign performance summary."""
        return {
            'total_spend': str(self.total_spend),
            'total_impressions': self.total_impressions,
            'total_clicks': self.total_clicks,
            'total_conversions': self.total_conversions,
            'ctr': (self.total_clicks / self.total_impressions * 100) if self.total_impressions > 0 else 0,
            'cpc': (self.total_spend / self.total_clicks) if self.total_clicks > 0 else 0,
            'cpm': (self.total_spend / self.total_impressions * 1000) if self.total_impressions > 0 else 0,
            'conversion_rate': (self.total_conversions / self.total_clicks * 100) if self.total_clicks > 0 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'objective': self.objective,
            'status': self.status.value,
            'budget': {
                'amount': str(self.budget.amount),
                'currency': self.budget.currency,
                'daily_limit': str(self.budget.daily_limit) if self.budget.daily_limit else None,
                'lifetime_limit': str(self.budget.lifetime_limit) if self.budget.lifetime_limit else None
            },
            'schedule': self.schedule.to_dict() if self.schedule else None,
            'targeting': self.targeting.to_dict(),
            'performance': self.get_performance_summary(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': str(self.created_by) if self.created_by else None,
            'platform': self.platform.value
        }


@dataclass
class AdGroup:
    """Advertisement group entity."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    status: AdStatus = AdStatus.DRAFT
    
    # Targeting
    targeting: TargetingCriteria = field(default_factory=TargetingCriteria)
    
    # Budget
    budget: Budget = field(default_factory=lambda: Budget(Decimal('500')))
    
    # Performance
    total_spend: Decimal = Decimal('0')
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    campaign_id: Optional[UUID] = None
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.name.strip():
            raise ValueError("Ad group name cannot be empty")
    
    def add_ad(self, ad: Ad) -> None:
        """Add an advertisement to the group."""
        ad.ad_group_id = self.id
        ad.targeting = self.targeting
        ad.budget = self.budget
    
    def remove_ad(self, ad: Ad) -> None:
        """Remove an advertisement from the group."""
        if ad.ad_group_id == self.id:
            ad.ad_group_id = None
    
    def update_performance(self, impressions: int, clicks: int, conversions: int, spend: Decimal) -> None:
        """Update group performance metrics."""
        self.total_impressions += impressions
        self.total_clicks += clicks
        self.total_conversions += conversions
        self.total_spend += spend
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'targeting': self.targeting.to_dict(),
            'budget': {
                'amount': str(self.budget.amount),
                'currency': self.budget.currency,
                'daily_limit': str(self.budget.daily_limit) if self.budget.daily_limit else None,
                'lifetime_limit': str(self.budget.lifetime_limit) if self.budget.lifetime_limit else None
            },
            'performance': {
                'total_spend': str(self.total_spend),
                'total_impressions': self.total_impressions,
                'total_clicks': self.total_clicks,
                'total_conversions': self.total_conversions
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'campaign_id': str(self.campaign_id) if self.campaign_id else None
        }


@dataclass
class AdPerformance:
    """Advertisement performance tracking entity."""
    
    ad_id: UUID
    id: UUID = field(default_factory=uuid4)
    campaign_id: Optional[UUID] = None
    ad_group_id: Optional[UUID] = None
    
    # Metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: Decimal = Decimal('0')
    
    # Calculated metrics
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpm: Optional[Decimal] = None
    conversion_rate: Optional[Decimal] = None
    
    # Time tracking
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc).date())
    hour: Optional[int] = None  # 0-23 for hourly tracking
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate derived performance metrics."""
        if self.impressions > 0:
            self.ctr = (self.clicks / self.impressions) * 100
            self.cpm = (self.spend / self.impressions) * 1000
        
        if self.clicks > 0:
            self.cpc = self.spend / self.clicks
            self.conversion_rate = (self.conversions / self.clicks) * 100
    
    def update_metrics(self, impressions: int = 0, clicks: int = 0, 
                      conversions: int = 0, spend: Decimal = Decimal('0')) -> None:
        """Update performance metrics."""
        self.impressions += impressions
        self.clicks += clicks
        self.conversions += conversions
        self.spend += spend
        self.updated_at = datetime.now(timezone.utc)
        self._calculate_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'id': str(self.id),
            'ad_id': str(self.ad_id),
            'campaign_id': str(self.campaign_id) if self.campaign_id else None,
            'ad_group_id': str(self.ad_group_id) if self.ad_group_id else None,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'spend': str(self.spend),
            'ctr': float(self.ctr) if self.ctr else None,
            'cpc': float(self.cpc) if self.cpc else None,
            'cpm': float(self.cpm) if self.cpm else None,
            'conversion_rate': float(self.conversion_rate) if self.conversion_rate else None,
            'date': self.date.isoformat(),
            'hour': self.hour,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
