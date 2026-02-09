"""
Unit tests for the ads domain layer.

This module consolidates tests for:
- Entities (Ad, AdCampaign, AdGroup, AdPerformance)
- Value Objects (AdStatus, AdType, Budget, TargetingCriteria)
- Repository interfaces
- Domain services
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List

from agents.backend.onyx.server.features.ads.domain.entities import (
    Ad, AdCampaign, AdGroup, AdPerformance
)
from agents.backend.onyx.server.features.ads.domain.value_objects import (
    AdStatus, AdType, Platform, Budget, TargetingCriteria, AdMetrics, AdSchedule
)
from agents.backend.onyx.server.features.ads.domain.repositories import (
    AdRepository, CampaignRepository, GroupRepository, PerformanceRepository
)
from agents.backend.onyx.server.features.ads.domain.services import (
    AdService, CampaignService, OptimizationService
)


class TestAdStatus:
    """Test AdStatus enum values and behavior."""
    
    def test_ad_status_values(self):
        """Test that all expected ad status values exist."""
        assert AdStatus.DRAFT == "draft"
        assert AdStatus.PENDING_REVIEW == "pending_review"
        assert AdStatus.APPROVED == "approved"
        assert AdStatus.ACTIVE == "active"
        assert AdStatus.PAUSED == "paused"
        assert AdStatus.ARCHIVED == "archived"
        assert AdStatus.REJECTED == "rejected"


class TestAdType:
    """Test AdType enum values and behavior."""
    
    def test_ad_type_values(self):
        """Test that all expected ad type values exist."""
        assert AdType.TEXT == "text"
        assert AdType.IMAGE == "image"
        assert AdType.VIDEO == "video"
        assert AdType.CAROUSEL == "carousel"
        assert AdType.STORY == "story"


class TestPlatform:
    """Test Platform enum values and behavior."""
    
    def test_platform_values(self):
        """Test that all expected platform values exist."""
        assert Platform.FACEBOOK == "facebook"
        assert Platform.INSTAGRAM == "instagram"
        assert Platform.GOOGLE == "google"
        assert Platform.LINKEDIN == "linkedin"
        assert Platform.TWITTER == "twitter"


class TestBudget:
    """Test Budget value object."""
    
    def test_budget_creation(self):
        """Test budget creation with valid values."""
        budget = Budget(
            daily_limit=Decimal("100.00"),
            total_limit=Decimal("1000.00"),
            currency="USD"
        )
        assert budget.daily_limit == Decimal("100.00")
        assert budget.total_limit == Decimal("1000.00")
        assert budget.currency == "USD"
    
    def test_budget_validation_daily_limit_positive(self):
        """Test that daily limit must be positive."""
        with pytest.raises(ValueError, match="Daily limit must be positive"):
            Budget(
                daily_limit=Decimal("-50.00"),
                total_limit=Decimal("1000.00"),
                currency="USD"
            )
    
    def test_budget_validation_total_limit_positive(self):
        """Test that total limit must be positive."""
        with pytest.raises(ValueError, match="Total limit must be positive"):
            Budget(
                daily_limit=Decimal("100.00"),
                total_limit=Decimal("-500.00"),
                currency="USD"
            )
    
    def test_budget_validation_daily_not_exceed_total(self):
        """Test that daily limit cannot exceed total limit."""
        with pytest.raises(ValueError, match="Daily limit cannot exceed total limit"):
            Budget(
                daily_limit=Decimal("1500.00"),
                total_limit=Decimal("1000.00"),
                currency="USD"
            )
    
    def test_budget_validation_currency_format(self):
        """Test that currency must be 3 uppercase letters."""
        with pytest.raises(ValueError, match="Currency must be 3 uppercase letters"):
            Budget(
                daily_limit=Decimal("100.00"),
                total_limit=Decimal("1000.00"),
                currency="usd"
            )


class TestTargetingCriteria:
    """Test TargetingCriteria value object."""
    
    def test_targeting_criteria_creation(self):
        """Test targeting criteria creation with valid values."""
        criteria = TargetingCriteria(
            demographics={"age_range": "25-34", "gender": "all"},
            interests=["technology", "business"],
            location={"country": "US", "cities": ["New York", "Los Angeles"]},
            behavior=["frequent_shoppers", "high_value_customers"]
        )
        assert criteria.demographics["age_range"] == "25-34"
        assert "technology" in criteria.interests
        assert criteria.location["country"] == "US"
        assert "frequent_shoppers" in criteria.behavior
    
    def test_targeting_criteria_validation_age_range(self):
        """Test age range validation."""
        with pytest.raises(ValueError, match="Invalid age range format"):
            TargetingCriteria(
                demographics={"age_range": "invalid"},
                interests=[],
                location={},
                behavior=[]
            )
    
    def test_targeting_criteria_validation_interests_not_empty(self):
        """Test that interests list cannot be empty."""
        with pytest.raises(ValueError, match="Interests list cannot be empty"):
            TargetingCriteria(
                demographics={},
                interests=[],
                location={},
                behavior=[]
            )


class TestAdMetrics:
    """Test AdMetrics value object."""
    
    def test_ad_metrics_creation(self):
        """Test ad metrics creation with valid values."""
        metrics = AdMetrics(
            impressions=1000,
            clicks=50,
            conversions=5,
            spend=Decimal("100.00"),
            ctr=0.05,
            cpc=Decimal("2.00"),
            cpm=Decimal("100.00")
        )
        assert metrics.impressions == 1000
        assert metrics.clicks == 50
        assert metrics.conversions == 5
        assert metrics.spend == Decimal("100.00")
        assert metrics.ctr == 0.05
        assert metrics.cpc == Decimal("2.00")
        assert metrics.cpm == Decimal("100.00")
    
    def test_ad_metrics_validation_positive_values(self):
        """Test that all metrics must be non-negative."""
        with pytest.raises(ValueError, match="Impressions must be non-negative"):
            AdMetrics(
                impressions=-100,
                clicks=50,
                conversions=5,
                spend=Decimal("100.00"),
                ctr=0.05,
                cpc=Decimal("2.00"),
                cpm=Decimal("100.00")
            )
    
    def test_ad_metrics_validation_ctr_range(self):
        """Test that CTR must be between 0 and 1."""
        with pytest.raises(ValueError, match="CTR must be between 0 and 1"):
            AdMetrics(
                impressions=1000,
                clicks=50,
                conversions=5,
                spend=Decimal("100.00"),
                ctr=1.5,
                cpc=Decimal("2.00"),
                cpm=Decimal("100.00")
            )


class TestAdSchedule:
    """Test AdSchedule value object."""
    
    def test_ad_schedule_creation(self):
        """Test ad schedule creation with valid values."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)
        
        schedule = AdSchedule(
            start_date=start_date,
            end_date=end_date,
            active_hours={"start": "09:00", "end": "17:00"},
            timezone="UTC",
            days_of_week=["monday", "tuesday", "wednesday", "thursday", "friday"]
        )
        assert schedule.start_date == start_date
        assert schedule.end_date == end_date
        assert schedule.active_hours["start"] == "09:00"
        assert schedule.timezone == "UTC"
        assert "monday" in schedule.days_of_week
    
    def test_ad_schedule_validation_dates(self):
        """Test that end date must be after start date."""
        start_date = datetime.now()
        end_date = start_date - timedelta(days=1)
        
        with pytest.raises(ValueError, match="End date must be after start date"):
            AdSchedule(
                start_date=start_date,
                end_date=end_date,
                active_hours={},
                timezone="UTC",
                days_of_week=[]
            )
    
    def test_ad_schedule_validation_time_format(self):
        """Test that time format must be HH:MM."""
        with pytest.raises(ValueError, match="Invalid time format"):
            AdSchedule(
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=30),
                active_hours={"start": "9:00", "end": "17:00"},
                timezone="UTC",
                days_of_week=[]
            )


class TestAd:
    """Test Ad entity."""
    
    def test_ad_creation(self):
        """Test ad creation with valid values."""
        ad = Ad(
            id="ad_123",
            campaign_id="campaign_456",
            group_id="group_789",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            status=AdStatus.DRAFT,
            platform=Platform.FACEBOOK,
            targeting_criteria=TargetingCriteria(
                demographics={"age_range": "25-34"},
                interests=["technology"],
                location={},
                behavior=[]
            ),
            budget=Budget(
                daily_limit=Decimal("50.00"),
                total_limit=Decimal("500.00"),
                currency="USD"
            ),
            created_at=datetime.now()
        )
        assert ad.id == "ad_123"
        assert ad.campaign_id == "campaign_456"
        assert ad.name == "Test Ad"
        assert ad.status == AdStatus.DRAFT
    
    def test_ad_status_transitions(self):
        """Test valid ad status transitions."""
        ad = Ad(
            id="ad_123",
            campaign_id="campaign_456",
            group_id="group_789",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            status=AdStatus.DRAFT,
            platform=Platform.FACEBOOK,
            targeting_criteria=TargetingCriteria(
                demographics={"age_range": "25-34"},
                interests=["technology"],
                location={},
                behavior=[]
            ),
            budget=Budget(
                daily_limit=Decimal("50.00"),
                total_limit=Decimal("500.00"),
                currency="USD"
            ),
            created_at=datetime.now()
        )
        
        # Draft -> Pending Review
        ad.submit_for_review()
        assert ad.status == AdStatus.PENDING_REVIEW
        
        # Pending Review -> Approved
        ad.approve()
        assert ad.status == AdStatus.APPROVED
        
        # Approved -> Active
        ad.activate()
        assert ad.status == AdStatus.ACTIVE
        
        # Active -> Paused
        ad.pause()
        assert ad.status == AdStatus.PAUSED
        
        # Paused -> Active
        ad.activate()
        assert ad.status == AdStatus.ACTIVE
    
    def test_ad_invalid_status_transitions(self):
        """Test invalid ad status transitions."""
        ad = Ad(
            id="ad_123",
            campaign_id="campaign_456",
            group_id="group_789",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            status=AdStatus.DRAFT,
            platform=Platform.FACEBOOK,
            targeting_criteria=TargetingCriteria(
                demographics={"age_range": "25-34"},
                interests=["technology"],
                location={},
                behavior=[]
            ),
            budget=Budget(
                daily_limit=Decimal("50.00"),
                total_limit=Decimal("500.00"),
                currency="USD"
            ),
            created_at=datetime.now()
        )
        
        # Cannot activate from draft
        with pytest.raises(ValueError, match="Cannot activate ad from draft status"):
            ad.activate()
        
        # Cannot pause from draft
        with pytest.raises(ValueError, match="Cannot pause ad from draft status"):
            ad.pause()


class TestAdCampaign:
    """Test AdCampaign entity."""
    
    def test_campaign_creation(self):
        """Test campaign creation with valid values."""
        campaign = AdCampaign(
            id="campaign_123",
            name="Test Campaign",
            description="A test advertising campaign",
            objective="awareness",
            status=AdStatus.DRAFT,
            budget=Budget(
                daily_limit=Decimal("200.00"),
                total_limit=Decimal("2000.00"),
                currency="USD"
            ),
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=60),
            created_at=datetime.now()
        )
        assert campaign.id == "campaign_123"
        assert campaign.name == "Test Campaign"
        assert campaign.objective == "awareness"
        assert campaign.status == AdStatus.DRAFT
    
    def test_campaign_status_transitions(self):
        """Test valid campaign status transitions."""
        campaign = AdCampaign(
            id="campaign_123",
            name="Test Campaign",
            description="A test advertising campaign",
            objective="awareness",
            status=AdStatus.DRAFT,
            budget=Budget(
                daily_limit=Decimal("200.00"),
                total_limit=Decimal("2000.00"),
                currency="USD"
            ),
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=60),
            created_at=datetime.now()
        )
        
        # Draft -> Active
        campaign.activate()
        assert campaign.status == AdStatus.ACTIVE
        
        # Active -> Paused
        campaign.pause()
        assert campaign.status == AdStatus.PAUSED
        
        # Paused -> Active
        campaign.activate()
        assert campaign.status == AdStatus.ACTIVE


class TestAdGroup:
    """Test AdGroup entity."""
    
    def test_group_creation(self):
        """Test ad group creation with valid values."""
        group = AdGroup(
            id="group_123",
            campaign_id="campaign_456",
            name="Test Ad Group",
            description="A test ad group",
            status=AdStatus.DRAFT,
            targeting_criteria=TargetingCriteria(
                demographics={"age_range": "25-34"},
                interests=["technology"],
                location={},
                behavior=[]
            ),
            bid_amount=Decimal("2.50"),
            created_at=datetime.now()
        )
        assert group.id == "group_123"
        assert group.campaign_id == "campaign_456"
        assert group.name == "Test Ad Group"
        assert group.bid_amount == Decimal("2.50")


class TestAdPerformance:
    """Test AdPerformance entity."""
    
    def test_performance_creation(self):
        """Test ad performance creation with valid values."""
        performance = AdPerformance(
            id="perf_123",
            ad_id="ad_456",
            date=datetime.now().date(),
            metrics=AdMetrics(
                impressions=1000,
                clicks=50,
                conversions=5,
                spend=Decimal("100.00"),
                ctr=0.05,
                cpc=Decimal("2.00"),
                cpm=Decimal("100.00")
            ),
            created_at=datetime.now()
        )
        assert performance.id == "perf_123"
        assert performance.ad_id == "ad_456"
        assert performance.metrics.impressions == 1000
        assert performance.metrics.ctr == 0.05
    
    def test_performance_update_metrics(self):
        """Test updating performance metrics."""
        performance = AdPerformance(
            id="perf_123",
            ad_id="ad_456",
            date=datetime.now().date(),
            metrics=AdMetrics(
                impressions=1000,
                clicks=50,
                conversions=5,
                spend=Decimal("100.00"),
                ctr=0.05,
                cpc=Decimal("2.00"),
                cpm=Decimal("100.00")
            ),
            created_at=datetime.now()
        )
        
        new_metrics = AdMetrics(
            impressions=1500,
            clicks=75,
            conversions=8,
            spend=Decimal("150.00"),
            ctr=0.05,
            cpc=Decimal("2.00"),
            cpm=Decimal("100.00")
        )
        
        performance.update_metrics(new_metrics)
        assert performance.metrics.impressions == 1500
        assert performance.metrics.clicks == 75
        assert performance.metrics.conversions == 8


class TestRepositoryInterfaces:
    """Test repository interface definitions."""
    
    def test_ad_repository_interface(self):
        """Test that AdRepository interface has required methods."""
        # This test ensures the interface contract is maintained
        required_methods = [
            'create', 'get_by_id', 'get_all', 'update', 'delete',
            'get_by_campaign', 'get_by_status', 'get_by_platform'
        ]
        
        for method_name in required_methods:
            assert hasattr(AdRepository, method_name)
    
    def test_campaign_repository_interface(self):
        """Test that CampaignRepository interface has required methods."""
        required_methods = [
            'create', 'get_by_id', 'get_all', 'update', 'delete',
            'get_by_status', 'get_by_objective'
        ]
        
        for method_name in required_methods:
            assert hasattr(CampaignRepository, method_name)


class TestDomainServices:
    """Test domain services."""
    
    def test_ad_service_creation(self):
        """Test AdService creation with dependencies."""
        # Mock repositories
        mock_ad_repo = pytest.Mock(spec=AdRepository)
        mock_campaign_repo = pytest.Mock(spec=CampaignRepository)
        
        service = AdService(
            ad_repository=mock_ad_repo,
            campaign_repository=mock_campaign_repo
        )
        
        assert service.ad_repository == mock_ad_repo
        assert service.campaign_repository == mock_campaign_repo
    
    def test_campaign_service_creation(self):
        """Test CampaignService creation with dependencies."""
        mock_campaign_repo = pytest.Mock(spec=CampaignRepository)
        mock_group_repo = pytest.Mock(spec=GroupRepository)
        
        service = CampaignService(
            campaign_repository=mock_campaign_repo,
            group_repository=mock_group_repo
        )
        
        assert service.campaign_repository == mock_campaign_repo
        assert service.group_repository == mock_group_repo
    
    def test_optimization_service_creation(self):
        """Test OptimizationService creation with dependencies."""
        mock_performance_repo = pytest.Mock(spec=PerformanceRepository)
        mock_analytics_repo = pytest.Mock()
        
        service = OptimizationService(
            performance_repository=mock_performance_repo,
            analytics_repository=mock_analytics_repo
        )
        
        assert service.performance_repository == mock_performance_repo
        assert service.analytics_repository == mock_analytics_repo


if __name__ == "__main__":
    pytest.main([__file__])
