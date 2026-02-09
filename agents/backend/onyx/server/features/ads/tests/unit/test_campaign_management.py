#!/usr/bin/env python3
"""
Comprehensive tests for campaign and ad group management
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone, timedelta

# Add current directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from domain.entities import Ad, AdCampaign, AdGroup
from domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdSchedule

class TestAdCampaign:
    """Test the AdCampaign entity"""
    
    def test_campaign_creation(self):
        """Test campaign creation with valid data"""
        campaign = AdCampaign(
            name="Test Campaign",
            description="Test Description",
            objective="AWARENESS",
            platform=Platform.FACEBOOK,
            budget=Budget(
                amount=5000.0,
                daily_limit=Decimal('500.0'),
                lifetime_limit=Decimal('5000.0'),
                currency="USD"
            ),
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
        
        assert campaign.name == "Test Campaign"
        assert campaign.description == "Test Description"
        assert campaign.objective == "AWARENESS"
        assert campaign.platform == Platform.FACEBOOK
        assert campaign.status == AdStatus.DRAFT
        assert campaign.budget.amount == 5000.0
    
    def test_campaign_validation(self):
        """Test campaign validation rules"""
        with pytest.raises(ValueError):
            AdCampaign(
                name="",  # Empty name should fail
                description="Test",
                objective="AWARENESS",
                platform=Platform.FACEBOOK,
                budget=Budget(
                    amount=1000.0,
                    daily_limit=Decimal('100.0'),
                    lifetime_limit=Decimal('1000.0'),
                    currency="USD"
                ),
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc) + timedelta(days=30)
            )
    
    def test_campaign_status_transitions(self):
        """Test campaign status transitions"""
        campaign = AdCampaign(
            name="Test Campaign",
            description="Test Description",
            objective="AWARENESS",
            platform=Platform.FACEBOOK,
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
        
        # Start with DRAFT
        assert campaign.status == AdStatus.DRAFT
        
        # Activate campaign
        campaign.activate()
        assert campaign.status == AdStatus.ACTIVE
        
        # Pause campaign
        campaign.pause()
        assert campaign.status == AdStatus.PAUSED
        
        # Resume campaign
        campaign.resume()
        assert campaign.status == AdStatus.ACTIVE
        
        # Complete campaign
        campaign.complete()
        assert campaign.status == AdStatus.COMPLETED
    
    def test_campaign_budget_management(self):
        """Test campaign budget management"""
        campaign = AdCampaign(
            name="Test Campaign",
            description="Test Description",
            objective="AWARENESS",
            platform=Platform.FACEBOOK,
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
        
        # Check initial budget
        assert campaign.budget.amount == 1000.0
        assert campaign.budget.daily_limit == Decimal('100.0')
        
        # Update budget
        new_budget = Budget(
            amount=2000.0,
            daily_limit=Decimal('200.0'),
            lifetime_limit=Decimal('2000.0'),
            currency="USD"
        )
        campaign.update_budget(new_budget)
        assert campaign.budget.amount == 2000.0
        assert campaign.budget.daily_limit == Decimal('200.0')

class TestAdGroup:
    """Test the AdGroup entity"""
    
    def test_ad_group_creation(self):
        """Test ad group creation"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US", "CA"],
            interests=["technology", "business"]
        )
        
        ad_group = AdGroup(
            name="Test Ad Group",
            description="Test Description",
            targeting=targeting,
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            campaign_id="test-campaign"
        )
        
        assert ad_group.name == "Test Ad Group"
        assert ad_group.targeting.age_min == 18
        assert ad_group.targeting.age_max == 65
        assert ad_group.status == AdStatus.DRAFT
    
    def test_ad_group_targeting(self):
        """Test ad group targeting functionality"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US", "CA"],
            interests=["technology", "business"],
            demographics=["college_educated"],
            behaviors=["frequent_travelers"]
        )
        
        ad_group = AdGroup(
            name="Test Group",
            description="Test",
            targeting=targeting,
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            campaign_id="test-campaign"
        )
        
        # Test targeting validation
        assert ad_group.targeting.is_valid()
        assert len(ad_group.targeting.locations) == 2
        assert "technology" in ad_group.targeting.interests
    
    def test_ad_group_status_management(self):
        """Test ad group status management"""
        ad_group = AdGroup(
            name="Test Group",
            description="Test",
            targeting=TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=["US"],
                interests=["technology"]
            ),
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            campaign_id="test-campaign"
        )
        
        # Test status transitions
        ad_group.activate()
        assert ad_group.status == AdStatus.ACTIVE
        
        ad_group.pause()
        assert ad_group.status == AdStatus.PAUSED
        
        ad_group.resume()
        assert ad_group.status == AdStatus.ACTIVE

class TestAdEntity:
    """Test the Ad entity"""
    
    def test_ad_creation(self):
        """Test ad creation with all required fields"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"]
        )
        
        budget = Budget(
            amount=500.0,
            daily_limit=Decimal('50.0'),
            lifetime_limit=Decimal('500.0'),
            currency="USD"
        )
        
        schedule = AdSchedule(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
        
        ad = Ad(
            name="Test Ad",
            description="Test Description",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            headline="Test Headline",
            body_text="Test Body Text",
            image_url="https://example.com/image.jpg",
            call_to_action="Learn More",
            targeting=targeting,
            budget=budget,
            schedule=schedule,
            campaign_id="test-campaign"
        )
        
        assert ad.name == "Test Ad"
        assert ad.ad_type == AdType.IMAGE
        assert ad.platform == Platform.FACEBOOK
        assert ad.status == AdStatus.DRAFT
        assert ad.schedule.is_active()
    
    def test_ad_lifecycle(self):
        """Test ad lifecycle methods"""
        ad = self._create_test_ad()
        
        # Test approval workflow
        ad.approve()
        assert ad.status == AdStatus.APPROVED
        
        ad.activate()
        assert ad.status == AdStatus.ACTIVE
        
        ad.pause()
        assert ad.status == AdStatus.PAUSED
        
        ad.resume()
        assert ad.status == AdStatus.ACTIVE
        
        ad.complete()
        assert ad.status == AdStatus.COMPLETED
    
    def test_ad_validation(self):
        """Test ad validation rules"""
        # Test with invalid image URL
        with pytest.raises(ValueError):
            Ad(
                name="Test Ad",
                description="Test",
                ad_type=AdType.IMAGE,
                platform=Platform.FACEBOOK,
                headline="Test",
                body_text="Test",
                image_url="invalid-url",  # Invalid URL
                call_to_action="Learn More",
                targeting=self._create_test_targeting(),
                budget=self._create_test_budget(),
                schedule=self._create_test_schedule(),
                campaign_id="test-campaign"
            )
    
    def test_ad_metrics(self):
        """Test ad metrics functionality"""
        ad = self._create_test_ad()
        
        # Add metrics
        ad.add_impression()
        ad.add_click()
        ad.add_conversion()
        
        assert ad.metrics.impressions == 1
        assert ad.metrics.clicks == 1
        assert ad.metrics.conversions == 1
        
        # Calculate CTR
        ctr = ad.metrics.calculate_ctr()
        assert ctr == 1.0  # 1 click / 1 impression = 100%
    
    def _create_test_ad(self):
        """Helper method to create a test ad"""
        return Ad(
            name="Test Ad",
            description="Test Description",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            headline="Test Headline",
            body_text="Test Body Text",
            image_url="https://example.com/image.jpg",
            call_to_action="Learn More",
            targeting=self._create_test_targeting(),
            budget=self._create_test_budget(),
            schedule=self._create_test_schedule(),
            campaign_id="test-campaign"
        )
    
    def _create_test_targeting(self):
        """Helper method to create test targeting"""
        return TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"]
        )
    
    def _create_test_budget(self):
        """Helper method to create test budget"""
        return Budget(
            amount=500.0,
            daily_limit=Decimal('50.0'),
            lifetime_limit=Decimal('500.0'),
            currency="USD"
        )
    
    def _create_test_schedule(self):
        """Helper method to create test schedule"""
        return AdSchedule(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )

class TestCampaignIntegration:
    """Integration tests for campaign management"""
    
    def test_campaign_with_ad_groups_and_ads(self):
        """Test complete campaign structure with ad groups and ads"""
        # Create campaign
        campaign = AdCampaign(
            name="Integration Test Campaign",
            description="Test campaign with full structure",
            objective="CONVERSIONS",
            platform=Platform.FACEBOOK,
            budget=Budget(
                amount=5000.0,
                daily_limit=Decimal('500.0'),
                lifetime_limit=Decimal('5000.0'),
                currency="USD"
            ),
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
        
        # Create ad group
        ad_group = AdGroup(
            name="Test Ad Group",
            description="Test group",
            targeting=TargetingCriteria(
                age_min=25,
                age_max=45,
                locations=["US"],
                interests=["business", "entrepreneurship"]
            ),
            budget=Budget(
                amount=2000.0,
                daily_limit=Decimal('200.0'),
                lifetime_limit=Decimal('2000.0'),
                currency="USD"
            ),
            campaign_id=campaign.id
        )
        
        # Create ad
        ad = Ad(
            name="Test Ad",
            description="Test ad",
            ad_type=AdType.VIDEO,
            platform=Platform.FACEBOOK,
            headline="Transform Your Business",
            body_text="Discover innovative solutions",
            image_url="https://example.com/video.jpg",
            call_to_action="Get Started",
            targeting=ad_group.targeting,
            budget=Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            ),
            schedule=AdSchedule(
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc) + timedelta(days=30),
                days_of_week=[0, 1, 2, 3, 4, 5, 6]
            ),
            campaign_id=campaign.id
        )
        
        # Test campaign activation
        campaign.activate()
        assert campaign.status == AdStatus.ACTIVE
        
        # Test ad group activation
        ad_group.activate()
        assert ad_group.status == AdStatus.ACTIVE
        
        # Test ad approval and activation
        ad.approve()
        ad.activate()
        assert ad.status == AdStatus.ACTIVE
        
        print("✅ Campaign integration test completed successfully")

if __name__ == "__main__":
    # Run tests
    print("🧪 Running Campaign Management Tests...")
    
    # Test AdCampaign
    print("\n1. Testing AdCampaign...")
    campaign_test = TestAdCampaign()
    campaign_test.test_campaign_creation()
    campaign_test.test_campaign_validation()
    campaign_test.test_campaign_status_transitions()
    campaign_test.test_campaign_budget_management()
    print("✅ AdCampaign tests passed")
    
    # Test AdGroup
    print("\n2. Testing AdGroup...")
    group_test = TestAdGroup()
    group_test.test_ad_group_creation()
    group_test.test_ad_group_targeting()
    group_test.test_ad_group_status_management()
    print("✅ AdGroup tests passed")
    
    # Test Ad
    print("\n3. Testing Ad Entity...")
    ad_test = TestAdEntity()
    ad_test.test_ad_creation()
    ad_test.test_ad_lifecycle()
    ad_test.test_ad_validation()
    ad_test.test_ad_metrics()
    print("✅ Ad Entity tests passed")
    
    # Test Integration
    print("\n4. Testing Integration...")
    integration_test = TestCampaignIntegration()
    integration_test.test_campaign_with_ad_groups_and_ads()
    print("✅ Integration tests passed")
    
    print("\n🎉 All campaign management tests completed successfully!")
