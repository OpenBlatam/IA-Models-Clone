#!/usr/bin/env python3
"""
Comprehensive integration tests for the entire ADS system
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import asyncio

# Add current directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from domain.entities import Ad, AdCampaign, AdGroup
from domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdSchedule
from optimization.factory import OptimizationFactory
from optimization.base_optimizer import OptimizationContext
from optimization.performance_optimizer import PerformanceOptimizer
from optimization.profiling_optimizer import ProfilingOptimizer
from optimization.gpu_optimizer import GPUOptimizer
from domain.value_objects import OptimizationStrategy, OptimizationLevel

class TestSystemIntegration:
    """Integration tests for the entire ADS system"""
    
    def test_complete_campaign_workflow(self):
        """Test complete campaign creation and management workflow"""
        print("🔄 Testing complete campaign workflow...")
        
        # 1. Create campaign
        campaign = self._create_test_campaign()
        assert campaign.status == AdStatus.DRAFT
        print("✅ Campaign created successfully")
        
        # 2. Create ad group
        ad_group = self._create_test_ad_group(campaign.id)
        assert ad_group.status == AdStatus.DRAFT
        print("✅ Ad group created successfully")
        
        # 3. Create ad
        ad = self._create_test_ad(campaign.id, ad_group.targeting)
        assert ad.status == AdStatus.DRAFT
        print("✅ Ad created successfully")
        
        # 4. Activate campaign
        campaign.activate()
        assert campaign.status == AdStatus.ACTIVE
        print("✅ Campaign activated successfully")
        
        # 5. Activate ad group
        ad_group.activate()
        assert ad_group.status == AdStatus.ACTIVE
        print("✅ Ad group activated successfully")
        
        # 6. Approve and activate ad
        ad.approve()
        assert ad.status == AdStatus.APPROVED
        print("✅ Ad approved successfully")
        
        ad.activate()
        assert ad.status == AdStatus.ACTIVE
        print("✅ Ad activated successfully")
        
        # 7. Simulate ad performance
        self._simulate_ad_performance(ad)
        print("✅ Ad performance simulated successfully")
        
        # 8. Pause and resume ad
        ad.pause()
        assert ad.status == AdStatus.PAUSED
        print("✅ Ad paused successfully")
        
        ad.resume()
        assert ad.status == AdStatus.ACTIVE
        print("✅ Ad resumed successfully")
        
        # 9. Complete campaign
        campaign.complete()
        assert campaign.status == AdStatus.COMPLETED
        print("✅ Campaign completed successfully")
        
        print("🎉 Complete campaign workflow test passed!")
    
    def test_optimization_system_integration(self):
        """Test optimization system integration with real entities"""
        print("🔄 Testing optimization system integration...")
        
        # Create test entities
        campaign = self._create_test_campaign()
        ad_group = self._create_test_ad_group(campaign.id)
        ad = self._create_test_ad(campaign.id, ad_group.targeting)
        
        # Initialize optimization factory
        factory = OptimizationFactory()
        
        # Register optimizers
        factory.register_optimizer('performance', PerformanceOptimizer, {})
        factory.register_optimizer('profiling', ProfilingOptimizer, {})
        factory.register_optimizer('gpu', GPUOptimizer, {})
        
        # Test each optimizer with real entities
        optimizers_to_test = [
            ('performance', PerformanceOptimizer),
            ('profiling', ProfilingOptimizer),
            ('gpu', GPUOptimizer)
        ]
        
        for opt_type, optimizer_class in optimizers_to_test:
            print(f"  🔧 Testing {opt_type} optimizer...")
            
            # Create optimizer
            optimizer = factory.create_optimizer(opt_type, name=f"Test {opt_type.title()}")
            assert isinstance(optimizer, optimizer_class)
            
            # Create optimization context
            context = OptimizationContext(
                target_entity="ad",
                entity_id=ad.id,
                optimization_type=OptimizationStrategy.PERFORMANCE,
                level=OptimizationLevel.HIGH
            )
            
            # Run optimization
            result = optimizer.optimize(context, ad)
            assert result is not None
            assert hasattr(result, 'optimization_results')
            
            print(f"    ✅ {opt_type.title()} optimizer completed successfully")
        
        print("🎉 Optimization system integration test passed!")
    
    def test_budget_management_integration(self):
        """Test budget management across campaign, ad group, and ad levels"""
        print("🔄 Testing budget management integration...")
        
        # Create campaign with budget
        campaign = self._create_test_campaign()
        initial_campaign_budget = campaign.budget.amount
        
        # Create ad group with budget
        ad_group = self._create_test_ad_group(campaign.id)
        initial_group_budget = ad_group.budget.amount
        
        # Create ad with budget
        ad = self._create_test_ad(campaign.id, ad_group.targeting)
        initial_ad_budget = ad.budget.amount
        
        # Simulate spending at different levels
        print("  💰 Simulating budget spending...")
        
        # Campaign level spending
        campaign.budget.spend(Decimal('100.0'))
        assert campaign.budget.spent == Decimal('100.0')
        assert campaign.budget.remaining == Decimal(str(initial_campaign_budget - 100.0))
        print("    ✅ Campaign budget spending tracked")
        
        # Ad group level spending
        ad_group.budget.spend(Decimal('50.0'))
        assert ad_group.budget.spent == Decimal('50.0')
        assert ad_group.budget.remaining == Decimal(str(initial_group_budget - 50.0))
        print("    ✅ Ad group budget spending tracked")
        
        # Ad level spending
        ad.budget.spend(Decimal('25.0'))
        assert ad.budget.spent == Decimal('25.0')
        assert ad.budget.remaining == Decimal(str(initial_ad_budget - 25.0))
        print("    ✅ Ad budget spending tracked")
        
        # Test budget limits
        print("  🚫 Testing budget limits...")
        
        # Try to exceed ad budget
        with pytest.raises(ValueError):
            ad.budget.spend(Decimal(str(initial_ad_budget + 100.0)))
        print("    ✅ Ad budget limit enforced")
        
        # Try to exceed daily limit
        with pytest.raises(ValueError):
            ad.budget.spend_daily(Decimal(str(ad.budget.daily_limit + 1.0)))
        print("    ✅ Daily budget limit enforced")
        
        print("🎉 Budget management integration test passed!")
    
    def test_targeting_integration(self):
        """Test targeting system integration across campaign hierarchy"""
        print("🔄 Testing targeting integration...")
        
        # Create campaign
        campaign = self._create_test_campaign()
        
        # Create ad group with specific targeting
        ad_group_targeting = TargetingCriteria(
            age_min=25,
            age_max=45,
            locations=["US", "CA"],
            interests=["business", "entrepreneurship"],
            demographics=["college_educated"],
            behaviors=["frequent_travelers"]
        )
        
        ad_group = AdGroup(
            name="Targeted Ad Group",
            description="Group with specific targeting",
            targeting=ad_group_targeting,
            budget=self._create_test_budget(2000.0),
            campaign_id=campaign.id
        )
        
        # Create ad with inherited targeting
        ad = self._create_test_ad(campaign.id, ad_group.targeting)
        
        # Verify targeting inheritance
        assert ad.targeting.age_min == ad_group.targeting.age_min
        assert ad.targeting.age_max == ad_group.targeting.age_max
        assert ad.targeting.locations == ad_group.targeting.locations
        assert ad.targeting.interests == ad_group.targeting.interests
        assert ad.targeting.demographics == ad_group.targeting.demographics
        assert ad.targeting.behaviors == ad_group.targeting.behaviors
        
        print("  ✅ Targeting inheritance verified")
        
        # Test targeting validation
        assert ad.targeting.is_valid() is True
        assert ad_group.targeting.is_valid() is True
        
        print("  ✅ Targeting validation verified")
        
        # Test targeting modifications
        new_targeting = TargetingCriteria(
            age_min=30,
            age_max=50,
            locations=["US"],
            interests=["technology", "innovation"]
        )
        
        ad_group.update_targeting(new_targeting)
        assert ad_group.targeting.age_min == 30
        assert ad_group.targeting.age_max == 50
        assert "technology" in ad_group.targeting.interests
        
        print("  ✅ Targeting modifications verified")
        
        print("🎉 Targeting integration test passed!")
    
    def test_scheduling_integration(self):
        """Test scheduling system integration"""
        print("🔄 Testing scheduling integration...")
        
        # Create campaign with schedule
        campaign = self._create_test_campaign()
        
        # Create ad with schedule
        ad = self._create_test_ad(campaign.id, self._create_test_targeting())
        
        # Test schedule validation
        assert ad.schedule.is_active() is True
        print("  ✅ Schedule active status verified")
        
        # Test schedule modifications
        new_end_date = datetime.now(timezone.utc) + timedelta(days=60)
        ad.schedule.update_end_date(new_end_date)
        assert ad.schedule.end_date == new_end_date
        
        print("  ✅ Schedule modifications verified")
        
        # Test schedule constraints
        # Create ad with restrictive schedule
        restrictive_schedule = AdSchedule(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            days_of_week=[0, 1, 2, 3, 4],  # Weekdays only
            start_time="09:00",
            end_time="17:00"
        )
        
        restrictive_ad = Ad(
            name="Restrictive Schedule Ad",
            description="Ad with restrictive schedule",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            headline="Test Headline",
            body_text="Test Body",
            image_url="https://example.com/image.jpg",
            call_to_action="Learn More",
            targeting=self._create_test_targeting(),
            budget=self._create_test_budget(500.0),
            schedule=restrictive_schedule,
            campaign_id=campaign.id
        )
        
        # Test schedule validation
        assert restrictive_ad.schedule.start_time == "09:00"
        assert restrictive_ad.schedule.end_time == "17:00"
        assert restrictive_ad.schedule.days_of_week == [0, 1, 2, 3, 4]
        
        print("  ✅ Restrictive schedule verified")
        
        print("🎉 Scheduling integration test passed!")
    
    def test_metrics_integration(self):
        """Test metrics system integration across entities"""
        print("🔄 Testing metrics integration...")
        
        # Create campaign and ad
        campaign = self._create_test_campaign()
        ad = self._create_test_ad(campaign.id, self._create_test_targeting())
        
        # Simulate ad performance
        print("  📊 Simulating ad performance...")
        
        # Add impressions
        for _ in range(1000):
            ad.add_impression()
        
        # Add clicks
        for _ in range(50):
            ad.add_click()
        
        # Add conversions
        for _ in range(10):
            ad.add_conversion()
        
        # Add spend
        ad.add_spend(Decimal('250.0'))
        
        # Verify metrics
        assert ad.metrics.impressions == 1000
        assert ad.metrics.clicks == 50
        assert ad.metrics.conversions == 10
        assert ad.metrics.spend == Decimal('250.0')
        
        print("    ✅ Metrics data collected")
        
        # Calculate performance metrics
        ctr = ad.metrics.calculate_ctr()
        conversion_rate = ad.metrics.calculate_conversion_rate()
        cpc = ad.metrics.calculate_cpc()
        
        expected_ctr = 50.0 / 1000.0  # 5%
        expected_conversion_rate = 10.0 / 50.0  # 20%
        expected_cpc = 250.0 / 50.0  # $5.00
        
        assert ctr == pytest.approx(expected_ctr)
        assert conversion_rate == pytest.approx(expected_conversion_rate)
        assert cpc == pytest.approx(expected_cpc)
        
        print("    ✅ Performance metrics calculated correctly")
        
        # Test metrics aggregation (if implemented)
        # This would typically be done at the campaign level
        print("  🔄 Testing metrics aggregation...")
        
        # Create multiple ads for aggregation testing
        ads = []
        for i in range(3):
            test_ad = self._create_test_ad(campaign.id, self._create_test_targeting())
            test_ad.add_impression()
            test_ad.add_click()
            test_ad.add_spend(Decimal('100.0'))
            ads.append(test_ad)
        
        # Calculate campaign-level metrics
        total_impressions = sum(ad.metrics.impressions for ad in ads)
        total_clicks = sum(ad.metrics.clicks for ad in ads)
        total_spend = sum(ad.metrics.spend for ad in ads)
        
        assert total_impressions == 3
        assert total_clicks == 3
        assert total_spend == Decimal('300.0')
        
        print("    ✅ Metrics aggregation verified")
        
        print("🎉 Metrics integration test passed!")
    
    def test_error_handling_integration(self):
        """Test error handling across the system"""
        print("🔄 Testing error handling integration...")
        
        # Test invalid campaign creation
        print("  🚫 Testing invalid campaign creation...")
        
        with pytest.raises(ValueError):
            AdCampaign(
                name="",  # Empty name
                description="Test",
                objective="AWARENESS",
                platform=Platform.FACEBOOK,
                budget=self._create_test_budget(1000.0),
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc) + timedelta(days=30)
            )
        print("    ✅ Invalid campaign creation handled")
        
        # Test invalid ad creation
        print("  🚫 Testing invalid ad creation...")
        
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
                budget=self._create_test_budget(500.0),
                schedule=self._create_test_schedule(),
                campaign_id="test-campaign"
            )
        print("    ✅ Invalid ad creation handled")
        
        # Test invalid budget operations
        print("  🚫 Testing invalid budget operations...")
        
        budget = self._create_test_budget(1000.0)
        
        with pytest.raises(ValueError):
            budget.spend(Decimal('1500.0'))  # Exceed budget
        
        with pytest.raises(ValueError):
            budget.spend_daily(Decimal('150.0'))  # Exceed daily limit
        
        print("    ✅ Invalid budget operations handled")
        
        # Test invalid targeting
        print("  🚫 Testing invalid targeting...")
        
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=25,
                age_max=18,  # min > max
                locations=["US"],
                interests=["technology"]
            )
        
        print("    ✅ Invalid targeting handled")
        
        print("🎉 Error handling integration test passed!")
    
    def test_performance_integration(self):
        """Test system performance under load"""
        print("🔄 Testing performance integration...")
        
        # Create multiple campaigns and ads
        print("  🚀 Creating multiple entities for performance test...")
        
        campaigns = []
        ads = []
        
        for i in range(10):
            # Create campaign
            campaign = AdCampaign(
                name=f"Performance Test Campaign {i}",
                description=f"Campaign {i} for performance testing",
                objective="AWARENESS",
                platform=Platform.FACEBOOK,
                budget=self._create_test_budget(1000.0),
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc) + timedelta(days=30)
            )
            campaigns.append(campaign)
            
            # Create ad for each campaign
            ad = self._create_test_ad(campaign.id, self._create_test_targeting())
            ads.append(ad)
        
        print(f"    ✅ Created {len(campaigns)} campaigns and {len(ads)} ads")
        
        # Test bulk operations
        print("  ⚡ Testing bulk operations...")
        
        # Activate all campaigns
        for campaign in campaigns:
            campaign.activate()
            assert campaign.status == AdStatus.ACTIVE
        
        print("    ✅ All campaigns activated")
        
        # Activate all ads
        for ad in ads:
            ad.approve()
            ad.activate()
            assert ad.status == AdStatus.ACTIVE
        
        print("    ✅ All ads activated")
        
        # Simulate performance data for all ads
        print("  📊 Simulating performance data...")
        
        for ad in ads:
            for _ in range(100):
                ad.add_impression()
            for _ in range(5):
                ad.add_click()
            ad.add_spend(Decimal('50.0'))
        
        print("    ✅ Performance data simulated")
        
        # Test optimization system with multiple entities
        print("  🔧 Testing optimization system performance...")
        
        factory = OptimizationFactory()
        factory.register_optimizer('performance', PerformanceOptimizer, {})
        
        for i, ad in enumerate(ads[:5]):  # Test with first 5 ads
            optimizer = factory.create_optimizer('performance', name=f"Perf Opt {i}")
            
            context = OptimizationContext(
                target_entity="ad",
                entity_id=ad.id,
                optimization_type=OptimizationStrategy.PERFORMANCE,
                level=OptimizationLevel.HIGH
            )
            
            result = optimizer.optimize(context, ad)
            assert result is not None
        
        print("    ✅ Optimization system performance verified")
        
        print("🎉 Performance integration test passed!")
    
    # Helper methods
    def _create_test_campaign(self):
        """Create a test campaign"""
        return AdCampaign(
            name="Integration Test Campaign",
            description="Campaign for integration testing",
            objective="AWARENESS",
            platform=Platform.FACEBOOK,
            budget=self._create_test_budget(5000.0),
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
    
    def _create_test_ad_group(self, campaign_id):
        """Create a test ad group"""
        return AdGroup(
            name="Integration Test Ad Group",
            description="Ad group for integration testing",
            targeting=self._create_test_targeting(),
            budget=self._create_test_budget(2000.0),
            campaign_id=campaign_id
        )
    
    def _create_test_ad(self, campaign_id, targeting):
        """Create a test ad"""
        return Ad(
            name="Integration Test Ad",
            description="Ad for integration testing",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            headline="Integration Test Headline",
            body_text="Integration test body text",
            image_url="https://example.com/integration-test.jpg",
            call_to_action="Test Integration",
            targeting=targeting,
            budget=self._create_test_budget(1000.0),
            schedule=self._create_test_schedule(),
            campaign_id=campaign_id
        )
    
    def _create_test_targeting(self):
        """Create test targeting criteria"""
        return TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology", "business"]
        )
    
    def _create_test_budget(self, amount):
        """Create test budget"""
        return Budget(
            amount=amount,
            daily_limit=Decimal(str(amount * 0.1)),
            lifetime_limit=Decimal(str(amount)),
            currency="USD"
        )
    
    def _create_test_schedule(self):
        """Create test schedule"""
        return AdSchedule(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
    
    def _simulate_ad_performance(self, ad):
        """Simulate ad performance metrics"""
        # Add some realistic performance data
        for _ in range(1000):
            ad.add_impression()
        
        for _ in range(50):
            ad.add_click()
        
        for _ in range(10):
            ad.add_conversion()
        
        ad.add_spend(Decimal('250.0'))

if __name__ == "__main__":
    # Run integration tests
    print("🧪 Running System Integration Tests...")
    
    integration_test = TestSystemIntegration()
    
    # Run all integration tests
    print("\n" + "="*60)
    print("🚀 STARTING INTEGRATION TESTS")
    print("="*60)
    
    try:
        # Test 1: Complete campaign workflow
        integration_test.test_complete_campaign_workflow()
        
        print("\n" + "-"*60)
        
        # Test 2: Optimization system integration
        integration_test.test_optimization_system_integration()
        
        print("\n" + "-"*60)
        
        # Test 3: Budget management integration
        integration_test.test_budget_management_integration()
        
        print("\n" + "-"*60)
        
        # Test 4: Targeting integration
        integration_test.test_targeting_integration()
        
        print("\n" + "-"*60)
        
        # Test 5: Scheduling integration
        integration_test.test_scheduling_integration()
        
        print("\n" + "-"*60)
        
        # Test 6: Metrics integration
        integration_test.test_metrics_integration()
        
        print("\n" + "-"*60)
        
        # Test 7: Error handling integration
        integration_test.test_error_handling_integration()
        
        print("\n" + "-"*60)
        
        # Test 8: Performance integration
        integration_test.test_performance_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
