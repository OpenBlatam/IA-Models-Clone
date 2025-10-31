#!/usr/bin/env python3
"""
Basic test file for ADS system core functionality
"""

import sys
import os
from pathlib import Path
from decimal import Decimal

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_domain_entities():
    """Test domain entities creation and basic functionality"""
    try:
        from domain.entities import Ad, AdCampaign, AdGroup, TargetingCriteria, Budget, AdSchedule
        from domain.value_objects import AdStatus, AdType, Platform
        
        print("✅ Successfully imported domain entities")
        
        # Create basic entities
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US", "CA"],
            interests=["technology", "business"]
        )
        
        budget = Budget(
            amount=Decimal('1000.0'),
            daily_limit=Decimal('100.0'),
            lifetime_limit=Decimal('1000.0'),
            currency="USD"
        )
        
        from datetime import datetime, timezone
        schedule = AdSchedule(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
        
        ad = Ad(
            id="test-ad-1",
            name="Test Advertisement",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            status=AdStatus.DRAFT,
            targeting=targeting,
            budget=budget,
            schedule=schedule
        )
        
        campaign = AdCampaign(
            id="test-campaign-1",
            name="Test Campaign",
            description="Test campaign description",
            objective="AWARENESS"
        )
        
        ad_group = AdGroup(
            id="test-group-1",
            name="Test Ad Group",
            campaign_id="test-campaign-1"
        )
        
        print("✅ Successfully created all basic entities")
        print(f"   - Ad: {ad.name} ({ad.status})")
        print(f"   - Campaign: {campaign.name}")
        print(f"   - Ad Group: {ad_group.name}")
        
        # Test basic functionality
        campaign.add_ad(ad)
        ad_group.add_ad(ad)
        
        print("✅ Successfully added ad to campaign and ad group")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test domain entities: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_system():
    """Test optimization system functionality"""
    try:
        from optimization.factory import OptimizationFactory
        
        factory = OptimizationFactory()
        
        # Test creating optimizers
        performance_opt = factory.create_optimizer("performance")
        profiling_opt = factory.create_optimizer("profiling")
        gpu_opt = factory.create_optimizer("gpu")
        
        print("✅ Successfully created optimization factory")
        print(f"   - Performance optimizer: {performance_opt.name}")
        print(f"   - Profiling optimizer: {profiling_opt.name}")
        print(f"   - GPU optimizer: {gpu_opt.name}")
        
        # Test optimization context
        from optimization.base_optimizer import OptimizationContext, OptimizationStrategy, OptimizationLevel
        
        context = OptimizationContext(
            target_entity="ad",
            entity_id="test-ad-1",
            optimization_type=OptimizationStrategy.PERFORMANCE,
            level=OptimizationLevel.AGGRESSIVE
        )
        
        print("✅ Successfully created optimization context")
        
        # Test basic optimization
        import asyncio
        result = asyncio.run(performance_opt.optimize(context))
        print(f"✅ Performance optimization completed: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test optimization system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_value_objects():
    """Test value objects functionality"""
    try:
        from domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria
        
        # Test enum values
        print("✅ AdStatus values:", [status.value for status in AdStatus])
        print("✅ AdType values:", [ad_type.value for ad_type in AdType])
        print("✅ Platform values:", [platform.value for platform in Platform])
        
        # Test budget validation
        try:
            invalid_budget = Budget(amount=Decimal('-100'))
            print("❌ Budget validation failed - should not accept negative amount")
            return False
        except ValueError:
            print("✅ Budget validation working - correctly rejected negative amount")
        
        # Test targeting criteria validation
        try:
            invalid_targeting = TargetingCriteria(age_min=70, age_max=80)
            print("❌ Targeting validation failed - should not accept invalid age range")
            return False
        except ValueError:
            print("✅ Targeting validation working - correctly rejected invalid age range")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test value objects: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entity_serialization():
    """Test entity serialization to dictionaries"""
    try:
        from domain.entities import Ad, AdCampaign, AdGroup, TargetingCriteria, Budget, AdSchedule
        from domain.value_objects import AdStatus, AdType, Platform
        
        # Create entities
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"]
        )
        
        budget = Budget(
            amount=Decimal('500.0'),
            daily_limit=Decimal('50.0'),
            currency="USD"
        )
        
        from datetime import datetime, timezone
        schedule = AdSchedule(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            days_of_week=[0, 1, 2, 3, 4]
        )
        
        ad = Ad(
            id="serialization-test-ad",
            name="Serialization Test Ad",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            status=AdStatus.DRAFT,
            targeting=targeting,
            budget=budget,
            schedule=schedule
        )
        
        # Test serialization
        ad_dict = ad.to_dict()
        print("✅ Successfully serialized ad to dictionary")
        print(f"   - Ad ID: {ad_dict['id']}")
        print(f"   - Ad Name: {ad_dict['name']}")
        print(f"   - Status: {ad_dict['status']}")
        
        # Test that required fields are present
        required_fields = ['id', 'name', 'ad_type', 'platform', 'status', 'targeting', 'budget', 'schedule']
        for field in required_fields:
            if field not in ad_dict:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ All required fields present in serialized ad")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test entity serialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_basic_tests():
    """Run all basic tests"""
    print("🧪 Running ADS System Basic Tests...")
    print("=" * 60)
    
    tests = [
        ("Domain Entities Test", test_domain_entities),
        ("Optimization System Test", test_optimization_system),
        ("Value Objects Test", test_value_objects),
        ("Entity Serialization Test", test_entity_serialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! ADS system core functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
