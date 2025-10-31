#!/usr/bin/env python3
"""
Simple test file to verify ADS system functionality
"""

import sys
import os
from pathlib import Path
from decimal import Decimal

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that we can import the main modules"""
    try:
        from domain.entities import Ad, AdCampaign, AdGroup
        print("✅ Successfully imported domain entities")
        return True
    except ImportError as e:
        print(f"❌ Failed to import domain entities: {e}")
        return False

def test_optimization():
    """Test that we can import optimization modules"""
    try:
        from optimization.factory import OptimizationFactory
        print("✅ Successfully imported optimization factory")
        return True
    except ImportError as e:
        print(f"❌ Failed to import optimization factory: {e}")
        return False

def test_training():
    """Test that we can import training modules"""
    try:
        from training.base_trainer import BaseTrainer
        print("✅ Successfully imported base trainer")
        return True
    except ImportError as e:
        print(f"❌ Failed to import base trainer: {e}")
        return False

def test_entity_creation():
    """Test creating basic entities"""
    try:
        from domain.entities import Ad, AdCampaign, AdGroup, TargetingCriteria, Budget, AdSchedule
        from domain.value_objects import AdStatus, AdType, Platform
        
        # Create basic entities
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US", "CA"],
            interests=["technology", "business"]
        )
        
        budget = Budget(
            amount=1000.0,
            daily_limit=Decimal('100.0'),
            lifetime_limit=Decimal('1000.0'),
            currency="USD"
        )
        
        schedule = AdSchedule(
            start_date="2024-01-01",
            end_date="2024-12-31",
            active_hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            active_days=[0, 1, 2, 3, 4, 5, 6]
        )
        
        ad = Ad(
            id="test-ad-1",
            name="Test Advertisement",
            ad_type=AdType.BANNER,
            platform=Platform.FACEBOOK,
            status=AdStatus.DRAFT,
            targeting_criteria=targeting,
            budget=budget,
            schedule=schedule
        )
        
        campaign = AdCampaign(
            id="test-campaign-1",
            name="Test Campaign",
            description="Test campaign description"
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
        return True
        
    except Exception as e:
        print(f"❌ Failed to create entities: {e}")
        return False

def test_optimization_factory():
    """Test optimization factory functionality"""
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
        return True
        
    except Exception as e:
        print(f"❌ Failed to test optimization factory: {e}")
        return False

def test_training_system():
    """Test training system functionality"""
    try:
        from training.base_trainer import BaseTrainer
        from training.model_trainer import ModelTrainer
        from training.data_trainer import DataTrainer
        from training.performance_trainer import PerformanceTrainer
        
        # Test creating trainers
        model_trainer = ModelTrainer("model-trainer")
        data_trainer = DataTrainer("data-trainer")
        perf_trainer = PerformanceTrainer("perf-trainer")
        
        print("✅ Successfully created training system")
        print(f"   - Model trainer: {model_trainer.name}")
        print(f"   - Data trainer: {data_trainer.name}")
        print(f"   - Performance trainer: {perf_trainer.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test training system: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🧪 Running ADS System Tests...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Optimization Tests", test_optimization),
        ("Training Tests", test_training),
        ("Entity Creation Tests", test_entity_creation),
        ("Optimization Factory Tests", test_optimization_factory),
        ("Training System Tests", test_training_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ⚠️  {test_name} failed")
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ADS system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
