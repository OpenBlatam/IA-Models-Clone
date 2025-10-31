#!/usr/bin/env python3
"""
Comprehensive tests for value objects
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta

# Add current directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from domain.value_objects import (
    AdStatus, AdType, Platform, OptimizationStrategy, OptimizationLevel,
    Budget, TargetingCriteria, AdSchedule, AdMetrics
)

class TestEnums:
    """Test enum value objects"""
    
    def test_ad_status_enum(self):
        """Test AdStatus enum values"""
        assert AdStatus.DRAFT == "DRAFT"
        assert AdStatus.PENDING_REVIEW == "PENDING_REVIEW"
        assert AdStatus.APPROVED == "APPROVED"
        assert AdStatus.ACTIVE == "ACTIVE"
        assert AdStatus.PAUSED == "PAUSED"
        assert AdStatus.COMPLETED == "COMPLETED"
        assert AdStatus.REJECTED == "REJECTED"
    
    def test_ad_type_enum(self):
        """Test AdType enum values"""
        assert AdType.IMAGE == "IMAGE"
        assert AdType.VIDEO == "VIDEO"
        assert AdType.CAROUSEL == "CAROUSEL"
        assert AdType.STORY == "STORY"
        assert AdType.CANVAS == "CANVAS"
    
    def test_platform_enum(self):
        """Test Platform enum values"""
        assert Platform.FACEBOOK == "FACEBOOK"
        assert Platform.INSTAGRAM == "INSTAGRAM"
        assert Platform.GOOGLE_ADS == "GOOGLE_ADS"
        assert Platform.LINKEDIN == "LINKEDIN"
        assert Platform.TWITTER == "TWITTER"
    
    def test_optimization_strategy_enum(self):
        """Test OptimizationStrategy enum values"""
        assert OptimizationStrategy.PERFORMANCE == "PERFORMANCE"
        assert OptimizationStrategy.COST_EFFICIENCY == "COST_EFFICIENCY"
        assert OptimizationStrategy.REACH == "REACH"
        assert OptimizationStrategy.ENGAGEMENT == "ENGAGEMENT"
    
    def test_optimization_level_enum(self):
        """Test OptimizationLevel enum values"""
        assert OptimizationLevel.LOW == "LOW"
        assert OptimizationLevel.MEDIUM == "MEDIUM"
        assert OptimizationLevel.HIGH == "HIGH"
        assert OptimizationLevel.EXTREME == "EXTREME"

class TestBudget:
    """Test the Budget value object"""
    
    def test_budget_creation(self):
        """Test budget creation with valid data"""
        budget = Budget(
            amount=1000.0,
            daily_limit=Decimal('100.0'),
            lifetime_limit=Decimal('1000.0'),
            currency="USD"
        )
        
        assert budget.amount == 1000.0
        assert budget.daily_limit == Decimal('100.0')
        assert budget.lifetime_limit == Decimal('1000.0')
        assert budget.currency == "USD"
    
    def test_budget_validation(self):
        """Test budget validation rules"""
        # Test negative amount
        with pytest.raises(ValueError):
            Budget(
                amount=-100.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            )
        
        # Test negative daily limit
        with pytest.raises(ValueError):
            Budget(
                amount=1000.0,
                daily_limit=Decimal('-100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            )
        
        # Test empty currency
        with pytest.raises(ValueError):
            Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency=""
            )
    
    def test_budget_limits_validation(self):
        """Test budget limits validation"""
        # Test daily limit greater than lifetime limit
        with pytest.raises(ValueError):
            Budget(
                amount=1000.0,
                daily_limit=Decimal('2000.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            )
        
        # Test amount greater than lifetime limit
        with pytest.raises(ValueError):
            Budget(
                amount=2000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="USD"
            )
    
    def test_budget_currency_validation(self):
        """Test budget currency validation"""
        valid_currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
        
        for currency in valid_currencies:
            budget = Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency=currency
            )
            assert budget.currency == currency
        
        # Test invalid currency
        with pytest.raises(ValueError):
            Budget(
                amount=1000.0,
                daily_limit=Decimal('100.0'),
                lifetime_limit=Decimal('1000.0'),
                currency="INVALID"
            )
    
    def test_budget_operations(self):
        """Test budget operations"""
        budget = Budget(
            amount=1000.0,
            daily_limit=Decimal('100.0'),
            lifetime_limit=Decimal('1000.0'),
            currency="USD"
        )
        
        # Test spending
        budget.spend(Decimal('50.0'))
        assert budget.spent == Decimal('50.0')
        assert budget.remaining == Decimal('950.0')
        
        # Test daily spending
        budget.spend_daily(Decimal('25.0'))
        assert budget.daily_spent == Decimal('25.0')
        assert budget.daily_remaining == Decimal('75.0')
    
    def test_budget_limits(self):
        """Test budget spending limits"""
        budget = Budget(
            amount=1000.0,
            daily_limit=Decimal('100.0'),
            lifetime_limit=Decimal('1000.0'),
            currency="USD"
        )
        
        # Test exceeding daily limit
        with pytest.raises(ValueError):
            budget.spend_daily(Decimal('150.0'))
        
        # Test exceeding lifetime limit
        with pytest.raises(ValueError):
            budget.spend(Decimal('1500.0'))

class TestTargetingCriteria:
    """Test the TargetingCriteria value object"""
    
    def test_targeting_creation(self):
        """Test targeting creation with valid data"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US", "CA"],
            interests=["technology", "business"]
        )
        
        assert targeting.age_min == 18
        assert targeting.age_max == 65
        assert targeting.locations == ["US", "CA"]
        assert targeting.interests == ["technology", "business"]
    
    def test_targeting_validation(self):
        """Test targeting validation rules"""
        # Test invalid age range
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=25,
                age_max=18,  # min > max
                locations=["US"],
                interests=["technology"]
            )
        
        # Test negative age
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=-5,
                age_max=65,
                locations=["US"],
                interests=["technology"]
            )
        
        # Test empty locations
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=[],
                interests=["technology"]
            )
    
    def test_targeting_age_validation(self):
        """Test targeting age validation"""
        # Test valid age ranges
        valid_ranges = [(18, 25), (25, 35), (35, 50), (50, 65), (18, 65)]
        
        for min_age, max_age in valid_ranges:
            targeting = TargetingCriteria(
                age_min=min_age,
                age_max=max_age,
                locations=["US"],
                interests=["technology"]
            )
            assert targeting.age_min == min_age
            assert targeting.age_max == max_age
        
        # Test edge cases
        edge_cases = [(18, 18), (65, 65)]  # Same min and max
        
        for min_age, max_age in edge_cases:
            targeting = TargetingCriteria(
                age_min=min_age,
                age_max=max_age,
                locations=["US"],
                interests=["technology"]
            )
            assert targeting.age_min == min_age
            assert targeting.age_max == max_age
    
    def test_targeting_locations(self):
        """Test targeting locations"""
        # Test valid locations
        valid_locations = [["US"], ["US", "CA"], ["US", "CA", "UK"]]
        
        for locations in valid_locations:
            targeting = TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=locations,
                interests=["technology"]
            )
            assert targeting.locations == locations
        
        # Test location format validation
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=["US", ""],  # Empty location
                interests=["technology"]
            )
    
    def test_targeting_interests(self):
        """Test targeting interests"""
        # Test valid interests
        valid_interests = [
            ["technology"],
            ["technology", "business"],
            ["technology", "business", "entrepreneurship"]
        ]
        
        for interests in valid_interests:
            targeting = TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=["US"],
                interests=interests
            )
            assert targeting.interests == interests
        
        # Test interest format validation
        with pytest.raises(ValueError):
            TargetingCriteria(
                age_min=18,
                age_max=65,
                locations=["US"],
                interests=["technology", ""]  # Empty interest
            )
    
    def test_targeting_demographics(self):
        """Test targeting demographics"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"],
            demographics=["college_educated", "high_income"]
        )
        
        assert "college_educated" in targeting.demographics
        assert "high_income" in targeting.demographics
    
    def test_targeting_behaviors(self):
        """Test targeting behaviors"""
        targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"],
            behaviors=["frequent_travelers", "early_adopters"]
        )
        
        assert "frequent_travelers" in targeting.behaviors
        assert "early_adopters" in targeting.behaviors
    
    def test_targeting_is_valid(self):
        """Test targeting validation method"""
        # Valid targeting
        valid_targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=["US"],
            interests=["technology"]
        )
        assert valid_targeting.is_valid() is True
        
        # Invalid targeting (empty locations)
        invalid_targeting = TargetingCriteria(
            age_min=18,
            age_max=65,
            locations=[],
            interests=["technology"]
        )
        assert invalid_targeting.is_valid() is False

class TestAdSchedule:
    """Test the AdSchedule value object"""
    
    def test_schedule_creation(self):
        """Test schedule creation with valid data"""
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)
        
        schedule = AdSchedule(
            start_date=start_date,
            end_date=end_date,
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
        
        assert schedule.start_date == start_date
        assert schedule.end_date == end_date
        assert schedule.days_of_week == [0, 1, 2, 3, 4, 5, 6]
    
    def test_schedule_validation(self):
        """Test schedule validation rules"""
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)
        
        # Test end date before start date
        with pytest.raises(ValueError):
            AdSchedule(
                start_date=end_date,
                end_date=start_date,
                days_of_week=[0, 1, 2, 3, 4, 5, 6]
            )
        
        # Test invalid day of week
        with pytest.raises(ValueError):
            AdSchedule(
                start_date=start_date,
                end_date=end_date,
                days_of_week=[0, 1, 2, 3, 4, 5, 6, 7]  # Invalid day 7
            )
    
    def test_schedule_days_of_week(self):
        """Test schedule days of week validation"""
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)
        
        # Test valid day ranges
        valid_days = [
            [0],  # Monday only
            [0, 1, 2, 3, 4],  # Weekdays
            [5, 6],  # Weekend
            [0, 1, 2, 3, 4, 5, 6]  # All days
        ]
        
        for days in valid_days:
            schedule = AdSchedule(
                start_date=start_date,
                end_date=end_date,
                days_of_week=days
            )
            assert schedule.days_of_week == days
        
        # Test invalid days
        invalid_days = [[-1], [7], [0, 1, 2, 3, 4, 5, 6, 7]]
        
        for days in invalid_days:
            with pytest.raises(ValueError):
                AdSchedule(
                    start_date=start_date,
                    end_date=end_date,
                    days_of_week=days
                )
    
    def test_schedule_is_active(self):
        """Test schedule active status"""
        # Create schedule active now
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=1)
        end_date = now + timedelta(days=30)
        
        schedule = AdSchedule(
            start_date=start_date,
            end_date=end_date,
            days_of_week=[now.weekday()]  # Active today
        )
        
        assert schedule.is_active() is True
        
        # Create schedule not active (future)
        future_start = now + timedelta(days=1)
        future_schedule = AdSchedule(
            start_date=future_start,
            end_date=end_date,
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
        
        assert future_schedule.is_active() is False
        
        # Create schedule not active (past)
        past_end = now - timedelta(days=1)
        past_schedule = AdSchedule(
            start_date=start_date,
            end_date=past_end,
            days_of_week=[0, 1, 2, 3, 4, 5, 6]
        )
        
        assert past_schedule.is_active() is False
    
    def test_schedule_time_restrictions(self):
        """Test schedule time restrictions"""
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)
        
        # Schedule with time restrictions
        schedule = AdSchedule(
            start_date=start_date,
            end_date=end_date,
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            start_time="09:00",
            end_time="17:00"
        )
        
        assert schedule.start_time == "09:00"
        assert schedule.end_time == "17:00"
        
        # Test time format validation
        with pytest.raises(ValueError):
            AdSchedule(
                start_date=start_date,
                end_date=end_date,
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                start_time="25:00",  # Invalid time
                end_time="17:00"
            )

class TestAdMetrics:
    """Test the AdMetrics value object"""
    
    def test_metrics_creation(self):
        """Test metrics creation"""
        metrics = AdMetrics()
        
        assert metrics.impressions == 0
        assert metrics.clicks == 0
        assert metrics.conversions == 0
        assert metrics.spend == Decimal('0.0')
    
    def test_metrics_increment(self):
        """Test metrics increment methods"""
        metrics = AdMetrics()
        
        # Increment impressions
        metrics.add_impression()
        assert metrics.impressions == 1
        
        metrics.add_impression()
        assert metrics.impressions == 2
        
        # Increment clicks
        metrics.add_click()
        assert metrics.clicks == 1
        
        # Increment conversions
        metrics.add_conversion()
        assert metrics.conversions == 1
    
    def test_metrics_spend(self):
        """Test metrics spend tracking"""
        metrics = AdMetrics()
        
        # Add spend
        metrics.add_spend(Decimal('10.50'))
        assert metrics.spend == Decimal('10.50')
        
        metrics.add_spend(Decimal('25.75'))
        assert metrics.spend == Decimal('36.25')
    
    def test_metrics_calculations(self):
        """Test metrics calculations"""
        metrics = AdMetrics()
        
        # Add some data
        metrics.add_impression()
        metrics.add_impression()
        metrics.add_impression()
        metrics.add_click()
        metrics.add_click()
        metrics.add_conversion()
        
        # Test CTR calculation
        ctr = metrics.calculate_ctr()
        expected_ctr = 2.0 / 3.0  # 2 clicks / 3 impressions
        assert ctr == pytest.approx(expected_ctr)
        
        # Test conversion rate
        conv_rate = metrics.calculate_conversion_rate()
        expected_conv_rate = 1.0 / 2.0  # 1 conversion / 2 clicks
        assert conv_rate == pytest.approx(expected_conv_rate)
        
        # Test CPC calculation
        metrics.add_spend(Decimal('100.0'))
        cpc = metrics.calculate_cpc()
        expected_cpc = 100.0 / 2.0  # $100 spend / 2 clicks
        assert cpc == pytest.approx(expected_cpc)
    
    def test_metrics_edge_cases(self):
        """Test metrics edge cases"""
        metrics = AdMetrics()
        
        # Test CTR with no impressions
        ctr = metrics.calculate_ctr()
        assert ctr == 0.0
        
        # Test conversion rate with no clicks
        conv_rate = metrics.calculate_conversion_rate()
        assert conv_rate == 0.0
        
        # Test CPC with no clicks
        cpc = metrics.calculate_cpc()
        assert cpc == 0.0
        
        # Test CTR with no clicks
        metrics.add_impression()
        ctr = metrics.calculate_ctr()
        assert ctr == 0.0
    
    def test_metrics_reset(self):
        """Test metrics reset functionality"""
        metrics = AdMetrics()
        
        # Add some data
        metrics.add_impression()
        metrics.add_click()
        metrics.add_conversion()
        metrics.add_spend(Decimal('50.0'))
        
        # Verify data exists
        assert metrics.impressions == 1
        assert metrics.clicks == 1
        assert metrics.conversions == 1
        assert metrics.spend == Decimal('50.0')
        
        # Reset metrics
        metrics.reset()
        
        # Verify reset
        assert metrics.impressions == 0
        assert metrics.clicks == 0
        assert metrics.conversions == 0
        assert metrics.spend == Decimal('0.0')

if __name__ == "__main__":
    # Run tests
    print("🧪 Running Value Objects Tests...")
    
    # Test Enums
    print("\n1. Testing Enums...")
    enum_test = TestEnums()
    enum_test.test_ad_status_enum()
    enum_test.test_ad_type_enum()
    enum_test.test_platform_enum()
    enum_test.test_optimization_strategy_enum()
    enum_test.test_optimization_level_enum()
    print("✅ Enum tests passed")
    
    # Test Budget
    print("\n2. Testing Budget...")
    budget_test = TestBudget()
    budget_test.test_budget_creation()
    budget_test.test_budget_validation()
    budget_test.test_budget_limits_validation()
    budget_test.test_budget_currency_validation()
    budget_test.test_budget_operations()
    budget_test.test_budget_limits()
    print("✅ Budget tests passed")
    
    # Test TargetingCriteria
    print("\n3. Testing TargetingCriteria...")
    targeting_test = TestTargetingCriteria()
    targeting_test.test_targeting_creation()
    targeting_test.test_targeting_validation()
    targeting_test.test_targeting_age_validation()
    targeting_test.test_targeting_locations()
    targeting_test.test_targeting_interests()
    targeting_test.test_targeting_demographics()
    targeting_test.test_targeting_behaviors()
    targeting_test.test_targeting_is_valid()
    print("✅ TargetingCriteria tests passed")
    
    # Test AdSchedule
    print("\n4. Testing AdSchedule...")
    schedule_test = TestAdSchedule()
    schedule_test.test_schedule_creation()
    schedule_test.test_schedule_validation()
    schedule_test.test_schedule_days_of_week()
    schedule_test.test_schedule_is_active()
    schedule_test.test_schedule_time_restrictions()
    print("✅ AdSchedule tests passed")
    
    # Test AdMetrics
    print("\n5. Testing AdMetrics...")
    metrics_test = TestAdMetrics()
    metrics_test.test_metrics_creation()
    metrics_test.test_metrics_increment()
    metrics_test.test_metrics_spend()
    metrics_test.test_metrics_calculations()
    metrics_test.test_metrics_edge_cases()
    metrics_test.test_metrics_reset()
    print("✅ AdMetrics tests passed")
    
    print("\n🎉 All value objects tests completed successfully!")
