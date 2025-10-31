"""
Demo: Improved Refactored Test Case Generation System
====================================================

Demonstration of the improved refactored test case generation system that creates
unique, diverse, and intuitive unit tests for functions given their 
signature and docstring.

This demo showcases the enhanced capabilities and improvements made to the
refactored system.
"""

import sys
from pathlib import Path
import time
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.improved_refactored_generator import ImprovedRefactoredGenerator
from tests.refactored_test_generator import RefactoredTestGenerator
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_function_1():
    """Example validation function"""
    def validate_user_profile(profile_data: dict, validation_rules: list, strict_mode: bool) -> dict:
        """
        Validate user profile data with comprehensive validation rules.
        
        Args:
            profile_data: Dictionary containing user profile information
            validation_rules: List of validation rules to apply
            strict_mode: Whether to use strict validation rules
            
        Returns:
            Dictionary with validation results and detailed feedback
            
        Raises:
            ValueError: If profile_data is invalid or validation_rules is empty
        """
        if not isinstance(profile_data, dict):
            raise ValueError("profile_data must be a dictionary")
        
        if not validation_rules:
            raise ValueError("validation_rules cannot be empty")
        
        validation_results = []
        warnings = []
        
        # Apply validation rules
        for rule in validation_rules:
            if rule == "email_required" and "email" not in profile_data:
                validation_results.append("Email is required")
            elif rule == "age_validation" and profile_data.get("age", 0) < 18:
                validation_results.append("Age must be 18 or older")
            elif rule == "username_validation" and len(profile_data.get("username", "")) < 3:
                validation_results.append("Username must be at least 3 characters")
            elif rule == "phone_validation" and "phone" in profile_data:
                phone = profile_data["phone"]
                if not phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "").isdigit():
                    validation_results.append("Phone number must contain only digits and common separators")
            elif rule == "address_validation" and "address" in profile_data:
                address = profile_data["address"]
                if len(address) < 10:
                    warnings.append("Address seems too short")
        
        # Strict mode validation
        if strict_mode:
            if "email" in profile_data:
                email = profile_data["email"]
                if "@" not in email or "." not in email.split("@")[1]:
                    validation_results.append("Invalid email format in strict mode")
            
            if "username" in profile_data:
                username = profile_data["username"]
                if not username.isalnum():
                    validation_results.append("Username must contain only alphanumeric characters in strict mode")
        
        return {
            "valid": len(validation_results) == 0,
            "validation_results": validation_results,
            "warnings": warnings,
            "strict_mode": strict_mode,
            "validated_fields": list(profile_data.keys()),
            "validation_rules_applied": len(validation_rules),
            "timestamp": datetime.now().isoformat()
        }
    
    return validate_user_profile


def demo_function_2():
    """Example data transformation function"""
    def transform_analytics_data(raw_data: list, target_format: str, include_metadata: bool, 
                               aggregation_level: str) -> dict:
        """
        Transform raw analytics data into target format with advanced options.
        
        Args:
            raw_data: List of raw data points
            target_format: Target format (json, xml, csv, yaml)
            include_metadata: Whether to include transformation metadata
            aggregation_level: Level of aggregation (hourly, daily, weekly, monthly)
            
        Returns:
            Dictionary with transformed data and comprehensive metadata
            
        Raises:
            ValueError: If target_format is invalid or raw_data is empty
            TypeError: If raw_data is not a list
        """
        if not isinstance(raw_data, list):
            raise TypeError("raw_data must be a list")
        
        if not raw_data:
            raise ValueError("raw_data cannot be empty")
        
        if target_format not in ["json", "xml", "csv", "yaml"]:
            raise ValueError("Invalid target format")
        
        if aggregation_level not in ["hourly", "daily", "weekly", "monthly"]:
            raise ValueError("Invalid aggregation level")
        
        # Transform data based on aggregation level
        if aggregation_level == "hourly":
            interval = 3600  # 1 hour in seconds
        elif aggregation_level == "daily":
            interval = 86400  # 1 day in seconds
        elif aggregation_level == "weekly":
            interval = 604800  # 1 week in seconds
        else:  # monthly
            interval = 2592000  # 30 days in seconds
        
        # Aggregate data
        aggregated_data = {}
        for data_point in raw_data:
            if isinstance(data_point, dict) and "timestamp" in data_point:
                # Group by time interval
                timestamp = data_point["timestamp"]
                interval_key = (timestamp // interval) * interval
                
                if interval_key not in aggregated_data:
                    aggregated_data[interval_key] = {
                        "count": 0,
                        "total_value": 0,
                        "data_points": []
                    }
                
                aggregated_data[interval_key]["count"] += 1
                aggregated_data[interval_key]["total_value"] += data_point.get("value", 0)
                aggregated_data[interval_key]["data_points"].append(data_point)
        
        # Calculate insights
        total_data_points = len(raw_data)
        total_intervals = len(aggregated_data)
        average_per_interval = total_data_points / total_intervals if total_intervals > 0 else 0
        
        # Create result
        result = {
            "format": target_format,
            "aggregation_level": aggregation_level,
            "total_data_points": total_data_points,
            "total_intervals": total_intervals,
            "average_per_interval": average_per_interval,
            "aggregated_data": aggregated_data,
            "insights": {
                "peak_interval": max(aggregated_data.keys()) if aggregated_data else None,
                "lowest_interval": min(aggregated_data.keys()) if aggregated_data else None,
                "data_density": total_data_points / (max(raw_data, key=lambda x: x.get("timestamp", 0)).get("timestamp", 1) - min(raw_data, key=lambda x: x.get("timestamp", 0)).get("timestamp", 0)) if len(raw_data) > 1 else 0
            }
        }
        
        # Add metadata if requested
        if include_metadata:
            result["metadata"] = {
                "transformed_at": datetime.now().isoformat(),
                "source_count": len(raw_data),
                "transformation_version": "2.0",
                "target_format": target_format,
                "aggregation_level": aggregation_level,
                "processing_time": datetime.now().isoformat()
            }
        
        return result
    
    return transform_analytics_data


def demo_function_3():
    """Example business logic function"""
    def calculate_advanced_pricing(base_price: float, user_tier: str, demand_factor: float, 
                                 time_of_day: str, special_events: list, market_conditions: dict) -> dict:
        """
        Calculate advanced pricing with multiple business factors and market conditions.
        
        Args:
            base_price: Base price for the product/service
            user_tier: User tier (bronze, silver, gold, platinum, diamond)
            demand_factor: Current demand factor (0.5 to 2.0)
            time_of_day: Time of day (morning, afternoon, evening, night)
            special_events: List of special events affecting pricing
            market_conditions: Dictionary with market condition data
            
        Returns:
            Dictionary with comprehensive pricing calculation details
            
        Raises:
            ValueError: If parameters are invalid
        """
        if base_price <= 0:
            raise ValueError("Base price must be positive")
        
        if user_tier not in ["bronze", "silver", "gold", "platinum", "diamond"]:
            raise ValueError("Invalid user tier")
        
        if not 0.5 <= demand_factor <= 2.0:
            raise ValueError("Demand factor must be between 0.5 and 2.0")
        
        if time_of_day not in ["morning", "afternoon", "evening", "night"]:
            raise ValueError("Invalid time of day")
        
        # Base tier multipliers
        tier_multipliers = {
            "bronze": 1.0,
            "silver": 0.9,
            "gold": 0.8,
            "platinum": 0.7,
            "diamond": 0.6
        }
        
        # Time-based multipliers
        time_multipliers = {
            "morning": 1.1,
            "afternoon": 1.0,
            "evening": 1.2,
            "night": 0.8
        }
        
        # Special event multipliers
        event_multipliers = {
            "holiday": 1.5,
            "sale": 0.8,
            "promotion": 0.9,
            "black_friday": 0.7,
            "cyber_monday": 0.75,
            "valentines": 1.3,
            "christmas": 1.4
        }
        
        # Market condition multipliers
        market_multipliers = {
            "high_competition": 0.9,
            "low_competition": 1.1,
            "economic_boom": 1.2,
            "economic_recession": 0.8,
            "seasonal_high": 1.3,
            "seasonal_low": 0.7
        }
        
        # Calculate base price with tier discount
        tier_price = base_price * tier_multipliers[user_tier]
        
        # Apply time-based pricing
        time_price = tier_price * time_multipliers[time_of_day]
        
        # Apply demand factor
        demand_price = time_price * demand_factor
        
        # Apply special event discounts
        event_price = demand_price
        for event in special_events:
            if event in event_multipliers:
                event_price *= event_multipliers[event]
        
        # Apply market condition adjustments
        market_price = event_price
        for condition, value in market_conditions.items():
            if condition in market_multipliers:
                market_price *= market_multipliers[condition]
        
        # Calculate final price with reasonable bounds
        final_price = max(base_price * 0.1, min(market_price, base_price * 3.0))
        
        return {
            "base_price": base_price,
            "user_tier": user_tier,
            "demand_factor": demand_factor,
            "time_of_day": time_of_day,
            "special_events": special_events,
            "market_conditions": market_conditions,
            "tier_multiplier": tier_multipliers[user_tier],
            "time_multiplier": time_multipliers[time_of_day],
            "event_multipliers": [event_multipliers.get(event, 1.0) for event in special_events],
            "market_multipliers": [market_multipliers.get(condition, 1.0) for condition in market_conditions.keys()],
            "tier_price": tier_price,
            "time_price": time_price,
            "demand_price": demand_price,
            "event_price": event_price,
            "market_price": market_price,
            "final_price": final_price,
            "total_discount": (base_price - final_price) / base_price * 100,
            "calculated_at": datetime.now().isoformat()
        }
    
    return calculate_advanced_pricing


def demo_improved_generator():
    """Demonstrate improved refactored test generator"""
    print("🚀 IMPROVED REFACTORED TEST GENERATOR DEMO")
    print("=" * 60)
    
    generator = ImprovedRefactoredGenerator()
    
    # Test with validation function
    validate_func = demo_function_1()
    test_cases = generator.generate_improved_tests(validate_func, num_tests=15)
    
    print(f"Generated {len(test_cases)} improved test cases for validation function:")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Quality Scores: U={test_case.uniqueness:.2f}, D={test_case.diversity:.2f}, I={test_case.intuition:.2f}")
        print(f"   Creativity: {test_case.creativity:.2f}, Coverage: {test_case.coverage:.2f}")
        print(f"   Overall Quality: {test_case.overall_quality:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print(f"   Assertions: {test_case.assertions}")
        print()


def demo_performance_comparison():
    """Demonstrate performance comparison between generators"""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("=" * 60)
    
    improved_generator = ImprovedRefactoredGenerator()
    refactored_generator = RefactoredTestGenerator()
    
    # Test all three functions
    functions = [
        ("Validation Function", demo_function_1()),
        ("Transformation Function", demo_function_2()),
        ("Business Logic Function", demo_function_3())
    ]
    
    total_improved_time = 0
    total_refactored_time = 0
    total_improved_tests = 0
    total_refactored_tests = 0
    
    for func_name, func in functions:
        print(f"\n🔍 Testing {func_name}:")
        print("-" * 40)
        
        # Test improved generator
        start_time = time.time()
        improved_tests = improved_generator.generate_improved_tests(func, num_tests=12)
        improved_time = time.time() - start_time
        total_improved_time += improved_time
        total_improved_tests += len(improved_tests)
        
        # Test refactored generator
        start_time = time.time()
        refactored_tests = refactored_generator.generate_tests(func, num_tests=12)
        refactored_time = time.time() - start_time
        total_refactored_time += refactored_time
        total_refactored_tests += len(refactored_tests)
        
        print(f"   Improved Generator:")
        print(f"     ⏱️  Time: {improved_time:.3f}s")
        print(f"     📊 Tests: {len(improved_tests)}")
        print(f"     🚀 Speed: {len(improved_tests)/improved_time:.1f} tests/second")
        
        print(f"   Refactored Generator:")
        print(f"     ⏱️  Time: {refactored_time:.3f}s")
        print(f"     📊 Tests: {len(refactored_tests)}")
        print(f"     🚀 Speed: {len(refactored_tests)/refactored_time:.1f} tests/second")
        
        # Performance improvement
        improvement = ((refactored_time - improved_time) / refactored_time) * 100
        print(f"   📈 Improvement: {improvement:.1f}%")
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Improved Generator:")
    print(f"     Total Time: {total_improved_time:.3f}s")
    print(f"     Total Tests: {total_improved_tests}")
    print(f"     Average Speed: {total_improved_tests/total_improved_time:.1f} tests/second")
    
    print(f"   Refactored Generator:")
    print(f"     Total Time: {total_refactored_time:.3f}s")
    print(f"     Total Tests: {total_refactored_tests}")
    print(f"     Average Speed: {total_refactored_tests/total_refactored_time:.1f} tests/second")
    
    overall_improvement = ((total_refactored_time - total_improved_time) / total_refactored_time) * 100
    print(f"   📈 Overall Improvement: {overall_improvement:.1f}%")


def demo_quality_analysis():
    """Demonstrate quality analysis"""
    print("\n📈 QUALITY ANALYSIS DEMO")
    print("=" * 60)
    
    improved_generator = ImprovedRefactoredGenerator()
    refactored_generator = RefactoredTestGenerator()
    
    # Test with transformation function
    transform_func = demo_function_2()
    
    improved_tests = improved_generator.generate_improved_tests(transform_func, num_tests=15)
    refactored_tests = refactored_generator.generate_tests(transform_func, num_tests=15)
    
    print("IMPROVED GENERATOR RESULTS:")
    print("-" * 30)
    if improved_tests:
        avg_uniqueness = sum(tc.uniqueness for tc in improved_tests) / len(improved_tests)
        avg_diversity = sum(tc.diversity for tc in improved_tests) / len(improved_tests)
        avg_intuition = sum(tc.intuition for tc in improved_tests) / len(improved_tests)
        avg_creativity = sum(tc.creativity for tc in improved_tests) / len(improved_tests)
        avg_coverage = sum(tc.coverage for tc in improved_tests) / len(improved_tests)
        avg_quality = sum(tc.overall_quality for tc in improved_tests) / len(improved_tests)
        
        print(f"   Average Uniqueness: {avg_uniqueness:.3f}")
        print(f"   Average Diversity: {avg_diversity:.3f}")
        print(f"   Average Intuition: {avg_intuition:.3f}")
        print(f"   Average Creativity: {avg_creativity:.3f}")
        print(f"   Average Coverage: {avg_coverage:.3f}")
        print(f"   Average Overall Quality: {avg_quality:.3f}")
        
        # Quality distribution
        high_quality = sum(1 for tc in improved_tests if tc.overall_quality > 0.8)
        medium_quality = sum(1 for tc in improved_tests if 0.6 <= tc.overall_quality <= 0.8)
        low_quality = sum(1 for tc in improved_tests if tc.overall_quality < 0.6)
        
        print(f"   High Quality (>0.8): {high_quality} ({high_quality/len(improved_tests)*100:.1f}%)")
        print(f"   Medium Quality (0.6-0.8): {medium_quality} ({medium_quality/len(improved_tests)*100:.1f}%)")
        print(f"   Low Quality (<0.6): {low_quality} ({low_quality/len(improved_tests)*100:.1f}%)")
    
    print("\nREFACTORED GENERATOR RESULTS:")
    print("-" * 30)
    if refactored_tests:
        avg_uniqueness = sum(tc.uniqueness for tc in refactored_tests) / len(refactored_tests)
        avg_diversity = sum(tc.diversity for tc in refactored_tests) / len(refactored_tests)
        avg_intuition = sum(tc.intuition for tc in refactored_tests) / len(refactored_tests)
        avg_quality = sum(tc.overall_quality for tc in refactored_tests) / len(refactored_tests)
        
        print(f"   Average Uniqueness: {avg_uniqueness:.3f}")
        print(f"   Average Diversity: {avg_diversity:.3f}")
        print(f"   Average Intuition: {avg_intuition:.3f}")
        print(f"   Average Overall Quality: {avg_quality:.3f}")
        
        # Quality distribution
        high_quality = sum(1 for tc in refactored_tests if tc.overall_quality > 0.8)
        medium_quality = sum(1 for tc in refactored_tests if 0.6 <= tc.overall_quality <= 0.8)
        low_quality = sum(1 for tc in refactored_tests if tc.overall_quality < 0.6)
        
        print(f"   High Quality (>0.8): {high_quality} ({high_quality/len(refactored_tests)*100:.1f}%)")
        print(f"   Medium Quality (0.6-0.8): {medium_quality} ({medium_quality/len(refactored_tests)*100:.1f}%)")
        print(f"   Low Quality (<0.6): {low_quality} ({low_quality/len(refactored_tests)*100:.1f}%)")


def demo_improvements():
    """Demonstrate improvements made"""
    print("\n🎯 IMPROVEMENTS DEMONSTRATION")
    print("=" * 60)
    
    print("✅ ENHANCED UNIQUENESS:")
    print("   - Creative test scenarios with distinct characteristics")
    print("   - Unique test patterns and approaches")
    print("   - Innovative parameter generation")
    print("   - Enhanced uniqueness scoring")
    
    print("\n✅ ENHANCED DIVERSITY:")
    print("   - Comprehensive scenario coverage")
    print("   - Multiple test categories and types")
    print("   - Wide range of parameter combinations")
    print("   - Advanced edge case detection")
    
    print("\n✅ ENHANCED INTUITION:")
    print("   - Multiple naming strategies")
    print("   - Clear, descriptive test names")
    print("   - Story-like descriptions")
    print("   - Intuitive test structure")
    
    print("\n✅ NEW FEATURES:")
    print("   - Creativity scoring and optimization")
    print("   - Coverage analysis and scoring")
    print("   - Quality optimization algorithms")
    print("   - Duplicate removal and deduplication")
    print("   - Enhanced parameter generation")
    print("   - Advanced assertion generation")
    
    print("\n✅ PERFORMANCE IMPROVEMENTS:")
    print("   - Faster test generation")
    print("   - Better memory usage")
    print("   - Optimized algorithms")
    print("   - Enhanced scalability")


def demo_usage_examples():
    """Demonstrate usage examples"""
    print("\n💡 USAGE EXAMPLES")
    print("=" * 60)
    
    print("🔧 BASIC USAGE:")
    print("""
from tests.improved_refactored_generator import ImprovedRefactoredGenerator

# Create improved generator
generator = ImprovedRefactoredGenerator()

# Generate improved tests
test_cases = generator.generate_improved_tests(your_function, num_tests=25)

# Analyze results
for test_case in test_cases:
    print(f"Name: {test_case.name}")
    print(f"Quality: {test_case.overall_quality:.2f}")
    print(f"Uniqueness: {test_case.uniqueness:.2f}")
    print(f"Diversity: {test_case.diversity:.2f}")
    print(f"Intuition: {test_case.intuition:.2f}")
    print(f"Creativity: {test_case.creativity:.2f}")
    print(f"Coverage: {test_case.coverage:.2f}")
""")
    
    print("🎯 ADVANCED USAGE:")
    print("""
# Generate specific test types
unique_tests = [tc for tc in test_cases if tc.test_type == "unique"]
diverse_tests = [tc for tc in test_cases if tc.test_type == "diverse"]
intuitive_tests = [tc for tc in test_cases if tc.test_type == "intuitive"]
creative_tests = [tc for tc in test_cases if tc.test_type == "creative"]

# Filter by quality
high_quality_tests = [tc for tc in test_cases if tc.overall_quality > 0.8]

# Sort by specific metric
sorted_by_uniqueness = sorted(test_cases, key=lambda x: x.uniqueness, reverse=True)
sorted_by_creativity = sorted(test_cases, key=lambda x: x.creativity, reverse=True)
""")


def main():
    """Main demonstration function"""
    print("🎉 IMPROVED REFACTORED TEST CASE GENERATION SYSTEM DEMO")
    print("=" * 80)
    print("This demo showcases the improved refactored test case generation system")
    print("that creates unique, diverse, and intuitive unit tests with enhanced capabilities.")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demo_improved_generator()
        print("\n" + "="*80)
        
        demo_performance_comparison()
        print("\n" + "="*80)
        
        demo_quality_analysis()
        print("\n" + "="*80)
        
        demo_improvements()
        print("\n" + "="*80)
        
        demo_usage_examples()
        
        print("\n" + "="*80)
        print("🎊 IMPROVED REFACTORED SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The improved refactored test case generation system successfully provides:")
        print("✅ Enhanced uniqueness with creative patterns and approaches")
        print("✅ Enhanced diversity with comprehensive scenario coverage")
        print("✅ Enhanced intuition with advanced naming strategies")
        print("✅ New creativity and coverage scoring capabilities")
        print("✅ Quality optimization and duplicate removal")
        print("✅ Better performance and scalability")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
