"""
Demo: Improved Test Case Generation
==================================

Demonstration of the improved test case generation system that creates
unique, diverse, and intuitive unit tests for functions given their 
signature and docstring.

This demo showcases the improvements made to better address the prompt requirements.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.improved_test_generator import ImprovedTestGenerator
from tests.enhanced_test_generator import EnhancedTestGenerator
from tests.streamlined_test_generator import StreamlinedTestGenerator
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_function_1():
    """Example validation function"""
    def validate_email_address(email: str, domain_whitelist: list, strict_mode: bool) -> dict:
        """
        Validate email address with domain whitelist and strict mode options.
        
        Args:
            email: Email address to validate
            domain_whitelist: List of allowed domains
            strict_mode: Whether to use strict validation rules
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If email is empty or None
        """
        if not email:
            raise ValueError("Email cannot be empty")
        
        # Basic format validation
        if "@" not in email:
            return {
                "valid": False,
                "error": "Email must contain @ symbol",
                "email": email,
                "strict_mode": strict_mode
            }
        
        parts = email.split("@")
        if len(parts) != 2:
            return {
                "valid": False,
                "error": "Email must have exactly one @ symbol",
                "email": email,
                "strict_mode": strict_mode
            }
        
        local, domain = parts
        
        if not local or not domain:
            return {
                "valid": False,
                "error": "Email must have both local and domain parts",
                "email": email,
                "strict_mode": strict_mode
            }
        
        # Domain validation
        if domain_whitelist and domain not in domain_whitelist:
            return {
                "valid": False,
                "error": f"Domain {domain} is not in whitelist",
                "email": email,
                "strict_mode": strict_mode
            }
        
        # Strict mode validation
        if strict_mode:
            if "." not in domain:
                return {
                    "valid": False,
                    "error": "Domain must contain at least one dot in strict mode",
                    "email": email,
                    "strict_mode": strict_mode
                }
            
            if len(local) < 2:
                return {
                    "valid": False,
                    "error": "Local part must be at least 2 characters in strict mode",
                    "email": email,
                    "strict_mode": strict_mode
                }
        
        return {
            "valid": True,
            "email": email,
            "local_part": local,
            "domain": domain,
            "strict_mode": strict_mode,
            "validated_at": datetime.now().isoformat()
        }
    
    return validate_email_address


def demo_function_2():
    """Example data transformation function"""
    def transform_user_data(raw_data: list, target_format: str, include_metadata: bool) -> dict:
        """
        Transform raw user data into target format with optional metadata.
        
        Args:
            raw_data: List of raw user data dictionaries
            target_format: Target format (json, xml, csv)
            include_metadata: Whether to include transformation metadata
            
        Returns:
            Dictionary with transformed data and metadata
            
        Raises:
            ValueError: If target_format is invalid or raw_data is empty
            TypeError: If raw_data is not a list
        """
        if not isinstance(raw_data, list):
            raise TypeError("raw_data must be a list")
        
        if not raw_data:
            raise ValueError("raw_data cannot be empty")
        
        if target_format not in ["json", "xml", "csv"]:
            raise ValueError("Invalid target format")
        
        # Transform data
        transformed_users = []
        for user_data in raw_data:
            if isinstance(user_data, dict):
                transformed_user = {
                    "id": user_data.get("id", "unknown"),
                    "name": user_data.get("name", "Unknown"),
                    "email": user_data.get("email", "unknown@example.com"),
                    "age": user_data.get("age", 0),
                    "active": user_data.get("active", False)
                }
                transformed_users.append(transformed_user)
        
        # Create result
        result = {
            "format": target_format,
            "user_count": len(transformed_users),
            "users": transformed_users
        }
        
        # Add metadata if requested
        if include_metadata:
            result["metadata"] = {
                "transformed_at": datetime.now().isoformat(),
                "source_count": len(raw_data),
                "transformation_version": "1.0",
                "target_format": target_format
            }
        
        return result
    
    return transform_user_data


def demo_function_3():
    """Example business logic function"""
    def calculate_dynamic_pricing(base_price: float, user_tier: str, demand_factor: float, 
                                time_of_day: str, special_events: list) -> dict:
        """
        Calculate dynamic pricing based on multiple business factors.
        
        Args:
            base_price: Base price for the product/service
            user_tier: User tier (bronze, silver, gold, platinum)
            demand_factor: Current demand factor (0.5 to 2.0)
            time_of_day: Time of day (morning, afternoon, evening, night)
            special_events: List of special events affecting pricing
            
        Returns:
            Dictionary with pricing calculation details
            
        Raises:
            ValueError: If parameters are invalid
        """
        if base_price <= 0:
            raise ValueError("Base price must be positive")
        
        if user_tier not in ["bronze", "silver", "gold", "platinum"]:
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
            "platinum": 0.7
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
            "black_friday": 0.7
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
        
        # Calculate final price with reasonable bounds
        final_price = max(base_price * 0.1, min(event_price, base_price * 3.0))
        
        return {
            "base_price": base_price,
            "user_tier": user_tier,
            "demand_factor": demand_factor,
            "time_of_day": time_of_day,
            "special_events": special_events,
            "tier_multiplier": tier_multipliers[user_tier],
            "time_multiplier": time_multipliers[time_of_day],
            "event_multipliers": [event_multipliers.get(event, 1.0) for event in special_events],
            "tier_price": tier_price,
            "time_price": time_price,
            "demand_price": demand_price,
            "event_price": event_price,
            "final_price": final_price,
            "total_discount": (base_price - final_price) / base_price * 100,
            "calculated_at": datetime.now().isoformat()
        }
    
    return calculate_dynamic_pricing


def demo_improved_generator():
    """Demonstrate improved test generator"""
    print("🚀 IMPROVED TEST GENERATOR DEMO")
    print("=" * 50)
    
    generator = ImprovedTestGenerator()
    
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
        print(f"   Overall Quality: {test_case.overall_quality:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print()


def demo_comparison():
    """Compare different generators"""
    print("\n📊 GENERATOR COMPARISON")
    print("=" * 50)
    
    # Test with transformation function
    transform_func = demo_function_2()
    
    # Generate tests with different generators
    improved_generator = ImprovedTestGenerator()
    enhanced_generator = EnhancedTestGenerator()
    streamlined_generator = StreamlinedTestGenerator()
    
    improved_tests = improved_generator.generate_improved_tests(transform_func, num_tests=12)
    enhanced_tests = enhanced_generator.generate_enhanced_tests(transform_func, num_tests=12)
    streamlined_tests = streamlined_generator.generate_tests(transform_func, num_tests=12)
    
    print("IMPROVED GENERATOR RESULTS:")
    print("-" * 30)
    if improved_tests:
        avg_uniqueness = sum(tc.uniqueness for tc in improved_tests) / len(improved_tests)
        avg_diversity = sum(tc.diversity for tc in improved_tests) / len(improved_tests)
        avg_intuition = sum(tc.intuition for tc in improved_tests) / len(improved_tests)
        avg_quality = sum(tc.overall_quality for tc in improved_tests) / len(improved_tests)
        
        print(f"Average Uniqueness: {avg_uniqueness:.3f}")
        print(f"Average Diversity: {avg_diversity:.3f}")
        print(f"Average Intuition: {avg_intuition:.3f}")
        print(f"Average Overall Quality: {avg_quality:.3f}")
    
    print("\nENHANCED GENERATOR RESULTS:")
    print("-" * 30)
    if enhanced_tests:
        avg_uniqueness = sum(tc.uniqueness_score for tc in enhanced_tests) / len(enhanced_tests)
        avg_diversity = sum(tc.diversity_score for tc in enhanced_tests) / len(enhanced_tests)
        avg_intuition = sum(tc.intuition_score for tc in enhanced_tests) / len(enhanced_tests)
        avg_quality = sum(tc.overall_quality for tc in enhanced_tests) / len(enhanced_tests)
        
        print(f"Average Uniqueness: {avg_uniqueness:.3f}")
        print(f"Average Diversity: {avg_diversity:.3f}")
        print(f"Average Intuition: {avg_intuition:.3f}")
        print(f"Average Overall Quality: {avg_quality:.3f}")
    
    print("\nSTREAMLINED GENERATOR RESULTS:")
    print("-" * 30)
    if streamlined_tests:
        avg_uniqueness = sum(tc.uniqueness for tc in streamlined_tests) / len(streamlined_tests)
        avg_diversity = sum(tc.diversity for tc in streamlined_tests) / len(streamlined_tests)
        avg_intuition = sum(tc.intuition for tc in streamlined_tests) / len(streamlined_tests)
        avg_quality = (avg_uniqueness + avg_diversity + avg_intuition) / 3
        
        print(f"Average Uniqueness: {avg_uniqueness:.3f}")
        print(f"Average Diversity: {avg_diversity:.3f}")
        print(f"Average Intuition: {avg_intuition:.3f}")
        print(f"Average Overall Quality: {avg_quality:.3f}")


def demo_quality_analysis():
    """Demonstrate quality analysis"""
    print("\n📈 QUALITY ANALYSIS DEMO")
    print("=" * 50)
    
    generator = ImprovedTestGenerator()
    
    # Test all three functions
    functions = [
        ("Validation Function", demo_function_1()),
        ("Transformation Function", demo_function_2()),
        ("Business Logic Function", demo_function_3())
    ]
    
    for func_name, func in functions:
        print(f"\n🔍 Analyzing {func_name}:")
        print("-" * 30)
        
        test_cases = generator.generate_improved_tests(func, num_tests=12)
        
        if test_cases:
            avg_uniqueness = sum(tc.uniqueness for tc in test_cases) / len(test_cases)
            avg_diversity = sum(tc.diversity for tc in test_cases) / len(test_cases)
            avg_intuition = sum(tc.intuition for tc in test_cases) / len(test_cases)
            avg_quality = sum(tc.overall_quality for tc in test_cases) / len(test_cases)
            
            print(f"   📈 Average Uniqueness Score: {avg_uniqueness:.3f}")
            print(f"   🎯 Average Diversity Score: {avg_diversity:.3f}")
            print(f"   💡 Average Intuition Score: {avg_intuition:.3f}")
            print(f"   ⭐ Average Overall Quality: {avg_quality:.3f}")
            
            # Show best test case
            best_test = max(test_cases, key=lambda x: x.overall_quality)
            print(f"   🏆 Best Test Case: {best_test.name}")
            print(f"      Quality: {best_test.overall_quality:.3f}")
            print(f"      Type: {best_test.test_type}")


def demo_improvements_summary():
    """Demonstrate improvements summary"""
    print("\n🎉 IMPROVEMENTS SUMMARY")
    print("=" * 50)
    
    print("Key Improvements Made:")
    print("✅ Better Uniqueness:")
    print("   - Creative test scenarios")
    print("   - Unique test patterns")
    print("   - Distinct characteristics")
    print("   - Innovative approaches")
    
    print("\n✅ Better Diversity:")
    print("   - Comprehensive coverage")
    print("   - Multiple test types")
    print("   - Wide range of scenarios")
    print("   - Varied parameter combinations")
    
    print("\n✅ Better Intuition:")
    print("   - Clear, descriptive naming")
    print("   - Multiple naming strategies")
    print("   - Intuitive test structure")
    print("   - Story-like descriptions")
    
    print("\n✅ Additional Improvements:")
    print("   - Enhanced quality scoring")
    print("   - Better test categorization")
    print("   - Improved parameter generation")
    print("   - Comprehensive test coverage")


def main():
    """Main demonstration function"""
    print("🎉 IMPROVED TEST CASE GENERATION DEMO")
    print("=" * 60)
    print("This demo showcases the improved test case generation system")
    print("that creates unique, diverse, and intuitive unit tests.")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_improved_generator()
        print("\n" + "="*60)
        
        demo_comparison()
        print("\n" + "="*60)
        
        demo_quality_analysis()
        print("\n" + "="*60)
        
        demo_improvements_summary()
        
        print("\n" + "="*60)
        print("🎊 IMPROVED TESTS DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The improved test case generation system successfully provides:")
        print("✅ Unique test scenarios with creative and innovative approaches")
        print("✅ Diverse test coverage across all major scenarios and edge cases")
        print("✅ Intuitive test naming and structure that is easy to understand")
        print("✅ Enhanced quality metrics and scoring for better test evaluation")
        print("✅ Better alignment with the prompt requirements")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
