"""
Demo: Enhanced Test Case Generation System
=========================================

Demonstration of the enhanced test case generation system that creates
unique, diverse, and intuitive unit tests for functions given their 
signature and docstring.

This demo showcases:
- Unique test scenarios with varied approaches
- Diverse test cases covering wide range of scenarios  
- Intuitive test naming and structure
- Advanced function analysis and pattern recognition
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.intelligent_test_generator import IntelligentTestGenerator, TestDiversity, TestIntuition
from tests.unique_diverse_test_generator import UniqueDiverseTestGenerator
from tests.comprehensive_test_generator import ComprehensiveTestGenerator
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_function_1():
    """Example function for validation testing"""
    def validate_user_profile(profile_data: dict, required_fields: list) -> dict:
        """
        Validate user profile data against required fields and business rules.
        
        Args:
            profile_data: Dictionary containing user profile information
            required_fields: List of required field names
            
        Returns:
            Dictionary with validation results and any errors
            
        Raises:
            ValueError: If profile_data is not a dictionary
            TypeError: If required_fields is not a list
        """
        if not isinstance(profile_data, dict):
            raise ValueError("profile_data must be a dictionary")
        
        if not isinstance(required_fields, list):
            raise TypeError("required_fields must be a list")
        
        errors = []
        warnings = []
        
        # Check required fields
        for field in required_fields:
            if field not in profile_data:
                errors.append(f"Required field '{field}' is missing")
            elif not profile_data[field]:
                warnings.append(f"Field '{field}' is empty")
        
        # Business rule validations
        if "email" in profile_data:
            email = profile_data["email"]
            if "@" not in email or "." not in email.split("@")[1]:
                errors.append("Invalid email format")
        
        if "age" in profile_data:
            age = profile_data["age"]
            if not isinstance(age, int) or age < 0 or age > 150:
                errors.append("Age must be a positive integer between 0 and 150")
        
        if "username" in profile_data:
            username = profile_data["username"]
            if len(username) < 3 or len(username) > 30:
                errors.append("Username must be between 3 and 30 characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_fields": list(profile_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    return validate_user_profile


def demo_function_2():
    """Example function for data transformation testing"""
    def transform_video_metadata(raw_metadata: dict, target_format: str) -> dict:
        """
        Transform video metadata from raw format to target format.
        
        Args:
            raw_metadata: Raw video metadata dictionary
            target_format: Target format (json, xml, yaml)
            
        Returns:
            Transformed metadata in target format
            
        Raises:
            ValueError: If target_format is not supported
            KeyError: If required metadata fields are missing
        """
        if target_format not in ["json", "xml", "yaml"]:
            raise ValueError("Unsupported target format")
        
        required_fields = ["title", "duration", "resolution", "format"]
        for field in required_fields:
            if field not in raw_metadata:
                raise KeyError(f"Required field '{field}' is missing")
        
        # Transform metadata
        transformed = {
            "video_info": {
                "title": raw_metadata["title"],
                "duration_seconds": raw_metadata["duration"],
                "resolution": raw_metadata["resolution"],
                "format": raw_metadata["format"]
            },
            "processing_info": {
                "transformed_at": datetime.now().isoformat(),
                "target_format": target_format,
                "source_fields": len(raw_metadata)
            }
        }
        
        # Add optional fields if present
        if "description" in raw_metadata:
            transformed["video_info"]["description"] = raw_metadata["description"]
        
        if "tags" in raw_metadata:
            transformed["video_info"]["tags"] = raw_metadata["tags"]
        
        if "created_by" in raw_metadata:
            transformed["processing_info"]["created_by"] = raw_metadata["created_by"]
        
        return transformed
    
    return transform_video_metadata


def demo_function_3():
    """Example function for business logic testing"""
    def calculate_subscription_pricing(user_tier: str, features: list, duration_months: int) -> dict:
        """
        Calculate subscription pricing based on user tier, features, and duration.
        
        Args:
            user_tier: User tier (basic, premium, enterprise)
            features: List of requested features
            duration_months: Subscription duration in months
            
        Returns:
            Dictionary with pricing calculation results
            
        Raises:
            ValueError: If user_tier is invalid or duration is negative
        """
        if user_tier not in ["basic", "premium", "enterprise"]:
            raise ValueError("Invalid user tier")
        
        if duration_months <= 0:
            raise ValueError("Duration must be positive")
        
        # Base pricing by tier
        base_prices = {
            "basic": 9.99,
            "premium": 19.99,
            "enterprise": 49.99
        }
        
        # Feature pricing
        feature_prices = {
            "video_generation": 5.00,
            "ai_voice": 3.00,
            "custom_avatars": 2.00,
            "analytics": 4.00,
            "api_access": 10.00,
            "priority_support": 8.00
        }
        
        # Duration discounts
        duration_discounts = {
            1: 0.0,    # No discount
            3: 0.05,   # 5% discount
            6: 0.10,   # 10% discount
            12: 0.15,  # 15% discount
            24: 0.20   # 20% discount
        }
        
        # Calculate base price
        base_price = base_prices[user_tier]
        
        # Calculate feature costs
        feature_cost = sum(feature_prices.get(feature, 0) for feature in features)
        
        # Calculate subtotal
        subtotal = base_price + feature_cost
        
        # Apply duration discount
        discount_rate = 0.0
        for duration, rate in sorted(duration_discounts.items(), reverse=True):
            if duration_months >= duration:
                discount_rate = rate
                break
        
        discount_amount = subtotal * discount_rate
        final_price = subtotal - discount_amount
        
        # Calculate monthly price
        monthly_price = final_price / duration_months
        
        return {
            "user_tier": user_tier,
            "features": features,
            "duration_months": duration_months,
            "base_price": base_price,
            "feature_cost": feature_cost,
            "subtotal": subtotal,
            "discount_rate": discount_rate,
            "discount_amount": discount_amount,
            "final_price": final_price,
            "monthly_price": monthly_price,
            "calculated_at": datetime.now().isoformat()
        }
    
    return calculate_subscription_pricing


def demo_intelligent_generator():
    """Demonstrate intelligent test generation"""
    print("🧠 INTELLIGENT TEST GENERATOR DEMO")
    print("=" * 50)
    
    generator = IntelligentTestGenerator(
        diversity_level=TestDiversity.COMPREHENSIVE,
        intuition_level=TestIntuition.DESCRIPTIVE
    )
    
    # Test with validation function
    validate_func = demo_function_1()
    test_cases = generator.generate_intelligent_tests(validate_func, num_tests=8)
    
    print(f"Generated {len(test_cases)} intelligent test cases for validation function:")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Scores: U={test_case.uniqueness_score:.2f}, D={test_case.diversity_score:.2f}, I={test_case.intuition_score:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print()


def demo_unique_diverse_generator():
    """Demonstrate unique diverse test generation"""
    print("🎯 UNIQUE DIVERSE TEST GENERATOR DEMO")
    print("=" * 50)
    
    generator = UniqueDiverseTestGenerator()
    
    # Test with transformation function
    transform_func = demo_function_2()
    test_cases = generator.generate_unique_diverse_tests(transform_func, num_tests=10)
    
    print(f"Generated {len(test_cases)} unique diverse test cases for transformation function:")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Quality: {test_case.overall_quality:.2f} (U:{test_case.uniqueness_score:.2f}, D:{test_case.diversity_score:.2f}, I:{test_case.intuition_score:.2f})")
        print(f"   Parameters: {test_case.parameters}")
        print()


def demo_comprehensive_generator():
    """Demonstrate comprehensive test generation"""
    print("🚀 COMPREHENSIVE TEST GENERATOR DEMO")
    print("=" * 50)
    
    generator = ComprehensiveTestGenerator()
    
    # Test with business logic function
    pricing_func = demo_function_3()
    
    # Test different strategies
    strategies = ["intelligent", "unique_diverse", "comprehensive", "focused", "exploratory"]
    
    for strategy in strategies:
        print(f"\n📋 {strategy.upper()} STRATEGY:")
        print("-" * 30)
        
        test_cases = generator.generate_comprehensive_tests(
            pricing_func, 
            strategy=strategy, 
            num_tests=6
        )
        
        print(f"Generated {len(test_cases)} test cases:")
        
        for i, test_case in enumerate(test_cases[:3], 1):  # Show first 3
            print(f"{i}. {test_case.name}")
            print(f"   Quality: {test_case.overall_quality:.2f}")
            print(f"   Category: {test_case.test_category}, Priority: {test_case.priority}")
            print(f"   Method: {test_case.generation_method}")
        
        if len(test_cases) > 3:
            print(f"   ... and {len(test_cases) - 3} more tests")


def demo_test_file_generation():
    """Demonstrate complete test file generation"""
    print("\n📄 TEST FILE GENERATION DEMO")
    print("=" * 50)
    
    generator = ComprehensiveTestGenerator()
    
    # Generate complete test file for validation function
    validate_func = demo_function_1()
    
    print("Generating comprehensive test file for validation function...")
    
    test_file_content = generator.generate_test_file(
        validate_func,
        "demo_validation_tests.py",
        strategy="comprehensive",
        num_tests=15
    )
    
    print(f"✅ Generated test file with {len(test_file_content)} characters")
    print(f"📊 File contains {len(test_file_content.splitlines())} lines")
    
    # Show file structure
    lines = test_file_content.splitlines()
    print(f"\n📋 File structure:")
    print(f"   - Header: {len([l for l in lines[:20] if l.strip()])} lines")
    print(f"   - Imports: {len([l for l in lines if l.startswith('import') or l.startswith('from')])} lines")
    print(f"   - Fixtures: {len([l for l in lines if '@pytest.fixture' in l])} fixtures")
    print(f"   - Test methods: {len([l for l in lines if 'def test_' in l])} test methods")
    
    # Show sample content
    print(f"\n📝 Sample content (first 300 characters):")
    print("-" * 40)
    print(test_file_content[:300] + "..." if len(test_file_content) > 300 else test_file_content)


def demo_quality_metrics():
    """Demonstrate quality metrics and scoring"""
    print("\n📊 QUALITY METRICS DEMO")
    print("=" * 50)
    
    generator = ComprehensiveTestGenerator()
    
    # Test all three functions
    functions = [
        ("Validation Function", demo_function_1()),
        ("Transformation Function", demo_function_2()),
        ("Business Logic Function", demo_function_3())
    ]
    
    for func_name, func in functions:
        print(f"\n🔍 Analyzing {func_name}:")
        print("-" * 30)
        
        test_cases = generator.generate_comprehensive_tests(func, strategy="comprehensive", num_tests=10)
        
        if test_cases:
            avg_uniqueness = sum(tc.uniqueness_score for tc in test_cases) / len(test_cases)
            avg_diversity = sum(tc.diversity_score for tc in test_cases) / len(test_cases)
            avg_intuition = sum(tc.intuition_score for tc in test_cases) / len(test_cases)
            avg_quality = sum(tc.overall_quality for tc in test_cases) / len(test_cases)
            
            print(f"   📈 Average Uniqueness Score: {avg_uniqueness:.3f}")
            print(f"   🎯 Average Diversity Score: {avg_diversity:.3f}")
            print(f"   💡 Average Intuition Score: {avg_intuition:.3f}")
            print(f"   ⭐ Overall Quality Score: {avg_quality:.3f}")
            
            # Show best test case
            best_test = max(test_cases, key=lambda x: x.overall_quality)
            print(f"   🏆 Best Test Case: {best_test.name}")
            print(f"      Quality: {best_test.overall_quality:.3f}")


def main():
    """Main demonstration function"""
    print("🎉 ENHANCED TEST CASE GENERATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the enhanced test case generation system")
    print("that creates unique, diverse, and intuitive unit tests.")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_intelligent_generator()
        print("\n" + "="*60)
        
        demo_unique_diverse_generator()
        print("\n" + "="*60)
        
        demo_comprehensive_generator()
        print("\n" + "="*60)
        
        demo_test_file_generation()
        print("\n" + "="*60)
        
        demo_quality_metrics()
        
        print("\n" + "="*60)
        print("🎊 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The enhanced test case generation system successfully created:")
        print("✅ Unique test scenarios with varied approaches")
        print("✅ Diverse test cases covering wide range of scenarios")
        print("✅ Intuitive test naming and structure")
        print("✅ Advanced function analysis and pattern recognition")
        print("✅ Comprehensive coverage of edge cases and error conditions")
        print("✅ Quality metrics and scoring for test evaluation")
        print("✅ Complete test file generation with proper structure")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
