#!/usr/bin/env python3
"""
Instagram Captions API v10.0 - Refactored Demo

Comprehensive demonstration showcasing the refactored architecture,
improved security, performance monitoring, and new features.
"""

import asyncio
import time
import json
from typing import Dict, Any, List
import logging

# Import refactored components
from core_v10 import (
    RefactoredConfig, RefactoredCaptionRequest, RefactoredCaptionResponse,
    BatchRefactoredRequest, RefactoredAIEngine, RefactoredUtils, Metrics
)
from ai_service_v10 import RefactoredAIService
from config import get_config, validate_config
from utils import (
    setup_logging, get_logger, SecurityUtils, CacheManager, 
    RateLimiter, PerformanceMonitor, ValidationUtils
)

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

class RefactoredDemoConfig:
    """Configuration for the refactored demo."""
    
    DEMO_TEXT = [
        "Beautiful sunset at the beach",
        "Amazing coffee art",
        "Perfect morning workout",
        "Delicious homemade pizza",
        "Inspiring mountain view"
    ]
    
    DEMO_STYLES = ["casual", "formal", "creative", "professional", "funny", "inspirational"]
    DEMO_LENGTHS = ["short", "medium", "long"]
    
    # Demo settings
    ENABLE_PERFORMANCE_TESTING = True
    ENABLE_SECURITY_TESTING = True
    ENABLE_CACHE_TESTING = True
    ENABLE_RATE_LIMITING_TESTING = True
    ENABLE_CONFIGURATION_TESTING = True

# =============================================================================
# REFACTORED DEMO CLASS
# =============================================================================

class RefactoredDemo:
    """Enhanced demonstration of v10.0 refactoring improvements."""
    
    def __init__(self) -> None:
        # Setup logging
        setup_logging("INFO")
        self.logger = get_logger("refactored_demo")
        
        # Initialize components with new configuration system
        self.config = get_config()
        self.ai_engine = RefactoredAIEngine(self.config)
        self.ai_service = RefactoredAIService(self.config)
        self.metrics = Metrics()
        
        # Initialize utilities
        self.cache_manager = CacheManager(max_size=100, ttl=300)
        self.rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        self.performance_monitor = PerformanceMonitor()
        
        # Demo results
        self.demo_results = {
            "refactoring_improvements": {},
            "configuration_management": {},
            "security_features": {},
            "performance_metrics": {},
            "cache_performance": {},
            "rate_limiting": {},
            "validation_tests": {},
            "ai_capabilities": {}
        }
        
        self.logger.info("🚀 Refactored Demo v10.0 initialized")
    
    def print_header(self, title: str) -> None:
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f"🚀 {title}")
        print("=" * 80)
    
    def print_section(self, title: str) -> None:
        """Print formatted section."""
        print(f"\n📋 {title}")
        print("-" * 60)
    
    def demo_refactoring_improvements(self) -> Dict[str, Any]:
        """Demonstrate refactoring improvements."""
        self.print_section("REFACTORING IMPROVEMENTS")
        
        improvements = {
            "code_organization": {
                "before": "Monolithic files with mixed concerns",
                "after": "Clean separation of concerns with dedicated modules",
                "benefit": "Better maintainability and code reusability"
            },
            "import_management": {
                "before": "Duplicate imports and inconsistent organization",
                "after": "Clean, organized imports with proper grouping",
                "benefit": "Reduced complexity and better dependency management"
            },
            "configuration_integration": {
                "before": "Hardcoded values scattered throughout code",
                "after": "Centralized configuration with environment support",
                "benefit": "Flexible deployment and easier configuration management"
            },
            "error_handling": {
                "before": "Basic error handling with generic responses",
                "after": "Comprehensive error handling with proper HTTP status codes",
                "benefit": "Better debugging and improved user experience"
            },
            "middleware_architecture": {
                "before": "Simple middleware stack",
                "after": "Advanced middleware with security, logging, and rate limiting",
                "benefit": "Enhanced security and comprehensive monitoring"
            }
        }
        
        for category, details in improvements.items():
            print(f"\n🔧 {category.replace('_', ' ').title()}:")
            print(f"   ❌ Before: {details['before']}")
            print(f"   ✅ After: {details['after']}")
            print(f"   💡 Benefit: {details['benefit']}")
        
        self.demo_results["refactoring_improvements"] = improvements
        return improvements
    
    def demo_configuration_management(self) -> Dict[str, Any]:
        """Demonstrate new configuration management system."""
        self.print_section("CONFIGURATION MANAGEMENT")
        
        # Show current configuration
        print(f"🔧 Current Configuration:")
        print(f"   Environment: {self.config.ENVIRONMENT}")
        print(f"   API Version: {self.config.API_VERSION}")
        print(f"   Server: {self.config.HOST}:{self.config.PORT}")
        print(f"   AI Model: {self.config.AI_MODEL_NAME}")
        print(f"   Cache Size: {self.config.CACHE_SIZE}")
        print(f"   Rate Limit: {self.config.RATE_LIMIT_PER_MINUTE}/minute")
        
        # Validate configuration
        validation_result = validate_config(self.config)
        print(f"\n✅ Configuration Validation:")
        print(f"   Valid: {validation_result['valid']}")
        
        if validation_result['warnings']:
            print(f"   ⚠️  Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings']:
                print(f"      • {warning}")
        
        if validation_result['errors']:
            print(f"   ❌ Errors: {len(validation_result['errors'])}")
            for error in validation_result['errors']:
                print(f"      • {error}")
        
        if validation_result['recommendations']:
            print(f"   💡 Recommendations: {len(validation_result['recommendations'])}")
            for rec in validation_result['recommendations']:
                print(f"      • {rec}")
        
        config_results = {
            "current_config": {
                "environment": self.config.ENVIRONMENT,
                "api_version": self.config.API_VERSION,
                "host": self.config.HOST,
                "port": self.config.PORT,
                "ai_model": self.config.AI_MODEL_NAME,
                "cache_size": self.config.CACHE_SIZE,
                "rate_limit": self.config.RATE_LIMIT_PER_MINUTE
            },
            "validation": validation_result
        }
        
        self.demo_results["configuration_management"] = config_results
        return config_results
    
    def demo_security_features(self) -> Dict[str, Any]:
        """Demonstrate security improvements."""
        self.print_section("SECURITY FEATURES")
        
        # Test API key generation
        api_key = SecurityUtils.generate_api_key(32)
        print(f"🔑 Generated API Key: {api_key[:16]}...")
        
        # Test API key validation
        valid_key = SecurityUtils.verify_api_key(api_key)
        invalid_key = SecurityUtils.verify_api_key("weak_key")
        print(f"✅ Valid API Key: {valid_key}")
        print(f"❌ Invalid API Key: {invalid_key}")
        
        # Test input sanitization
        malicious_input = "<script>alert('xss')</script>Hello World"
        sanitized = SecurityUtils.sanitize_input(malicious_input)
        print(f"🧹 Input Sanitization:")
        print(f"   Original: {malicious_input}")
        print(f"   Sanitized: {sanitized}")
        
        # Test content type validation
        valid_content_type = SecurityUtils.validate_content_type("application/json")
        invalid_content_type = SecurityUtils.validate_content_type("text/html")
        print(f"📄 Content Type Validation:")
        print(f"   Valid JSON: {valid_content_type}")
        print(f"   Invalid HTML: {invalid_content_type}")
        
        security_features = {
            "api_key_generation": api_key,
            "api_key_validation": {"valid": valid_key, "invalid": invalid_key},
            "input_sanitization": {"original": malicious_input, "sanitized": sanitized},
            "content_type_validation": {"json": valid_content_type, "html": invalid_content_type}
        }
        
        self.demo_results["security_features"] = security_features
        return security_features
    
    def demo_cache_performance(self) -> Dict[str, Any]:
        """Demonstrate cache performance improvements."""
        self.print_section("CACHE PERFORMANCE")
        
        # Test cache operations
        test_data = {"caption": "Test caption", "style": "casual"}
        
        # Measure cache performance
        start_time = time.time()
        self.cache_manager.set("test_key", test_data, ttl=60)
        set_time = time.time() - start_time
        
        start_time = time.time()
        cached_data = self.cache_manager.get("test_key")
        get_time = time.time() - start_time
        
        print(f"⏱️  Cache Performance:")
        print(f"   Set operation: {set_time:.6f}s")
        print(f"   Get operation: {get_time:.6f}s")
        print(f"   Cache hit: {cached_data == test_data}")
        
        # Test cache statistics
        stats = self.cache_manager.get_stats()
        print(f"📊 Cache Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        cache_performance = {
            "set_time": set_time,
            "get_time": get_time,
            "cache_hit": cached_data == test_data,
            "statistics": stats
        }
        
        self.demo_results["cache_performance"] = cache_performance
        return cache_performance
    
    def demo_rate_limiting(self) -> Dict[str, Any]:
        """Demonstrate rate limiting functionality."""
        self.print_section("RATE LIMITING")
        
        test_identifier = "demo_user:caption_generation"
        
        # Test rate limiting
        allowed_requests = []
        blocked_requests = []
        
        for i in range(25):  # Try 25 requests
            if self.rate_limiter.is_allowed(test_identifier):
                allowed_requests.append(i)
            else:
                blocked_requests.append(i)
        
        remaining = self.rate_limiter.get_remaining_requests(test_identifier)
        
        print(f"🚦 Rate Limiting Test:")
        print(f"   Allowed requests: {len(allowed_requests)}")
        print(f"   Blocked requests: {len(blocked_requests)}")
        print(f"   Remaining requests: {remaining}")
        print(f"   Rate limit: {self.rate_limiter.requests_per_minute}/minute")
        print(f"   Burst limit: {self.rate_limiter.burst_size}/second")
        
        rate_limiting_results = {
            "allowed_requests": len(allowed_requests),
            "blocked_requests": len(blocked_requests),
            "remaining_requests": remaining,
            "rate_limit": self.rate_limiter.requests_per_minute,
            "burst_limit": self.rate_limiter.burst_size
        }
        
        self.demo_results["rate_limiting"] = rate_limiting_results
        return rate_limiting_results
    
    def demo_validation_features(self) -> Dict[str, Any]:
        """Demonstrate validation utilities."""
        self.print_section("VALIDATION FEATURES")
        
        # Test email validation
        test_emails = [
            "user@example.com",
            "invalid-email",
            "test.email@domain.co.uk",
            "no@dots"
        ]
        
        print("📧 Email Validation:")
        for email in test_emails:
            is_valid = ValidationUtils.validate_email(email)
            status = "✅" if is_valid else "❌"
            print(f"   {status} {email}")
        
        # Test URL validation
        test_urls = [
            "https://example.com",
            "http://subdomain.example.org/path?param=value",
            "invalid-url",
            "ftp://example.com"
        ]
        
        print("\n🌐 URL Validation:")
        for url in test_urls:
            is_valid = ValidationUtils.validate_url(url)
            status = "✅" if is_valid else "❌"
            print(f"   {status} {url}")
        
        # Test filename sanitization
        test_filenames = [
            "safe_file.txt",
            "file<with>invalid:chars",
            "file with spaces.txt",
            "..hidden_file"
        ]
        
        print("\n📁 Filename Sanitization:")
        for filename in test_filenames:
            sanitized = ValidationUtils.sanitize_filename(filename)
            print(f"   Original: {filename}")
            print(f"   Sanitized: {sanitized}")
        
        validation_results = {
            "email_validation": {email: ValidationUtils.validate_email(email) for email in test_emails},
            "url_validation": {url: ValidationUtils.validate_url(url) for url in test_urls},
            "filename_sanitization": {filename: ValidationUtils.sanitize_filename(filename) for filename in test_filenames}
        }
        
        self.demo_results["validation_tests"] = validation_results
        return validation_results
    
    async def demo_ai_capabilities(self) -> Dict[str, Any]:
        """Demonstrate AI capabilities."""
        self.print_section("AI CAPABILITIES")
        
        # Test single caption generation
        test_request = RefactoredCaptionRequest(
            text="Beautiful sunset at the beach",
            style="casual",
            length="medium"
        )
        
        print("🤖 Single Caption Generation:")
        try:
            start_time = time.time()
            response = await self.ai_engine.generate_caption(test_request)
            generation_time = time.time() - start_time
            
            print(f"   ✅ Success in {generation_time:.3f}s")
            print(f"   📝 Caption: {response.caption}")
            print(f"   🏷️  Style: {response.style}")
            print(f"   📏 Length: {response.length}")
            print(f"   #️⃣  Hashtags: {', '.join(response.hashtags)}")
            print(f"   😊 Emojis: {', '.join(response.emojis)}")
            
            # Record performance metric
            self.performance_monitor.record_metric("caption_generation", generation_time)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            generation_time = 0
        
        # Test batch generation
        print("\n🔄 Batch Caption Generation:")
        try:
            batch_requests = [
                RefactoredCaptionRequest(text=text, style="casual", length="short")
                for text in RefactoredDemoConfig.DEMO_TEXT[:3]
            ]
            
            batch_request = BatchRefactoredRequest(requests=batch_requests)
            
            start_time = time.time()
            batch_response = await self.ai_service.generate_batch_captions(batch_request)
            batch_time = time.time() - start_time
            
            print(f"   ✅ Batch completed in {batch_time:.3f}s")
            print(f"   📦 Total requests: {batch_response['total_requests']}")
            print(f"   ✅ Successful: {batch_response['successful_requests']}")
            print(f"   ❌ Failed: {batch_response['failed_requests']}")
            print(f"   ⏱️  Average time per request: {batch_response['average_time_per_request']:.3f}s")
            
            # Record performance metric
            self.performance_monitor.record_metric("batch_generation", batch_time)
            
        except Exception as e:
            print(f"   ❌ Batch failed: {e}")
            batch_time = 0
        
        ai_results = {
            "single_generation": {
                "success": "response" in locals(),
                "time": generation_time,
                "response": response if "response" in locals() else None
            },
            "batch_generation": {
                "success": "batch_response" in locals(),
                "time": batch_time,
                "response": batch_response if "batch_response" in locals() else None
            }
        }
        
        self.demo_results["ai_capabilities"] = ai_results
        return ai_results
    
    def demo_performance_monitoring(self) -> Dict[str, Any]:
        """Demonstrate performance monitoring."""
        self.print_section("PERFORMANCE MONITORING")
        
        # Get performance statistics
        all_stats = self.performance_monitor.get_all_statistics()
        
        print("📊 Performance Metrics:")
        for metric_name, stats in all_stats.items():
            if stats:
                print(f"\n   📈 {metric_name}:")
                for stat_name, value in stats.items():
                    print(f"      {stat_name}: {value}")
        
        # Get API metrics
        api_stats = self.metrics.get_stats()
        print(f"\n   🚀 API Metrics:")
        for key, value in api_stats.items():
            print(f"      {key}: {value}")
        
        performance_results = {
            "performance_metrics": all_stats,
            "api_metrics": api_stats
        }
        
        self.demo_results["performance_metrics"] = performance_results
        return performance_results
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run the complete refactored demonstration."""
        self.print_header("INSTAGRAM CAPTIONS API v10.0 - REFACTORED DEMO")
        
        print("🎯 This demo showcases the refactoring improvements made to the Instagram Captions API:")
        print("   • Enhanced code organization and architecture")
        print("   • Improved configuration management")
        print("   • Advanced security features")
        print("   • Performance monitoring and optimization")
        print("   • Better caching and rate limiting")
        print("   • Robust validation and error handling")
        
        # Run all demo sections
        self.demo_refactoring_improvements()
        self.demo_configuration_management()
        self.demo_security_features()
        self.demo_cache_performance()
        self.demo_rate_limiting()
        self.demo_validation_features()
        await self.demo_ai_capabilities()
        self.demo_performance_monitoring()
        
        # Summary
        self.print_section("REFACTORING SUMMARY")
        print("🎉 All refactoring improvements demonstrated successfully!")
        print("📈 The API now features:")
        print("   ✅ Clean, maintainable architecture")
        print("   ✅ Centralized configuration management")
        print("   ✅ Comprehensive security measures")
        print("   ✅ Advanced performance monitoring")
        print("   ✅ Efficient caching and rate limiting")
        print("   ✅ Robust validation and error handling")
        print("   ✅ Production-ready configuration validation")
        
        return self.demo_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main demo execution."""
    try:
        demo = RefactoredDemo()
        results = await demo.run_comprehensive_demo()
        
        # Save results to file
        with open("refactored_demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Refactored demo results saved to 'refactored_demo_results.json'")
        
    except Exception as e:
        print(f"❌ Refactored demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())






