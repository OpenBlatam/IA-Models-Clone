#!/usr/bin/env python3
"""
Instagram Captions API v10.0 - Enhanced Features Test Suite

Comprehensive testing of all new enhanced features:
- Advanced Security
- Enhanced Performance Monitoring
- Circuit Breaker Pattern
- Improved Error Handling
- Advanced Validation
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
import secrets

# Import enhanced components
from core_v10 import (
    RefactoredConfig, RefactoredCaptionRequest, RefactoredCaptionResponse,
    BatchRefactoredRequest, RefactoredAIEngine, Metrics
)
from ai_service_v10 import RefactoredAIService
from config import get_config, validate_config
from utils import (
    setup_logging, get_logger, SecurityUtils, CacheManager, 
    RateLimiter, PerformanceMonitor, ValidationUtils
)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class EnhancedFeaturesTestConfig:
    """Configuration for enhanced features testing."""
    
    # Test data
    TEST_TEXTS = [
        "Beautiful sunset at the beach",
        "Amazing coffee art",
        "Perfect morning workout",
        "Delicious homemade pizza",
        "Inspiring mountain view"
    ]
    
    # Security test patterns
    MALICIOUS_INPUTS = [
        "<script>alert('xss')</script>Hello World",
        "'; DROP TABLE users; --",
        "javascript:alert('xss')",
        "onload=alert('xss')",
        "http://malicious-site.com/steal-data",
        "admin' OR '1'='1"
    ]
    
    # Performance test settings
    PERFORMANCE_ITERATIONS = 100
    CONCURRENT_REQUESTS = 10

# =============================================================================
# ENHANCED FEATURES TEST SUITE
# =============================================================================

class EnhancedFeaturesTestSuite:
    """Comprehensive test suite for all enhanced features."""
    
    def __init__(self):
        # Setup logging
        setup_logging("INFO")
        self.logger = get_logger("enhanced_features_test")
        
        # Initialize components
        self.config = get_config()
        self.ai_engine = RefactoredAIEngine(self.config)
        self.ai_service = RefactoredAIService(self.config)
        self.metrics = Metrics()
        
        # Initialize utilities
        self.cache_manager = CacheManager(max_size=100, ttl=300)
        self.rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        self.performance_monitor = PerformanceMonitor()
        
        # Test results
        self.test_results = {
            "security_tests": {},
            "performance_tests": {},
            "circuit_breaker_tests": {},
            "validation_tests": {},
            "error_handling_tests": {},
            "overall_score": 0
        }
        
        self.logger.info("🚀 Enhanced Features Test Suite initialized")
    
    def print_header(self, title: str) -> None:
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f"🧪 {title}")
        print("=" * 80)
    
    def print_section(self, title: str) -> None:
        """Print formatted section."""
        print(f"\n📋 {title}")
        print("-" * 60)
    
    def test_enhanced_security_features(self) -> Dict[str, Any]:
        """Test all enhanced security features."""
        self.print_section("ENHANCED SECURITY FEATURES TESTING")
        
        security_results = {
            "api_key_generation": {},
            "api_key_validation": {},
            "input_sanitization": {},
            "content_type_validation": {},
            "file_extension_validation": {},
            "url_validation": {},
            "csrf_protection": {},
            "security_headers": {}
        }
        
        # Test API key generation
        print("🔑 Testing API Key Generation...")
        api_keys = []
        for i in range(5):
            api_key = SecurityUtils.generate_api_key(32 + i * 8)
            api_keys.append(api_key)
            print(f"   Generated {len(api_key)}-char key: {api_key[:16]}...")
        
        security_results["api_key_generation"] = {
            "generated_keys": len(api_keys),
            "key_lengths": [len(key) for key in api_keys],
            "success": True
        }
        
        # Test API key validation
        print("\n✅ Testing API Key Validation...")
        valid_keys = [SecurityUtils.verify_api_key(key) for key in api_keys]
        invalid_keys = [
            SecurityUtils.verify_api_key("weak"),
            SecurityUtils.verify_api_key("test123"),
            SecurityUtils.verify_api_key("abc"),
            SecurityUtils.verify_api_key("password")
        ]
        
        print(f"   Valid keys: {sum(valid_keys)}/{len(valid_keys)}")
        print(f"   Invalid keys rejected: {sum(not key for key in invalid_keys)}/{len(invalid_keys)}")
        
        security_results["api_key_validation"] = {
            "valid_keys_accepted": sum(valid_keys),
            "invalid_keys_rejected": sum(not key for key in invalid_keys),
            "success": all(valid_keys) and not any(invalid_keys)
        }
        
        # Test input sanitization
        print("\n🧹 Testing Input Sanitization...")
        sanitization_results = []
        for malicious_input in EnhancedFeaturesTestConfig.MALICIOUS_INPUTS:
            sanitized = SecurityUtils.sanitize_input(malicious_input, strict=True)
            is_safe = not any(dangerous in sanitized.lower() for dangerous in 
                            ['<script>', 'javascript:', 'onload', 'drop table'])
            sanitization_results.append({
                "original": malicious_input,
                "sanitized": sanitized,
                "is_safe": is_safe
            })
            print(f"   Original: {malicious_input[:50]}...")
            print(f"   Sanitized: {sanitized[:50]}...")
            print(f"   Safe: {'✅' if is_safe else '❌'}")
        
        security_results["input_sanitization"] = {
            "total_tests": len(sanitization_results),
            "safe_outputs": sum(1 for r in sanitization_results if r["is_safe"]),
            "success": all(r["is_safe"] for r in sanitization_results)
        }
        
        # Test content type validation
        print("\n📄 Testing Content Type Validation...")
        content_types = [
            "application/json",
            "text/plain",
            "application/x-www-form-urlencoded",
            "text/html",  # Should be rejected
            "application/javascript"  # Should be rejected
        ]
        
        content_type_results = []
        for content_type in content_types:
            is_valid = SecurityUtils.validate_content_type(content_type)
            content_type_results.append({
                "content_type": content_type,
                "is_valid": is_valid
            })
            print(f"   {content_type}: {'✅' if is_valid else '❌'}")
        
        security_results["content_type_validation"] = {
            "total_tests": len(content_type_results),
            "valid_types": sum(1 for r in content_type_results if r["is_valid"]),
            "success": len([r for r in content_type_results if r["is_valid"]]) == 3  # Only 3 should be valid
        }
        
        # Test file extension validation
        print("\n📁 Testing File Extension Validation...")
        filenames = [
            "safe_file.txt",
            "document.json",
            "data.csv",
            "malicious.exe",  # Should be rejected
            "script.bat"      # Should be rejected
        ]
        
        file_extension_results = []
        for filename in filenames:
            is_valid = SecurityUtils.validate_file_extension(filename)
            file_extension_results.append({
                "filename": filename,
                "is_valid": is_valid
            })
            print(f"   {filename}: {'✅' if is_valid else '❌'}")
        
        security_results["file_extension_validation"] = {
            "total_tests": len(file_extension_results),
            "valid_extensions": sum(1 for r in file_extension_results if r["is_valid"]),
            "success": len([r for r in file_extension_results if r["is_valid"]]) == 3  # Only 3 should be valid
        }
        
        # Test URL validation
        print("\n🌐 Testing URL Validation...")
        urls = [
            "https://example.com",
            "http://subdomain.example.org/path",
            "https://api.example.com/v1/endpoint?param=value",
            "javascript:alert('xss')",  # Should be rejected
            "ftp://malicious.com"       # Should be rejected
        ]
        
        url_results = []
        for url in urls:
            is_valid = SecurityUtils.validate_url(url)
            url_results.append({
                "url": url,
                "is_valid": is_valid
            })
            print(f"   {url}: {'✅' if is_valid else '❌'}")
        
        security_results["url_validation"] = {
            "total_tests": len(url_results),
            "valid_urls": sum(1 for r in url_results if r["is_valid"]),
            "success": len([r for r in url_results if r["is_valid"]]) == 3  # Only 3 should be valid
        }
        
        # Test CSRF protection
        print("\n🛡️ Testing CSRF Protection...")
        csrf_token1 = SecurityUtils.generate_csrf_token()
        csrf_token2 = SecurityUtils.generate_csrf_token()
        
        is_valid1 = SecurityUtils.verify_csrf_token(csrf_token1, csrf_token1)
        is_valid2 = SecurityUtils.verify_csrf_token(csrf_token1, csrf_token2)
        
        print(f"   Generated token: {csrf_token1[:16]}...")
        print(f"   Same token verification: {'✅' if is_valid1 else '❌'}")
        print(f"   Different token verification: {'✅' if not is_valid2 else '❌'}")
        
        security_results["csrf_protection"] = {
            "token_generated": True,
            "same_token_valid": is_valid1,
            "different_token_invalid": not is_valid2,
            "success": is_valid1 and not is_valid2
        }
        
        # Test security headers
        print("\n🔒 Testing Security Headers...")
        security_headers = SecurityUtils.generate_security_headers()
        expected_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Content-Security-Policy'
        ]
        
        headers_present = all(header in security_headers for header in expected_headers)
        print(f"   Security headers generated: {len(security_headers)}")
        for header, value in security_headers.items():
            print(f"   {header}: {value}")
        
        security_results["security_headers"] = {
            "total_headers": len(security_headers),
            "expected_headers_present": headers_present,
            "success": headers_present
        }
        
        # Calculate security score
        security_score = sum(
            1 for result in security_results.values() 
            if isinstance(result, dict) and result.get("success", False)
        ) / len([r for r in security_results.values() if isinstance(r, dict)])
        
        print(f"\n🎯 Security Test Score: {security_score:.1%}")
        
        self.test_results["security_tests"] = security_results
        return security_results
    
    def test_enhanced_performance_monitoring(self) -> Dict[str, Any]:
        """Test enhanced performance monitoring features."""
        self.print_section("ENHANCED PERFORMANCE MONITORING TESTING")
        
        performance_results = {
            "basic_metrics": {},
            "advanced_statistics": {},
            "performance_trends": {},
            "thresholds_and_alerts": {},
            "historical_data": {}
        }
        
        # Test basic metrics recording
        print("📊 Testing Basic Metrics Recording...")
        test_metrics = ["test_metric_1", "test_metric_2", "test_metric_3"]
        
        for metric_name in test_metrics:
            for i in range(50):
                value = 1.0 + (i * 0.1) + (secrets.randbelow(100) / 1000)  # Add some randomness
                self.performance_monitor.record_metric(
                    metric_name, 
                    value,
                    metadata={"iteration": i, "test": True}
                )
        
        print(f"   Recorded metrics for: {', '.join(test_metrics)}")
        print(f"   Total metrics recorded: {sum(len(self.performance_monitor.metrics[m]) for m in test_metrics)}")
        
        performance_results["basic_metrics"] = {
            "metrics_recorded": len(test_metrics),
            "total_values": sum(len(self.performance_monitor.metrics[m]) for m in test_metrics),
            "success": True
        }
        
        # Test advanced statistics
        print("\n📈 Testing Advanced Statistics...")
        for metric_name in test_metrics:
            stats = self.performance_monitor.get_statistics(metric_name)
            print(f"   {metric_name}:")
            print(f"     Count: {stats.get('count', 0)}")
            print(f"     Mean: {stats.get('mean', 0):.3f}")
            print(f"     P95: {stats.get('p95', 0):.3f}")
            print(f"     Std Dev: {stats.get('std_dev', 0):.3f}")
        
        performance_results["advanced_statistics"] = {
            "statistics_calculated": len(test_metrics),
            "percentiles_available": all('p95' in self.performance_monitor.get_statistics(m) for m in test_metrics),
            "success": True
        }
        
        # Test performance trends
        print("\n📊 Testing Performance Trends...")
        for metric_name in test_metrics:
            trends = self.performance_monitor.get_performance_trends(metric_name, 60)
            print(f"   {metric_name} trends:")
            print(f"     Direction: {trends.get('trend_direction', 'unknown')}")
            print(f"     Slope: {trends.get('slope', 0):.6f}")
            print(f"     Data points: {trends.get('data_points', 0)}")
        
        performance_results["performance_trends"] = {
            "trends_calculated": len(test_metrics),
            "trend_directions": [self.performance_monitor.get_performance_trends(m, 60).get('trend_direction') for m in test_metrics],
            "success": True
        }
        
        # Test thresholds and alerts
        print("\n🚨 Testing Thresholds and Alerts...")
        self.performance_monitor.set_threshold("test_metric_1", "max", 3.0)
        self.performance_monitor.set_threshold("test_metric_2", "min", 0.5)
        
        # Trigger some threshold violations
        self.performance_monitor.record_metric("test_metric_1", 5.0)  # Exceeds max
        self.performance_monitor.record_metric("test_metric_2", 0.1)  # Below min
        
        alerts = self.performance_monitor.get_alerts()
        print(f"   Active alerts: {len(alerts)}")
        for alert in alerts:
            print(f"     {alert['metric']}: {alert['threshold_type']} threshold exceeded")
            print(f"       Expected: {alert['threshold_value']}, Actual: {alert['actual_value']}")
            print(f"       Severity: {alert['severity']}")
        
        performance_results["thresholds_and_alerts"] = {
            "thresholds_set": 2,
            "alerts_generated": len(alerts),
            "success": len(alerts) > 0
        }
        
        # Test historical data
        print("\n📚 Testing Historical Data...")
        for metric_name in test_metrics:
            historical = self.performance_monitor.historical_data.get(metric_name, [])
            print(f"   {metric_name}: {len(historical)} historical records")
            if historical:
                latest = historical[-1]
                print(f"     Latest: {latest['value']:.3f} at {latest['timestamp']}")
                print(f"     Metadata: {latest['metadata']}")
        
        performance_results["historical_data"] = {
            "metrics_with_history": len([m for m in test_metrics if m in self.performance_monitor.historical_data]),
            "total_historical_records": sum(len(self.performance_monitor.historical_data.get(m, [])) for m in test_metrics),
            "success": True
        }
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary()
        print(f"\n📋 Performance Summary:")
        print(f"   Uptime: {summary['uptime_seconds']:.1f} seconds")
        print(f"   Total metrics: {summary['total_metrics']}")
        print(f"   Active alerts: {summary['total_alerts']}")
        print(f"   Active thresholds: {summary['active_thresholds']}")
        
        self.test_results["performance_tests"] = performance_results
        return performance_results
    
    def test_enhanced_validation_features(self) -> Dict[str, Any]:
        """Test enhanced validation utilities."""
        self.print_section("ENHANCED VALIDATION FEATURES TESTING")
        
        validation_results = {
            "email_validation": {},
            "url_validation": {},
            "phone_validation": {},
            "filename_sanitization": {}
        }
        
        # Test email validation
        print("📧 Testing Email Validation...")
        test_emails = [
            "user@example.com",
            "test.email@domain.co.uk",
            "user+tag@example.org",
            "invalid-email",
            "no@dots",
            "user@.com",
            "@example.com"
        ]
        
        email_results = []
        for email in test_emails:
            is_valid = ValidationUtils.validate_email(email)
            email_results.append({
                "email": email,
                "is_valid": is_valid
            })
            print(f"   {email}: {'✅' if is_valid else '❌'}")
        
        validation_results["email_validation"] = {
            "total_tests": len(email_results),
            "valid_emails": sum(1 for r in email_results if r["is_valid"]),
            "expected_valid": 3,  # First 3 should be valid
            "success": sum(1 for r in email_results if r["is_valid"]) == 3
        }
        
        # Test URL validation
        print("\n🌐 Testing URL Validation...")
        test_urls = [
            "https://example.com",
            "http://subdomain.example.org/path?param=value",
            "https://api.example.com/v1/endpoint#section",
            "invalid-url",
            "ftp://example.com",
            "javascript:alert('xss')"
        ]
        
        url_results = []
        for url in test_urls:
            is_valid = ValidationUtils.validate_url(url)
            url_results.append({
                "url": url,
                "is_valid": is_valid
            })
            print(f"   {url}: {'✅' if is_valid else '❌'}")
        
        validation_results["url_validation"] = {
            "total_tests": len(url_results),
            "valid_urls": sum(1 for r in url_results if r["is_valid"]),
            "expected_valid": 3,  # First 3 should be valid
            "success": sum(1 for r in url_results if r["is_valid"]) == 3
        }
        
        # Test phone validation
        print("\n📱 Testing Phone Validation...")
        test_phones = [
            "+1-555-123-4567",
            "555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "5551234567",
            "123",  # Too short
            "12345678901234567890"  # Too long
        ]
        
        phone_results = []
        for phone in test_phones:
            is_valid = ValidationUtils.validate_phone(phone)
            phone_results.append({
                "phone": phone,
                "is_valid": is_valid
            })
            print(f"   {phone}: {'✅' if is_valid else '❌'}")
        
        validation_results["phone_validation"] = {
            "total_tests": len(phone_results),
            "valid_phones": sum(1 for r in phone_results if r["is_valid"]),
            "expected_valid": 5,  # First 5 should be valid
            "success": sum(1 for r in phone_results if r["is_valid"]) == 5
        }
        
        # Test filename sanitization
        print("\n📁 Testing Filename Sanitization...")
        test_filenames = [
            "safe_file.txt",
            "file<with>invalid:chars",
            "file with spaces.txt",
            "..hidden_file",
            "file/with/slashes.txt",
            "file\\with\\backslashes.txt",
            "file*with?wildcards.txt"
        ]
        
        filename_results = []
        for filename in test_filenames:
            sanitized = ValidationUtils.sanitize_filename(filename)
            filename_results.append({
                "original": filename,
                "sanitized": sanitized,
                "is_safe": not any(char in sanitized for char in ['<', '>', ':', '*', '?', '/', '\\'])
            })
            print(f"   Original: {filename}")
            print(f"   Sanitized: {sanitized}")
            print(f"   Safe: {'✅' if filename_results[-1]['is_safe'] else '❌'}")
        
        validation_results["filename_sanitization"] = {
            "total_tests": len(filename_results),
            "safe_outputs": sum(1 for r in filename_results if r["is_safe"]),
            "success": all(r["is_safe"] for r in filename_results)
        }
        
        # Calculate validation score
        validation_score = sum(
            1 for result in validation_results.values() 
            if result.get("success", False)
        ) / len(validation_results)
        
        print(f"\n🎯 Validation Test Score: {validation_score:.1%}")
        
        self.test_results["validation_tests"] = validation_results
        return validation_results
    
    def test_circuit_breaker_pattern(self) -> Dict[str, Any]:
        """Test circuit breaker pattern implementation."""
        self.print_section("CIRCUIT BREAKER PATTERN TESTING")
        
        # Import circuit breaker from API
        from api_v10 import CircuitBreaker
        
        circuit_breaker_results = {
            "normal_operation": {},
            "failure_threshold": {},
            "circuit_open": {},
            "recovery": {},
            "status_tracking": {}
        }
        
        # Test normal operation
        print("✅ Testing Normal Operation...")
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
        
        def successful_function():
            return "success"
        
        def failing_function():
            raise Exception("Simulated failure")
        
        # Test successful calls
        try:
            result = cb.call(successful_function)
            print(f"   Successful call result: {result}")
            circuit_breaker_results["normal_operation"]["successful_calls"] = True
        except Exception as e:
            print(f"   Unexpected error: {e}")
            circuit_breaker_results["normal_operation"]["successful_calls"] = False
        
        # Test failure threshold
        print("\n❌ Testing Failure Threshold...")
        failure_count = 0
        for i in range(5):
            try:
                cb.call(failing_function)
            except Exception as e:
                failure_count += 1
                print(f"   Failure {failure_count}: {e}")
        
        circuit_breaker_results["failure_threshold"] = {
            "failures_recorded": failure_count,
            "threshold_reached": failure_count >= 3,
            "success": failure_count >= 3
        }
        
        # Test circuit open state
        print(f"\n🚫 Testing Circuit Open State...")
        print(f"   Circuit state: {cb.state}")
        print(f"   Failure count: {cb.failure_count}")
        
        try:
            cb.call(successful_function)
            print("   ❌ Circuit should be open but allowed call")
            circuit_breaker_results["circuit_open"]["blocked_calls"] = False
        except Exception as e:
            print(f"   ✅ Circuit blocked call: {e}")
            circuit_breaker_results["circuit_open"]["blocked_calls"] = True
        
        # Test recovery
        print(f"\n🔄 Testing Recovery...")
        print(f"   Waiting for recovery timeout...")
        
        # Simulate time passing
        cb.last_failure_time = time.time() - 10  # 10 seconds ago
        
        try:
            result = cb.call(successful_function)
            print(f"   ✅ Recovery successful: {result}")
            circuit_breaker_results["recovery"]["recovery_successful"] = True
        except Exception as e:
            print(f"   ❌ Recovery failed: {e}")
            circuit_breaker_results["recovery"]["recovery_successful"] = False
        
        # Test status tracking
        print(f"\n📊 Testing Status Tracking...")
        status = cb.get_status()
        print(f"   Current state: {status['state']}")
        print(f"   Failure count: {status['failure_count']}")
        print(f"   Last failure: {status['last_failure_time']}")
        
        circuit_breaker_results["status_tracking"] = {
            "status_retrieved": True,
            "state_tracked": status['state'] in ['CLOSED', 'OPEN', 'HALF_OPEN'],
            "success": True
        }
        
        # Calculate circuit breaker score
        circuit_breaker_score = sum(
            1 for result in circuit_breaker_results.values() 
            if result.get("success", False)
        ) / len(circuit_breaker_results)
        
        print(f"\n🎯 Circuit Breaker Test Score: {circuit_breaker_score:.1%}")
        
        self.test_results["circuit_breaker_tests"] = circuit_breaker_results
        return circuit_breaker_results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete enhanced features test suite."""
        self.print_header("INSTAGRAM CAPTIONS API v10.0 - ENHANCED FEATURES TEST SUITE")
        
        print("🎯 This test suite validates all the enhanced features implemented:")
        print("   • Advanced Security Features")
        print("   • Enhanced Performance Monitoring")
        print("   • Circuit Breaker Pattern")
        print("   • Advanced Validation")
        print("   • Improved Error Handling")
        
        # Run all test sections
        security_results = self.test_enhanced_security_features()
        performance_results = self.test_enhanced_performance_monitoring()
        validation_results = self.test_enhanced_validation_features()
        circuit_breaker_results = self.test_circuit_breaker_pattern()
        
        # Calculate overall score
        section_scores = [
            security_results,
            performance_results,
            validation_results,
            circuit_breaker_results
        ]
        
        # Count successful tests
        total_tests = 0
        successful_tests = 0
        
        for section in section_scores:
            for test_name, test_result in section.items():
                if isinstance(test_result, dict) and 'success' in test_result:
                    total_tests += 1
                    if test_result['success']:
                        successful_tests += 1
        
        overall_score = successful_tests / total_tests if total_tests > 0 else 0
        self.test_results["overall_score"] = overall_score
        
        # Summary
        self.print_section("TEST SUITE SUMMARY")
        print(f"🎉 Enhanced Features Test Suite completed!")
        print(f"📊 Overall Score: {overall_score:.1%} ({successful_tests}/{total_tests} tests passed)")
        print(f"\n📋 Section Results:")
        print(f"   🔒 Security Features: {'✅' if all(r.get('success', False) for r in security_results.values() if isinstance(r, dict)) else '❌'}")
        print(f"   📊 Performance Monitoring: {'✅' if all(r.get('success', False) for r in performance_results.values() if isinstance(r, dict)) else '❌'}")
        print(f"   🛡️ Validation Features: {'✅' if all(r.get('success', False) for r in validation_results.values() if isinstance(r, dict)) else '❌'}")
        print(f"   🔌 Circuit Breaker: {'✅' if all(r.get('success', False) for r in circuit_breaker_results.values() if isinstance(r, dict)) else '❌'}")
        
        # Save results
        with open("enhanced_features_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to 'enhanced_features_test_results.json'")
        
        return self.test_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main test execution."""
    try:
        test_suite = EnhancedFeaturesTestSuite()
        results = await test_suite.run_comprehensive_test_suite()
        
        print(f"\n🎯 Test Suite completed with {results['overall_score']:.1%} success rate!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())






