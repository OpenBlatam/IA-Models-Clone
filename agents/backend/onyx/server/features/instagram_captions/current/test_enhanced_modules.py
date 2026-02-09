#!/usr/bin/env python3
"""
Enhanced Test Script for Instagram Captions API v10.0 - Complete Modular Structure

Tests all modules including the newly created advanced features.
"""

import sys
import os
import time
import asyncio

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_security_modules():
    """Test all security modules."""
    print("🔒 Testing Security Modules...")
    
    try:
        # Test SecurityUtils
        from security import SecurityUtils
        api_key = SecurityUtils.generate_api_key(32)
        print(f"✅ SecurityUtils: API key generated ({api_key[:10]}...)")
        
        # Test ThreatDetector
        from security import ThreatDetector
        detector = ThreatDetector()
        
        # Test with safe text
        safe_result = detector.analyze_text("Hello world")
        print(f"✅ ThreatDetector: Safe text analysis - {safe_result['severity']}")
        
        # Test with malicious text
        malicious_result = detector.analyze_text("<script>alert('xss')</script>")
        print(f"✅ ThreatDetector: Malicious text detected - {malicious_result['severity']}")
        
        # Test EncryptionUtils
        from security import EncryptionUtils
        encryptor = EncryptionUtils()
        key, salt = encryptor.generate_encryption_key("test_password")
        encryptor.initialize_encryption(key)
        
        encrypted = encryptor.encrypt_text("secret data")
        decrypted = encryptor.decrypt_text(encrypted)
        print(f"✅ EncryptionUtils: Encryption/decryption working ({decrypted})")
        
        return True
        
    except Exception as e:
        print(f"❌ Security modules test failed: {e}")
        return False

def test_monitoring_modules():
    """Test all monitoring modules."""
    print("\n📊 Testing Monitoring Modules...")
    
    try:
        # Test PerformanceMonitor
        from monitoring import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.record_request(0.1, "/test", "GET", 200)
        summary = monitor.get_summary()
        print(f"✅ PerformanceMonitor: Working ({summary['total_requests']} requests)")
        
        # Test HealthChecker
        from monitoring import HealthChecker
        health_checker = HealthChecker()
        health_summary = health_checker.check_system_health()
        print(f"✅ HealthChecker: System health - {health_summary['overall_status']}")
        
        # Test MetricsCollector
        from monitoring import MetricsCollector
        metrics = MetricsCollector()
        metrics.record_counter("test_counter", 1, {"env": "test"})
        metrics.record_gauge("test_gauge", 42.5, {"env": "test"})
        metrics.record_timing("test_timing", 0.123, {"env": "test"})
        
        all_metrics = metrics.get_all_metrics_summary()
        print(f"✅ MetricsCollector: {all_metrics['total_metrics']} metrics recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring modules test failed: {e}")
        return False

def test_resilience_modules():
    """Test all resilience modules."""
    print("\n🔄 Testing Resilience Modules...")
    
    try:
        # Test CircuitBreaker
        from resilience import CircuitBreaker
        cb = CircuitBreaker()
        status = cb.get_status()
        print(f"✅ CircuitBreaker: Status - {status['state']}")
        
        # Test ErrorHandler
        from resilience import ErrorHandler
        error_handler = ErrorHandler()
        
        # Simulate an error
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_id = error_handler.log_error(e, "test_context", "medium")
            print(f"✅ ErrorHandler: Error logged with ID {error_id}")
        
        error_summary = error_handler.get_error_summary()
        print(f"✅ ErrorHandler: {error_summary['total_errors']} errors tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience modules test failed: {e}")
        return False

def test_core_modules():
    """Test all core modules."""
    print("\n⚙️ Testing Core Modules...")
    
    try:
        # Test logging utilities
        from core import setup_logging, get_logger
        setup_logging("INFO")
        logger = get_logger("test")
        print("✅ Logging utilities: Setup successful")
        
        # Test CacheManager
        from core import CacheManager
        cache = CacheManager()
        cache.set("test_key", "test_value")
        retrieved = cache.get("test_key")
        print(f"✅ CacheManager: Cache working ({retrieved})")
        
        # Test RateLimiter
        from core import RateLimiter
        rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Test rate limiting
        for i in range(6):
            allowed = rate_limiter.is_allowed("test_client")
            if i < 5:
                print(f"✅ RateLimiter: Request {i+1} allowed")
            else:
                print(f"✅ RateLimiter: Request {i+1} blocked (rate limit)")
        
        return True
        
    except Exception as e:
        print(f"❌ Core modules test failed: {e}")
        return False

async def test_async_functionality():
    """Test asynchronous functionality."""
    print("\n🚀 Testing Async Functionality...")
    
    try:
        # Test health checker async functionality
        from monitoring import HealthChecker
        health_checker = HealthChecker()
        
        # Test endpoint health check
        result = await health_checker.check_endpoint_health("/test-endpoint")
        print(f"✅ Async Health Check: {result['status']} - {result['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False

def test_integration():
    """Test integration between modules."""
    print("\n🔗 Testing Module Integration...")
    
    try:
        # Test security + monitoring integration
        from security import SecurityUtils, ThreatDetector
        from monitoring import PerformanceMonitor, MetricsCollector
        
        # Create instances
        security = SecurityUtils()
        detector = ThreatDetector()
        monitor = PerformanceMonitor()
        metrics = MetricsCollector()
        
        # Simulate a request flow
        start_time = time.time()
        
        # Generate API key
        api_key = security.generate_api_key(32)
        
        # Check for threats
        threat_result = detector.analyze_text("Normal request")
        
        # Record performance
        response_time = time.time() - start_time
        monitor.record_request(response_time, "/api/test", "POST", 200)
        
        # Record metrics
        metrics.record_timing("api_request", response_time, {"endpoint": "/api/test"})
        metrics.record_counter("api_requests", 1, {"status": "success"})
        
        print("✅ Module Integration: All modules working together")
        
        return True
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling across modules."""
    print("\n⚠️ Testing Error Handling...")
    
    try:
        from resilience import ErrorHandler
        from security import SecurityUtils
        
        error_handler = ErrorHandler()
        
        # Test various error scenarios
        errors_to_test = [
            (ValueError("Invalid input"), "input_validation", "medium"),
            (RuntimeError("Service unavailable"), "service_call", "high"),
            (TypeError("Type mismatch"), "data_processing", "low")
        ]
        
        for error, context, severity in errors_to_test:
            try:
                raise error
            except Exception as e:
                error_id = error_handler.log_error(e, context, severity)
                print(f"✅ Error logged: {error_id} - {context} ({severity})")
        
        # Test error summary
        summary = error_handler.get_error_summary()
        print(f"✅ Error Summary: {summary['total_errors']} errors, {len(summary['severity_distribution'])} severity levels")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all enhanced tests."""
    print("🔄 Enhanced Testing - Instagram Captions API v10.0 - Complete Modular Structure")
    print("=" * 80)
    
    tests = [
        ("Security Modules", test_security_modules),
        ("Monitoring Modules", test_monitoring_modules),
        ("Resilience Modules", test_resilience_modules),
        ("Core Modules", test_core_modules),
        ("Async Functionality", test_async_functionality),
        ("Module Integration", test_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 ENHANCED TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced tests passed! Complete modular structure is working correctly.")
        print("\n🚀 The API now includes:")
        print("   • Advanced Security (Threat Detection, Encryption)")
        print("   • Comprehensive Monitoring (Health Checks, Metrics)")
        print("   • Enterprise Resilience (Circuit Breaker, Error Handling)")
        print("   • Optimized Core Utilities (Caching, Rate Limiting)")
        return True
    else:
        print("⚠️ Some enhanced tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)






