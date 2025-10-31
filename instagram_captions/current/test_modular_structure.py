#!/usr/bin/env python3
"""
Test Script for Modular Structure

Validates that the new modular architecture works correctly.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing modular imports...")
    
    try:
        # Test security module
        from security import SecurityUtils
        print("✅ Security module imported successfully")
        
        # Test monitoring module
        from monitoring import PerformanceMonitor
        print("✅ Monitoring module imported successfully")
        
        # Test resilience module
        from resilience import CircuitBreaker, ErrorHandler
        print("✅ Resilience module imported successfully")
        
        # Test core module
        from core import setup_logging, get_logger, CacheManager, RateLimiter
        print("✅ Core module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_functionality():
    """Test basic functionality of imported modules."""
    print("\n🔧 Testing module functionality...")
    
    try:
        # Test SecurityUtils
        from security import SecurityUtils
        api_key = SecurityUtils.generate_api_key(32)
        print(f"✅ API key generated: {api_key[:10]}...")
        
        # Test PerformanceMonitor
        from monitoring import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.record_request(0.1, "/test", "GET", 200)
        print("✅ Performance monitor working")
        
        # Test CircuitBreaker
        from resilience import CircuitBreaker
        cb = CircuitBreaker()
        print(f"✅ Circuit breaker status: {cb.get_status()['state']}")
        
        # Test ErrorHandler
        from resilience import ErrorHandler
        eh = ErrorHandler()
        print("✅ Error handler initialized")
        
        # Test core utilities
        from core import setup_logging, CacheManager, RateLimiter
        setup_logging("INFO")
        print("✅ Logging setup successful")
        
        cache = CacheManager()
        cache.set("test", "value")
        print("✅ Cache manager working")
        
        rate_limiter = RateLimiter()
        print("✅ Rate limiter initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_api_structure():
    """Test that the refactored API can be imported."""
    print("\n🚀 Testing API structure...")
    
    try:
        # Test that we can import the refactored API
        import api_refactored
        print("✅ Refactored API imported successfully")
        
        # Test that we can import the refactored utils
        import utils_refactored
        print("✅ Refactored utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔄 Testing Instagram Captions API v10.0 - Modular Structure")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Module Functionality", test_functionality),
        ("API Structure", test_api_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular structure is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






