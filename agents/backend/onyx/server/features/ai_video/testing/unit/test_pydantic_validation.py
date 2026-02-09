from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import sys
from pathlib import Path
        from pydantic_schemas import (
        from pydantic_schemas import VideoGenerationInput
        from pydantic import ValidationError
        from pydantic_validation import (
        from pydantic_schemas import (
        from pydantic_validation import ValidationPerformanceMonitor
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Test Pydantic Validation System
===============================

Quick test to verify the Pydantic validation system works correctly.
"""


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_validation():
    """Test basic Pydantic validation functionality."""
    print("🧪 Testing Basic Pydantic Validation...")
    
    try:
            VideoGenerationInput, VideoGenerationResponse,
            VideoStatus, QualityLevel, ModelType
        )
        
        # Test valid input
        valid_input = VideoGenerationInput(
            prompt="A beautiful sunset over mountains",
            quality=QualityLevel.HIGH,
            model_type=ModelType.STABLE_DIFFUSION
        )
        print(f"✅ Valid input created: {valid_input.prompt}")
        
        # Test response creation
        response = VideoGenerationResponse(
            video_id="test-123",
            status=VideoStatus.PROCESSING,
            message="Video generation started",
            progress=0.0
        )
        print(f"✅ Response created: {response.video_id}")
        
        # Test computed fields
        print(f"📊 Estimated size: {valid_input.estimated_size_mb:.2f} MB")
        print(f"📊 Total pixels: {valid_input.total_pixels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic validation test failed: {e}")
        return False

def test_validation_errors():
    """Test validation error handling."""
    print("\n🧪 Testing Validation Errors...")
    
    try:
        
        # Test invalid input
        try:
            invalid_input = VideoGenerationInput(
                prompt="",  # Empty prompt
                height=100,  # Invalid height
                width=100    # Invalid width
            )
            print("❌ Should have raised validation error")
            return False
        except ValidationError as e:
            print(f"✅ Validation error caught: {len(e.errors())} errors")
            return True
            
    except Exception as e:
        print(f"❌ Validation error test failed: {e}")
        return False

def test_middleware_creation():
    """Test middleware creation."""
    print("\n🧪 Testing Middleware Creation...")
    
    try:
            ValidationConfig, create_validation_middleware,
            ValidationPerformanceMonitor, create_performance_monitor
        )
        
        # Create configuration
        config = ValidationConfig(
            enable_request_validation=True,
            enable_response_validation=True,
            enable_performance_monitoring=True
        )
        print("✅ Configuration created")
        
        # Create middleware
        middleware = create_validation_middleware(config)
        print("✅ Middleware created")
        
        # Create performance monitor
        monitor = create_performance_monitor()
        print("✅ Performance monitor created")
        
        return True
        
    except Exception as e:
        print(f"❌ Middleware creation test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\n🧪 Testing Utility Functions...")
    
    try:
            create_video_id, create_batch_id,
            create_error_response, create_success_response,
            VideoStatus
        )
        
        # Test ID generation
        video_id = create_video_id()
        batch_id = create_batch_id()
        print(f"✅ Video ID generated: {video_id}")
        print(f"✅ Batch ID generated: {batch_id}")
        
        # Test error response
        error_response = create_error_response(
            error_code="TEST_ERROR",
            error_type="test",
            message="Test error message"
        )
        print(f"✅ Error response created: {error_response.error_code}")
        
        # Test success response
        success_response = create_success_response(
            video_id=video_id,
            status=VideoStatus.PROCESSING,
            message="Test success"
        )
        print(f"✅ Success response created: {success_response.video_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

async def test_async_validation():
    """Test async validation functionality."""
    print("\n🧪 Testing Async Validation...")
    
    try:
        
        # Create performance monitor
        monitor = create_performance_monitor()
        
        # Simulate async validation
        async with monitor.monitor_validation("TestSchema"):
            await asyncio.sleep(0.1)  # Simulate validation time
            monitor.record_cache_hit()
        
        # Get stats
        stats = monitor.get_stats()
        print(f"✅ Async validation completed: {stats['total_validations']} validations")
        
        return True
        
    except Exception as e:
        print(f"❌ Async validation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Running Pydantic Validation System Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Validation", test_basic_validation),
        ("Validation Errors", test_validation_errors),
        ("Middleware Creation", test_middleware_creation),
        ("Utility Functions", test_utility_functions),
    ]
    
    results = []
    
    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Run async test
    print(f"\n📋 Running Async Validation...")
    try:
        async_result = asyncio.run(test_async_validation())
        results.append(("Async Validation", async_result))
        if async_result:
            print(f"✅ Async Validation PASSED")
        else:
            print(f"❌ Async Validation FAILED")
    except Exception as e:
        print(f"❌ Async Validation ERROR: {e}")
        results.append(("Async Validation", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pydantic validation system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 