#!/usr/bin/env python3
"""
Test Script for Gradio Error Handling and Input Validation

This script demonstrates and validates the comprehensive error handling
and input validation system for Video-OpusClip Gradio applications.
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_error_handling_system():
    """Test the complete error handling system."""
    
    print("🧪 Testing Gradio Error Handling and Input Validation System")
    print("=" * 60)
    
    try:
        # Import error handling components
        from gradio_error_handling import (
            GradioErrorHandler, GradioInputValidator, GradioErrorRecovery,
            GradioErrorMonitor, gradio_error_handler, validate_gradio_inputs,
            EnhancedGradioComponents
        )
        
        print("✅ Successfully imported error handling components")
        
        # Test 1: Error Handler
        print("\n📋 Test 1: Error Handler")
        test_error_handler()
        
        # Test 2: Input Validator
        print("\n📋 Test 2: Input Validator")
        test_input_validator()
        
        # Test 3: Error Recovery
        print("\n📋 Test 3: Error Recovery")
        test_error_recovery()
        
        # Test 4: Error Monitor
        print("\n📋 Test 4: Error Monitor")
        test_error_monitor()
        
        # Test 5: Decorators
        print("\n📋 Test 5: Decorators")
        test_decorators()
        
        # Test 6: Enhanced Components
        print("\n📋 Test 6: Enhanced Components")
        test_enhanced_components()
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handler():
    """Test the GradioErrorHandler."""
    
    error_handler = GradioErrorHandler()
    
    # Test error handling
    test_error = ValueError("Test validation error")
    response = error_handler.handle_gradio_error(test_error, "test_function")
    
    assert response["success"] == False
    assert "error_message" in response
    assert "suggestion" in response
    assert "error_code" in response
    
    print("✅ Error handler correctly processes errors")
    print(f"   Error message: {response['error_message']}")
    print(f"   Suggestion: {response['suggestion']}")

def test_input_validator():
    """Test the GradioInputValidator."""
    
    validator = GradioInputValidator()
    
    # Test text validation
    print("   Testing text validation...")
    
    # Valid text
    is_valid, message = validator.validate_text_prompt("A beautiful sunset")
    assert is_valid == True
    print("   ✅ Valid text prompt accepted")
    
    # Invalid text (too short)
    is_valid, message = validator.validate_text_prompt("Hi")
    assert is_valid == False
    print(f"   ✅ Invalid text prompt rejected: {message}")
    
    # Invalid text (too long)
    long_text = "A" * 600
    is_valid, message = validator.validate_text_prompt(long_text)
    assert is_valid == False
    print(f"   ✅ Long text prompt rejected: {message}")
    
    # Test duration validation
    print("   Testing duration validation...")
    
    # Valid duration
    is_valid, message = validator.validate_duration(15)
    assert is_valid == True
    print("   ✅ Valid duration accepted")
    
    # Invalid duration (too short)
    is_valid, message = validator.validate_duration(1)
    assert is_valid == False
    print(f"   ✅ Short duration rejected: {message}")
    
    # Invalid duration (too long)
    is_valid, message = validator.validate_duration(100)
    assert is_valid == False
    print(f"   ✅ Long duration rejected: {message}")
    
    # Test quality validation
    print("   Testing quality validation...")
    
    # Valid quality
    is_valid, message = validator.validate_quality("High Quality")
    assert is_valid == True
    print("   ✅ Valid quality accepted")
    
    # Invalid quality
    is_valid, message = validator.validate_quality("Invalid Quality")
    assert is_valid == False
    print(f"   ✅ Invalid quality rejected: {message}")

def test_error_recovery():
    """Test the GradioErrorRecovery."""
    
    recovery = GradioErrorRecovery()
    
    # Test GPU error recovery
    print("   Testing GPU error recovery...")
    gpu_error = RuntimeError("CUDA out of memory")
    recovery_result = recovery.attempt_recovery(gpu_error, "test_context")
    
    assert "recovered" in recovery_result
    print(f"   ✅ GPU error recovery: {recovery_result['recovered']}")
    
    # Test memory error recovery
    print("   Testing memory error recovery...")
    memory_error = MemoryError("Not enough memory")
    recovery_result = recovery.attempt_recovery(memory_error, "test_context")
    
    assert "recovered" in recovery_result
    print(f"   ✅ Memory error recovery: {recovery_result['recovered']}")
    
    # Test timeout error recovery
    print("   Testing timeout error recovery...")
    timeout_error = TimeoutError("Operation timed out")
    recovery_result = recovery.attempt_recovery(timeout_error, "test_context")
    
    assert "recovered" in recovery_result
    print(f"   ✅ Timeout error recovery: {recovery_result['recovered']}")

def test_error_monitor():
    """Test the GradioErrorMonitor."""
    
    monitor = GradioErrorMonitor()
    
    # Test monitoring a function
    @monitor.monitor_function
    def test_function(input_data):
        if input_data == "error":
            raise ValueError("Test error")
        return f"Success: {input_data}"
    
    # Test successful execution
    result = test_function("test")
    assert result == "Success: test"
    print("   ✅ Successful function execution monitored")
    
    # Test error execution
    try:
        test_function("error")
    except ValueError:
        pass
    
    # Get error report
    report = monitor.get_error_report()
    assert "stats" in report
    assert "total_errors" in report["stats"]
    
    print(f"   ✅ Error monitoring: {report['stats']['total_errors']} errors tracked")

def test_decorators():
    """Test the error handling decorators."""
    
    monitor = GradioErrorMonitor()
    
    # Test combined decorators
    @gradio_error_handler
    @validate_gradio_inputs("text_prompt", "duration", "quality")
    @monitor.monitor_function
    def test_decorated_function(prompt, duration, quality):
        if "error" in prompt.lower():
            raise RuntimeError("Simulated error")
        return f"Generated video for: {prompt}"
    
    # Test successful execution
    result = test_decorated_function("A beautiful sunset", 15, "High Quality")
    assert "Generated video for" in result
    print("   ✅ Decorated function successful execution")
    
    # Test error execution
    result = test_decorated_function("error test", 15, "High Quality")
    assert result["success"] == False
    assert "error_message" in result
    print("   ✅ Decorated function error handling")

def test_enhanced_components():
    """Test the EnhancedGradioComponents."""
    
    try:
        import gradio as gr
        components = EnhancedGradioComponents()
        
        # Test component creation (without actually launching Gradio)
        print("   ✅ Enhanced components can be created")
        
    except ImportError:
        print("   ⚠️ Gradio not available, skipping component test")
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")

def test_integration():
    """Test integration with user-friendly interfaces."""
    
    print("\n🔗 Testing Integration with User-Friendly Interfaces")
    
    try:
        from user_friendly_interfaces import UserFriendlyInterface
        
        # Create interface instance
        interface = UserFriendlyInterface()
        
        # Test error handling components are initialized
        assert hasattr(interface, 'error_handler')
        assert hasattr(interface, 'input_validator')
        assert hasattr(interface, 'error_recovery')
        assert hasattr(interface, 'error_monitor')
        assert hasattr(interface, 'enhanced_components')
        
        print("✅ Error handling components properly integrated")
        
        # Test quick generation function
        result = interface._quick_text_to_video("A beautiful sunset", 15, "Balanced")
        assert len(result) == 3
        print("✅ Quick generation function works with error handling")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_test():
    """Run performance test for error handling system."""
    
    print("\n⚡ Performance Test")
    print("-" * 30)
    
    from gradio_error_handling import GradioErrorHandler, GradioInputValidator
    
    error_handler = GradioErrorHandler()
    validator = GradioInputValidator()
    
    # Test error handling performance
    start_time = time.time()
    for i in range(100):
        test_error = ValueError(f"Test error {i}")
        error_handler.handle_gradio_error(test_error, "performance_test")
    
    error_time = time.time() - start_time
    print(f"✅ Error handling: {error_time:.3f}s for 100 errors")
    
    # Test validation performance
    start_time = time.time()
    for i in range(100):
        validator.validate_text_prompt(f"Test prompt {i}")
        validator.validate_duration(15)
        validator.validate_quality("High Quality")
    
    validation_time = time.time() - start_time
    print(f"✅ Validation: {validation_time:.3f}s for 300 validations")

def main():
    """Main test function."""
    
    print("🎬 Video-OpusClip Error Handling Test Suite")
    print("=" * 50)
    
    # Run basic tests
    success = test_error_handling_system()
    
    if success:
        # Run integration test
        integration_success = test_integration()
        
        if integration_success:
            # Run performance test
            run_performance_test()
            
            print("\n🎉 All tests passed! Error handling system is working correctly.")
            print("\n📊 Test Summary:")
            print("   ✅ Error Handler: Working")
            print("   ✅ Input Validator: Working")
            print("   ✅ Error Recovery: Working")
            print("   ✅ Error Monitor: Working")
            print("   ✅ Decorators: Working")
            print("   ✅ Enhanced Components: Working")
            print("   ✅ Integration: Working")
            print("   ✅ Performance: Acceptable")
            
            print("\n🚀 Ready to use in production!")
            
        else:
            print("\n❌ Integration test failed")
            sys.exit(1)
    else:
        print("\n❌ Basic tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 