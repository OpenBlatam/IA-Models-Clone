#!/usr/bin/env python3
"""
Test script for Error Handling and Input Validation in Gradio Apps
Comprehensive testing of the error handling system and input validation
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the parent directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_llm_seo_engine import (
    GradioErrorHandler, 
    InputValidator, 
    GradioErrorBoundary
)

class TestGradioErrorHandler(unittest.TestCase):
    """Test cases for GradioErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = GradioErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        self.assertIsInstance(self.error_handler.error_log, list)
        self.assertEqual(self.error_handler.max_error_log_size, 100)
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        error = ValueError("Test error message")
        result = self.error_handler.handle_error(error, "test_context")
        
        self.assertIn("error", result)
        self.assertTrue(result["error"])
        self.assertIn("message", result)
        self.assertIn("error_code", result)
        self.assertIn("suggestions", result)
    
    def test_handle_error_cuda_error(self):
        """Test CUDA error handling."""
        error = RuntimeError("CUDA out of memory")
        result = self.error_handler.handle_error(error, "gpu_operation")
        
        self.assertIn("GPU memory or compatibility issue", result["message"])
        self.assertEqual(result["error_code"], "GPU_001")
    
    def test_handle_error_memory_error(self):
        """Test memory error handling."""
        error = MemoryError("Memory limit exceeded")
        result = self.error_handler.handle_error(error, "memory_operation")
        
        self.assertIn("Memory limit exceeded", result["message"])
        self.assertEqual(result["error_code"], "MEM_001")
    
    def test_handle_error_validation_error(self):
        """Test validation error handling."""
        error = ValueError("Invalid input format")
        result = self.error_handler.handle_error(error, "validation")
        
        self.assertIn("Invalid input provided", result["message"])
        self.assertEqual(result["error_code"], "VAL_001")
    
    def test_error_logging(self):
        """Test error logging functionality."""
        initial_log_size = len(self.error_handler.error_log)
        error = ValueError("Test error")
        
        self.error_handler.handle_error(error, "test")
        
        self.assertEqual(len(self.error_handler.error_log), initial_log_size + 1)
        self.assertEqual(self.error_handler.error_log[-1]["error_type"], "ValueError")
    
    def test_error_log_size_limit(self):
        """Test error log size limiting."""
        # Fill the log beyond the limit
        for i in range(150):
            error = ValueError(f"Error {i}")
            self.error_handler.handle_error(error, f"context_{i}")
        
        self.assertLessEqual(len(self.error_handler.error_log), self.error_handler.max_error_log_size)
    
    def test_get_error_summary(self):
        """Test error summary generation."""
        # Add some errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            self.error_handler.handle_error(error, f"context_{i}")
        
        summary = self.error_handler.get_error_summary()
        
        self.assertIn("total_errors", summary)
        self.assertIn("error_types", summary)
        self.assertIn("recent_errors", summary)
        self.assertEqual(summary["total_errors"], 5)
    
    def test_development_mode_detection(self):
        """Test development mode detection."""
        # Test with environment variable not set
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.error_handler._is_development_mode())
        
        # Test with environment variable set to true
        with patch.dict(os.environ, {"GRADIO_DEBUG": "true"}, clear=True):
            self.assertTrue(self.error_handler._is_development_mode())

class TestInputValidator(unittest.TestCase):
    """Test cases for InputValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIn("text", self.validator.validation_rules)
        self.assertIn("url", self.validator.validation_rules)
        self.assertIn("email", self.validator.validation_rules)
        self.assertIn("number", self.validator.validation_rules)
        self.assertIn("file_path", self.validator.validation_rules)
        self.assertIn("json", self.validator.validation_rules)
    
    def test_validate_text_success(self):
        """Test successful text validation."""
        is_valid, error_msg = self.validator.validate_text("Valid text", "test_field")
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)
    
    def test_validate_text_empty(self):
        """Test text validation with empty input."""
        is_valid, error_msg = self.validator.validate_text("", "test_field")
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", error_msg)
    
    def test_validate_text_too_long(self):
        """Test text validation with overly long input."""
        long_text = "a" * 15000
        is_valid, error_msg = self.validator.validate_text(long_text, "test_field")
        self.assertFalse(is_valid)
        self.assertIn("must be no more than", error_msg)
    
    def test_validate_url_success(self):
        """Test successful URL validation."""
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://sub.domain.co.uk/query?param=value"
        ]
        
        for url in valid_urls:
            is_valid, error_msg = self.validator.validate_url(url, "test_url")
            self.assertTrue(is_valid, f"URL {url} should be valid")
            self.assertIsNone(error_msg)
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "not_a_url",
            "ftp://example.com",
            "https://",
            "http://invalid",
            "https://example.com:invalid_port"
        ]
        
        for url in invalid_urls:
            is_valid, error_msg = self.validator.validate_url(url, "test_url")
            self.assertFalse(is_valid, f"URL {url} should be invalid")
            self.assertIsNotNone(error_msg)
    
    def test_validate_email_success(self):
        """Test successful email validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@example.co.uk",
            "123@test-domain.net"
        ]
        
        for email in valid_emails:
            is_valid, error_msg = self.validator.validate_email(email, "test_email")
            self.assertTrue(is_valid, f"Email {email} should be valid")
            self.assertIsNone(error_msg)
    
    def test_validate_email_invalid(self):
        """Test email validation with invalid emails."""
        invalid_emails = [
            "invalid_email",
            "@example.com",
            "user@",
            "user@.com",
            "user space@example.com"
        ]
        
        for email in invalid_emails:
            is_valid, error_msg = self.validator.validate_email(email, "test_email")
            self.assertFalse(is_valid, f"Email {email} should be invalid")
            self.assertIsNotNone(error_msg)
    
    def test_validate_number_success(self):
        """Test successful number validation."""
        test_cases = [
            (42, 0, 100, False),  # Integer within range
            (3.14, 0, 10, False),  # Float within range
            (0, 0, 100, True),     # Integer at boundary
            (100, 0, 100, True),   # Integer at boundary
        ]
        
        for value, min_val, max_val, integer_only in test_cases:
            is_valid, error_msg = self.validator.validate_number(
                value, "test_number", min_val, max_val, integer_only
            )
            self.assertTrue(is_valid, f"Number {value} should be valid")
            self.assertIsNone(error_msg)
    
    def test_validate_number_invalid(self):
        """Test number validation with invalid inputs."""
        test_cases = [
            (None, 0, 100, False),  # None value
            (150, 0, 100, False),   # Above maximum
            (-5, 0, 100, False),    # Below minimum
            (3.14, 0, 100, True),   # Float when integer required
        ]
        
        for value, min_val, max_val, integer_only in test_cases:
            is_valid, error_msg = self.validator.validate_number(
                value, "test_number", min_val, max_val, integer_only
            )
            self.assertFalse(is_valid, f"Number {value} should be invalid")
            self.assertIsNotNone(error_msg)
    
    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            is_valid, error_msg = self.validator.validate_file_path(temp_path, "test_file")
            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_path_invalid_extension(self):
        """Test file path validation with invalid extension."""
        is_valid, error_msg = self.validator.validate_file_path("test.xyz", "test_file")
        self.assertFalse(is_valid)
        self.assertIn("must have one of these extensions", error_msg)
    
    def test_validate_json_success(self):
        """Test successful JSON validation."""
        valid_jsons = [
            '{"key": "value"}',
            '[1, 2, 3]',
            '{"nested": {"key": "value"}}',
            '{"array": [1, 2, {"nested": 3}]}'
        ]
        
        for json_str in valid_jsons:
            is_valid, error_msg = self.validator.validate_json(json_str, "test_json")
            self.assertTrue(is_valid, f"JSON {json_str} should be valid")
            self.assertIsNone(error_msg)
    
    def test_validate_json_invalid(self):
        """Test JSON validation with invalid inputs."""
        invalid_jsons = [
            '{"key": "value"',  # Missing closing brace
            '{"key": value}',   # Missing quotes
            '[1, 2, 3',         # Missing closing bracket
            'not json at all'    # Plain text
        ]
        
        for json_str in invalid_jsons:
            is_valid, error_msg = self.validator.validate_json(json_str, "test_json")
            self.assertFalse(is_valid, f"JSON {json_str} should be invalid")
            self.assertIsNotNone(error_msg)
    
    def test_validate_json_too_deep(self):
        """Test JSON validation with overly deep nesting."""
        # Create deeply nested JSON
        deep_json = '{"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": {"level11": "value"}}}}}}}}}}}'
        
        is_valid, error_msg = self.validator.validate_json(deep_json, "test_json")
        self.assertFalse(is_valid)
        self.assertIn("must not exceed", error_msg)
    
    def test_validate_seo_inputs_success(self):
        """Test successful SEO inputs validation."""
        valid_inputs = {
            "content": "Valid content for SEO analysis",
            "title": "SEO Title",
            "description": "SEO description",
            "max_length": 1000,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "num_epochs": 10,
            "metadata": '{"category": "tech", "language": "en"}'
        }
        
        is_valid, errors = self.validator.validate_seo_inputs(valid_inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_seo_inputs_failure(self):
        """Test SEO inputs validation with invalid inputs."""
        invalid_inputs = {
            "content": "",  # Empty content
            "max_length": 15000,  # Too long
            "batch_size": 200,    # Too large
            "learning_rate": 1e-1,  # Too high
            "metadata": '{"invalid": json'  # Invalid JSON
        }
        
        is_valid, errors = self.validator.validate_seo_inputs(invalid_inputs)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_json_depth_calculation(self):
        """Test JSON depth calculation."""
        test_cases = [
            ('{"key": "value"}', 1),
            ('{"nested": {"key": "value"}}', 2),
            ('{"level1": {"level2": {"level3": "value"}}}', 3),
            ('[1, 2, 3]', 1),
            ('[{"key": "value"}, {"nested": {"key": "value"}}]', 3)
        ]
        
        for json_str, expected_depth in test_cases:
            parsed = json.loads(json_str)
            actual_depth = self.validator._get_json_depth(parsed)
            self.assertEqual(actual_depth, expected_depth, 
                           f"Expected depth {expected_depth} for {json_str}")
    
    def test_json_item_count(self):
        """Test JSON item count calculation."""
        test_cases = [
            ('{"key": "value"}', 2),  # 1 key + 1 value
            ('{"key1": "value1", "key2": "value2"}', 4),  # 2 keys + 2 values
            ('[1, 2, 3]', 3),  # 3 items
            ('{"array": [1, 2, 3], "nested": {"key": "value"}}', 7)  # Complex structure
        ]
        
        for json_str, expected_count in test_cases:
            parsed = json.loads(json_str)
            actual_count = self.validator._count_json_items(parsed)
            self.assertEqual(actual_count, expected_count,
                           f"Expected count {expected_count} for {json_str}")

class TestGradioErrorBoundary(unittest.TestCase):
    """Test cases for GradioErrorBoundary class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = GradioErrorHandler()
        self.error_boundary = GradioErrorBoundary(self.error_handler)
    
    def test_error_boundary_initialization(self):
        """Test error boundary initialization."""
        self.assertEqual(self.error_boundary.error_handler, self.error_handler)
    
    def test_error_boundary_decorator_success(self):
        """Test error boundary decorator with successful function."""
        @self.error_boundary
        def successful_function():
            return "success"
        
        result = successful_function()
        self.assertEqual(result, "success")
    
    def test_error_boundary_decorator_failure(self):
        """Test error boundary decorator with failing function."""
        @self.error_boundary
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        self.assertIn("error", result)
        self.assertTrue(result["error"])
        self.assertIn("message", result)
    
    def test_error_boundary_context_capture(self):
        """Test error boundary context capture."""
        @self.error_boundary
        def function_with_args(arg1, arg2, arg3):
            raise RuntimeError("Test error")
        
        result = function_with_args("test", "args", "here")
        self.assertIn("Function: function_with_args", result["details"])
    
    def test_error_boundary_preserves_arguments(self):
        """Test that error boundary preserves function arguments."""
        @self.error_boundary
        def function_with_args(arg1, arg2):
            return f"{arg1} {arg2}"
        
        result = function_with_args("hello", "world")
        self.assertEqual(result, "hello world")

class TestIntegration(unittest.TestCase):
    """Integration tests for error handling and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = GradioErrorHandler()
        self.validator = InputValidator()
        self.error_boundary = GradioErrorBoundary(self.error_handler)
    
    def test_error_handling_with_validation(self):
        """Test error handling integrated with validation."""
        @self.error_boundary
        def validate_and_process(data):
            # Validate input
            is_valid, errors = self.validator.validate_seo_inputs(data)
            if not is_valid:
                raise ValueError(f"Validation failed: {errors}")
            
            # Process data
            return {"processed": True, "data": data}
        
        # Test with invalid data
        invalid_data = {"content": "", "max_length": 15000}
        result = validate_and_process(invalid_data)
        
        self.assertIn("error", result)
        self.assertTrue(result["error"])
        self.assertIn("Validation failed", result["message"])
    
    def test_comprehensive_error_scenarios(self):
        """Test comprehensive error scenarios."""
        error_scenarios = [
            (ValueError("Invalid input"), "input_validation"),
            (RuntimeError("CUDA out of memory"), "gpu_operation"),
            (MemoryError("Memory limit exceeded"), "memory_operation"),
            (ConnectionError("Connection failed"), "network_operation"),
            (FileNotFoundError("File not found"), "file_operation"),
            (PermissionError("Permission denied"), "file_operation")
        ]
        
        for error, context in error_scenarios:
            result = self.error_handler.handle_error(error, context)
            
            self.assertIn("error", result)
            self.assertTrue(result["error"])
            self.assertIn("message", result)
            self.assertIn("error_code", result)
            self.assertIn("suggestions", result)
    
    def test_validation_rules_consistency(self):
        """Test validation rules consistency across different input types."""
        # Test that all validation methods exist for each rule type
        for rule_type in self.validator.validation_rules:
            validation_method = f"validate_{rule_type}"
            self.assertTrue(hasattr(self.validator, validation_method),
                          f"Missing validation method: {validation_method}")
    
    def test_error_log_persistence(self):
        """Test error log persistence across multiple operations."""
        initial_errors = len(self.error_handler.error_log)
        
        # Generate multiple errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            self.error_handler.handle_error(error, f"context_{i}")
        
        # Verify errors are logged
        self.assertEqual(len(self.error_handler.error_log), initial_errors + 5)
        
        # Verify error details are preserved
        for i, error_entry in enumerate(self.error_handler.error_log[-5:]):
            self.assertEqual(error_entry["error_type"], "ValueError")
            self.assertIn(f"Error {i}", error_entry["error_message"])

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Error Handling and Input Validation Tests...")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGradioErrorHandler,
        TestInputValidator,
        TestGradioErrorBoundary,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests passed! Error handling and input validation system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the error handling and input validation implementation.")
        sys.exit(1)






