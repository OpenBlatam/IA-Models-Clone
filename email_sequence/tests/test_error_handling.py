from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import unittest
import asyncio
import tempfile
import json
import pandas as pd
from pathlib import Path
import sys
import os
from core.error_handling import (
        import shutil
from typing import Any, List, Dict, Optional
import logging
"""
Error Handling and Validation Test Suite

Comprehensive tests for error handling, input validation, and debugging
capabilities in the email sequence system.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

    ErrorHandler, InputValidator, DataLoaderErrorHandler,
    ModelInferenceErrorHandler, GradioErrorHandler,
    ValidationError, ModelError, DataError, ConfigurationError,
    handle_async_operation, handle_model_operation, handle_data_operation
)


class TestErrorHandler(unittest.TestCase):
    """Test cases for the ErrorHandler class"""
    
    def setUp(self) -> Any:
        """Set up test fixtures"""
        self.error_handler = ErrorHandler(debug_mode=True)
    
    def test_error_logging(self) -> Any:
        """Test error logging functionality"""
        
        # Test logging a simple error
        test_error = ValueError("Test error message")
        self.error_handler.log_error(test_error, "Test context", "test_operation")
        
        # Check that error was logged
        self.assertEqual(len(self.error_handler.error_log), 1)
        
        error_info = self.error_handler.error_log[0]
        self.assertEqual(error_info["error_type"], "ValueError")
        self.assertEqual(error_info["error_message"], "Test error message")
        self.assertEqual(error_info["context"], "Test context")
        self.assertEqual(error_info["operation"], "test_operation")
        self.assertIsNotNone(error_info["timestamp"])
        self.assertIsNotNone(error_info["traceback"])
    
    def test_safe_execute_success(self) -> Any:
        """Test safe execution with successful operation"""
        
        def test_function(x, y) -> Any:
            return x + y
        
        result, error = self.error_handler.safe_execute(test_function, 5, 3, context="Addition test")
        
        self.assertEqual(result, 8)
        self.assertIsNone(error)
        self.assertEqual(len(self.error_handler.error_log), 0)
    
    def test_safe_execute_failure(self) -> Any:
        """Test safe execution with failed operation"""
        
        def test_function(x, y) -> Any:
            return x / y
        
        result, error = self.error_handler.safe_execute(test_function, 5, 0, context="Division test")
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(len(self.error_handler.error_log), 1)
    
    def test_safe_async_execute_success(self) -> Any:
        """Test safe async execution with successful operation"""
        
        async def test_async_function(x, y) -> Any:
            await asyncio.sleep(0.1)
            return x * y
        
        result, error = asyncio.run(
            self.error_handler.safe_async_execute(test_async_function, 4, 5, context="Async multiplication test")
        )
        
        self.assertEqual(result, 20)
        self.assertIsNone(error)
        self.assertEqual(len(self.error_handler.error_log), 0)
    
    def test_safe_async_execute_failure(self) -> Any:
        """Test safe async execution with failed operation"""
        
        async def test_async_function(x, y) -> Any:
            await asyncio.sleep(0.1)
            raise ValueError("Async test error")
        
        result, error = asyncio.run(
            self.error_handler.safe_async_execute(test_async_function, 4, 5, context="Async error test")
        )
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(len(self.error_handler.error_log), 1)
    
    def test_error_summary(self) -> Any:
        """Test error summary generation"""
        
        # Log some test errors
        self.error_handler.log_error(ValueError("Error 1"), "Context 1", "Operation 1")
        self.error_handler.log_error(TypeError("Error 2"), "Context 2", "Operation 2")
        self.error_handler.log_error(ValueError("Error 3"), "Context 3", "Operation 3")
        
        summary = self.error_handler.get_error_summary()
        
        self.assertEqual(summary["total_errors"], 3)
        self.assertEqual(len(summary["recent_errors"]), 3)
        self.assertEqual(summary["error_type_distribution"]["ValueError"], 2)
        self.assertEqual(summary["error_type_distribution"]["TypeError"], 1)
        self.assertIsNotNone(summary["last_error"])


class TestInputValidator(unittest.TestCase):
    """Test cases for the InputValidator class"""
    
    def setUp(self) -> Any:
        """Set up test fixtures"""
        self.validator = InputValidator()
    
    def test_validate_model_type_valid(self) -> bool:
        """Test model type validation with valid inputs"""
        
        valid_models = ["GPT-3.5", "GPT-4", "Claude", "Custom", "Custom Model"]
        
        for model in valid_models:
            is_valid, error_msg = self.validator.validate_model_type(model)
            self.assertTrue(is_valid, f"Model {model} should be valid")
            self.assertEqual(error_msg, "")
    
    def test_validate_model_type_invalid(self) -> bool:
        """Test model type validation with invalid inputs"""
        
        invalid_models = ["", None, "Invalid Model", "GPT-5", "Claude-2"]
        
        for model in invalid_models:
            is_valid, error_msg = self.validator.validate_model_type(model)
            self.assertFalse(is_valid, f"Model {model} should be invalid")
            self.assertNotEqual(error_msg, "")
    
    def test_validate_sequence_length_valid(self) -> bool:
        """Test sequence length validation with valid inputs"""
        
        valid_lengths = [1, 3, 5, 7, 10]
        
        for length in valid_lengths:
            is_valid, error_msg = self.validator.validate_sequence_length(length)
            self.assertTrue(is_valid, f"Length {length} should be valid")
            self.assertEqual(error_msg, "")
    
    def test_validate_sequence_length_invalid(self) -> bool:
        """Test sequence length validation with invalid inputs"""
        
        invalid_lengths = [0, -1, 11, 15, "5", 3.5]
        
        for length in invalid_lengths:
            is_valid, error_msg = self.validator.validate_sequence_length(length)
            self.assertFalse(is_valid, f"Length {length} should be invalid")
            self.assertNotEqual(error_msg, "")
    
    def test_validate_creativity_level_valid(self) -> bool:
        """Test creativity level validation with valid inputs"""
        
        valid_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for level in valid_levels:
            is_valid, error_msg = self.validator.validate_creativity_level(level)
            self.assertTrue(is_valid, f"Level {level} should be valid")
            self.assertEqual(error_msg, "")
    
    def test_validate_creativity_level_invalid(self) -> bool:
        """Test creativity level validation with invalid inputs"""
        
        invalid_levels = [0.0, -0.1, 1.1, 2.0, "0.5", None]
        
        for level in invalid_levels:
            is_valid, error_msg = self.validator.validate_creativity_level(level)
            self.assertFalse(is_valid, f"Level {level} should be invalid")
            self.assertNotEqual(error_msg, "")
    
    def test_validate_subscriber_data_valid(self) -> bool:
        """Test subscriber data validation with valid inputs"""
        
        valid_subscriber = {
            "id": "test_123",
            "email": "test@example.com",
            "name": "Test User",
            "company": "Test Company"
        }
        
        is_valid, error_msg = self.validator.validate_subscriber_data(valid_subscriber)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_subscriber_data_invalid(self) -> bool:
        """Test subscriber data validation with invalid inputs"""
        
        # Test missing fields
        invalid_subscriber = {
            "id": "test_123",
            "email": "test@example.com"
            # Missing name and company
        }
        
        is_valid, error_msg = self.validator.validate_subscriber_data(invalid_subscriber)
        self.assertFalse(is_valid)
        self.assertNotEqual(error_msg, "")
        
        # Test invalid email
        invalid_email_subscriber = {
            "id": "test_123",
            "email": "invalid-email",
            "name": "Test User",
            "company": "Test Company"
        }
        
        is_valid, error_msg = self.validator.validate_subscriber_data(invalid_email_subscriber)
        self.assertFalse(is_valid)
        self.assertNotEqual(error_msg, "")
    
    def test_validate_training_config_valid(self) -> bool:
        """Test training config validation with valid inputs"""
        
        valid_config = {
            "max_epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        is_valid, error_msg = self.validator.validate_training_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_training_config_invalid(self) -> bool:
        """Test training config validation with invalid inputs"""
        
        # Test missing fields
        invalid_config = {
            "max_epochs": 100
            # Missing batch_size and learning_rate
        }
        
        is_valid, error_msg = self.validator.validate_training_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertNotEqual(error_msg, "")
        
        # Test invalid values
        invalid_values_config = {
            "max_epochs": 0,  # Must be at least 1
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        is_valid, error_msg = self.validator.validate_training_config(invalid_values_config)
        self.assertFalse(is_valid)
        self.assertNotEqual(error_msg, "")


class TestDataLoaderErrorHandler(unittest.TestCase):
    """Test cases for the DataLoaderErrorHandler class"""
    
    def setUp(self) -> Any:
        """Set up test fixtures"""
        self.error_handler = ErrorHandler(debug_mode=True)
        self.data_handler = DataLoaderErrorHandler(self.error_handler)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_safe_load_csv_success(self) -> Any:
        """Test successful CSV loading"""
        
        # Create a test CSV file
        test_data = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob'],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'company': ['Company A', 'Company B', 'Company C']
        })
        
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        test_data.to_csv(csv_path, index=False)
        
        # Test loading
        result, error = self.data_handler.safe_load_csv(csv_path)
        
        self.assertIsNotNone(result)
        self.assertIsNone(error)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['name', 'email', 'company'])
    
    def test_safe_load_csv_file_not_found(self) -> Any:
        """Test CSV loading with non-existent file"""
        
        non_existent_path = os.path.join(self.temp_dir, 'non_existent.csv')
        result, error = self.data_handler.safe_load_csv(non_existent_path)
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn("File not found", error)
    
    def test_safe_load_csv_empty_file(self) -> Any:
        """Test CSV loading with empty file"""
        
        # Create an empty CSV file
        empty_csv_path = os.path.join(self.temp_dir, 'empty.csv')
        with open(empty_csv_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write('')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        result, error = self.data_handler.safe_load_csv(empty_csv_path)
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn("empty", error.lower())
    
    def test_safe_load_json_success(self) -> Any:
        """Test successful JSON loading"""
        
        # Create a test JSON file
        test_data = {
            "name": "Test User",
            "email": "test@example.com",
            "settings": {
                "model_type": "GPT-3.5",
                "creativity": 0.7
            }
        }
        
        json_path = os.path.join(self.temp_dir, 'test.json')
        with open(json_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(test_data, f)
        
        # Test loading
        result, error = self.data_handler.safe_load_json(json_path)
        
        self.assertIsNotNone(result)
        self.assertIsNone(error)
        self.assertEqual(result["name"], "Test User")
        self.assertEqual(result["settings"]["model_type"], "GPT-3.5")
    
    def test_safe_load_json_invalid_format(self) -> Any:
        """Test JSON loading with invalid format"""
        
        # Create an invalid JSON file
        invalid_json_path = os.path.join(self.temp_dir, 'invalid.json')
        with open(invalid_json_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write('{"name": "Test", "invalid": json}')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        result, error = self.data_handler.safe_load_json(invalid_json_path)
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn("Invalid JSON format", error)
    
    def test_safe_save_data_json_success(self) -> Any:
        """Test successful JSON saving"""
        
        test_data = {"name": "Test", "value": 123}
        json_path = os.path.join(self.temp_dir, 'output.json')
        
        success, error = self.data_handler.safe_save_data(test_data, json_path, "json")
        
        self.assertTrue(success)
        self.assertIsNone(error)
        
        # Verify file was created
        self.assertTrue(os.path.exists(json_path))
        
        # Verify content
        with open(json_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_data)
    
    def test_safe_save_data_csv_success(self) -> Any:
        """Test successful CSV saving"""
        
        test_data = pd.DataFrame({
            'name': ['John', 'Jane'],
            'email': ['john@test.com', 'jane@test.com']
        })
        
        csv_path = os.path.join(self.temp_dir, 'output.csv')
        
        success, error = self.data_handler.safe_save_data(test_data, csv_path, "csv")
        
        self.assertTrue(success)
        self.assertIsNone(error)
        
        # Verify file was created
        self.assertTrue(os.path.exists(csv_path))
    
    def test_safe_save_data_unsupported_type(self) -> Any:
        """Test saving with unsupported file type"""
        
        test_data = {"name": "Test"}
        file_path = os.path.join(self.temp_dir, 'output.txt')
        
        success, error = self.data_handler.safe_save_data(test_data, file_path, "txt")
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("Unsupported file type", error)


class TestModelInferenceErrorHandler(unittest.TestCase):
    """Test cases for the ModelInferenceErrorHandler class"""
    
    def setUp(self) -> Any:
        """Set up test fixtures"""
        self.error_handler = ErrorHandler(debug_mode=True)
        self.model_handler = ModelInferenceErrorHandler(self.error_handler)
    
    def test_safe_model_load_file_not_found(self) -> Any:
        """Test model loading with non-existent file"""
        
        non_existent_path = "non_existent_model.pth"
        result, error = self.model_handler.safe_model_load(non_existent_path)
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn("Model file not found", error)
    
    def test_safe_model_load_unsupported_type(self) -> Any:
        """Test model loading with unsupported model type"""
        
        # Create a dummy file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            dummy_path = f.name
        
        try:
            result, error = self.model_handler.safe_model_load(dummy_path, "unsupported_type")
            
            self.assertIsNone(result)
            self.assertIsNotNone(error)
            self.assertIn("Unsupported model type", error)
        finally:
            os.unlink(dummy_path)


class TestGradioErrorHandler(unittest.TestCase):
    """Test cases for the GradioErrorHandler class"""
    
    def setUp(self) -> Any:
        """Set up test fixtures"""
        self.error_handler = ErrorHandler(debug_mode=True)
        self.gradio_handler = GradioErrorHandler(self.error_handler, debug_mode=True)
    
    def test_format_gradio_error(self) -> Any:
        """Test Gradio error formatting"""
        
        error_type = "Test Error"
        message = "Test error message"
        
        formatted_error = self.gradio_handler._format_gradio_error(error_type, message)
        
        self.assertTrue(formatted_error["error"])
        self.assertEqual(formatted_error["error_type"], error_type)
        self.assertEqual(formatted_error["message"], message)
        self.assertIn("timestamp", formatted_error)
    
    def test_validate_gradio_inputs_valid(self) -> bool:
        """Test Gradio input validation with valid inputs"""
        
        valid_inputs = {
            "model_type": "GPT-3.5",
            "sequence_length": 5,
            "creativity_level": 0.7
        }
        
        is_valid, errors = self.gradio_handler.validate_gradio_inputs(valid_inputs)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_gradio_inputs_invalid(self) -> bool:
        """Test Gradio input validation with invalid inputs"""
        
        invalid_inputs = {
            "model_type": "Invalid Model",
            "sequence_length": 15,  # Out of range
            "creativity_level": 1.5  # Out of range
        }
        
        is_valid, errors = self.gradio_handler.validate_gradio_inputs(invalid_inputs)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestDecorators(unittest.TestCase):
    """Test cases for error handling decorators"""
    
    def test_handle_async_operation_success(self) -> Any:
        """Test async operation decorator with success"""
        
        @handle_async_operation
        async def test_async_function(x, y) -> Any:
            await asyncio.sleep(0.1)
            return x + y
        
        result = asyncio.run(test_async_function(3, 4))
        self.assertEqual(result, 7)
    
    def test_handle_async_operation_failure(self) -> Any:
        """Test async operation decorator with failure"""
        
        @handle_async_operation
        async def test_async_function(x, y) -> Any:
            await asyncio.sleep(0.1)
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            asyncio.run(test_async_function(3, 4))
    
    def test_handle_data_operation_success(self) -> Any:
        """Test data operation decorator with success"""
        
        @handle_data_operation
        def test_data_function(data) -> Any:
            return data.upper()
        
        result = test_data_function("test")
        self.assertEqual(result, "TEST")
    
    def test_handle_data_operation_failure(self) -> Any:
        """Test data operation decorator with failure"""
        
        @handle_data_operation
        def test_data_function(data) -> Any:
            raise FileNotFoundError("Test file not found")
        
        with self.assertRaises(DataError):
            test_data_function("test")


def run_error_handling_tests():
    """Run all error handling tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestErrorHandler,
        TestInputValidator,
        TestDataLoaderErrorHandler,
        TestModelInferenceErrorHandler,
        TestGradioErrorHandler,
        TestDecorators
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Error Handling Test Suite...")
    success = run_error_handling_tests()
    
    if success:
        print("\n✅ All error handling tests passed!")
    else:
        print("\n❌ Some error handling tests failed!")
    
    sys.exit(0 if success else 1) 