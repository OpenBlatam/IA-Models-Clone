from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from ..gradio_error_handling import (
from ..core import ValidationError, ConfigurationError, WorkflowError
from ..models import VideoRequest
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Test Suite for Gradio Error Handling and Input Validation

Comprehensive tests for the Gradio error handling system, input validation,
and error categorization functionality.
"""


# Local imports
    GradioErrorHandler, GradioInputValidator, InputValidationRule,
    ErrorSeverity, ErrorCategory, GradioErrorInfo,
    gradio_error_handler, gradio_input_validator,
    handle_gradio_error, validate_gradio_inputs,
    create_gradio_error_components, update_error_display
)


class TestErrorSeverity:
    """Test error severity enumeration."""
    
    def test_error_severity_values(self) -> Any:
        """Test error severity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Test error category enumeration."""
    
    def test_error_category_values(self) -> Any:
        """Test error category enum values."""
        assert ErrorCategory.INPUT_VALIDATION.value == "input_validation"
        assert ErrorCategory.FILE_PROCESSING.value == "file_processing"
        assert ErrorCategory.MODEL_LOADING.value == "model_loading"
        assert ErrorCategory.GPU_MEMORY.value == "gpu_memory"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestGradioErrorInfo:
    """Test GradioErrorInfo model."""
    
    def test_gradio_error_info_creation(self) -> Any:
        """Test creating GradioErrorInfo instance."""
        error_info = GradioErrorInfo(
            error_id="test_123",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INPUT_VALIDATION,
            user_message="Test user message",
            technical_message="Test technical message",
            error_code="TEST_001",
            suggestions=["Suggestion 1", "Suggestion 2"],
            retry_allowed=True,
            max_retries=3
        )
        
        assert error_info.error_id == "test_123"
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.category == ErrorCategory.INPUT_VALIDATION
        assert error_info.user_message == "Test user message"
        assert error_info.technical_message == "Test technical message"
        assert error_info.error_code == "TEST_001"
        assert len(error_info.suggestions) == 2
        assert error_info.retry_allowed is True
        assert error_info.max_retries == 3
    
    def test_gradio_error_info_defaults(self) -> Any:
        """Test GradioErrorInfo with default values."""
        error_info = GradioErrorInfo(
            error_id="test_456",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            user_message="Test message",
            technical_message="Test technical"
        )
        
        assert error_info.error_code is None
        assert error_info.suggestions == []
        assert error_info.retry_allowed is True
        assert error_info.max_retries == 3


class TestInputValidationRule:
    """Test InputValidationRule model."""
    
    def test_input_validation_rule_creation(self) -> Any:
        """Test creating InputValidationRule instance."""
        rule = InputValidationRule(
            field_name="test_field",
            required=True,
            min_length=10,
            max_length=100,
            min_value=1,
            max_value=100,
            allowed_values=["a", "b", "c"],
            file_types=["txt", "pdf"],
            max_file_size=1024 * 1024,
            custom_validator=lambda x: None
        )
        
        assert rule.field_name == "test_field"
        assert rule.required is True
        assert rule.min_length == 10
        assert rule.max_length == 100
        assert rule.min_value == 1
        assert rule.max_value == 100
        assert rule.allowed_values == ["a", "b", "c"]
        assert rule.file_types == ["txt", "pdf"]
        assert rule.max_file_size == 1024 * 1024
        assert rule.custom_validator is not None
    
    def test_input_validation_rule_defaults(self) -> Any:
        """Test InputValidationRule with default values."""
        rule = InputValidationRule(field_name="test_field")
        
        assert rule.field_name == "test_field"
        assert rule.required is True
        assert rule.min_length is None
        assert rule.max_length is None
        assert rule.min_value is None
        assert rule.max_value is None
        assert rule.allowed_values is None
        assert rule.file_types is None
        assert rule.max_file_size is None
        assert rule.custom_validator is None


class TestGradioErrorHandler:
    """Test GradioErrorHandler class."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.error_handler = GradioErrorHandler()
    
    def test_error_handler_initialization(self) -> Any:
        """Test error handler initialization."""
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.error_counts) == 0
        assert self.error_handler.max_history_size == 1000
        assert len(self.error_handler.error_templates) == 9
        assert len(self.error_handler.user_messages) == 12
    
    def test_create_error_info(self) -> Any:
        """Test creating error info."""
        error = ValidationError("Test validation error")
        error_info = self.error_handler.create_error_info(
            error=error,
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="User friendly message",
            technical_message="Technical details",
            error_code="VAL_001",
            suggestions=["Fix input", "Try again"]
        )
        
        assert error_info.error_id.startswith("input_validation_")
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.category == ErrorCategory.INPUT_VALIDATION
        assert error_info.user_message == "User friendly message"
        assert error_info.technical_message == "Technical details"
        assert error_info.error_code == "VAL_001"
        assert len(error_info.suggestions) == 2
    
    def test_can_retry_logic(self) -> Any:
        """Test retry logic for different error categories."""
        # Critical errors should not be retryable
        assert not self.error_handler._can_retry(ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL)
        
        # Permission errors should not be retryable
        assert not self.error_handler._can_retry(ErrorCategory.PERMISSION, ErrorSeverity.HIGH)
        
        # Network errors should be retryable
        assert self.error_handler._can_retry(ErrorCategory.NETWORK, ErrorSeverity.MEDIUM)
        
        # Timeout errors should be retryable
        assert self.error_handler._can_retry(ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM)
    
    def test_get_max_retries(self) -> Optional[Dict[str, Any]]:
        """Test maximum retry configuration."""
        assert self.error_handler._get_max_retries(ErrorCategory.NETWORK) == 3
        assert self.error_handler._get_max_retries(ErrorCategory.TIMEOUT) == 2
        assert self.error_handler._get_max_retries(ErrorCategory.GPU_MEMORY) == 1
        assert self.error_handler._get_max_retries(ErrorCategory.INPUT_VALIDATION) == 0
        assert self.error_handler._get_max_retries(ErrorCategory.PERMISSION) == 0
        assert self.error_handler._get_max_retries(ErrorCategory.UNKNOWN) == 1
    
    def test_categorize_error(self) -> Any:
        """Test error categorization."""
        # GPU memory error
        gpu_error = Exception("CUDA out of memory")
        category, severity = self.error_handler.categorize_error(gpu_error)
        assert category == ErrorCategory.GPU_MEMORY
        assert severity == ErrorSeverity.HIGH
        
        # Network error
        network_error = Exception("Connection timeout")
        category, severity = self.error_handler.categorize_error(network_error)
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.MEDIUM
        
        # File processing error
        file_error = Exception("Invalid file format")
        category, severity = self.error_handler.categorize_error(file_error)
        assert category == ErrorCategory.FILE_PROCESSING
        assert severity == ErrorSeverity.MEDIUM
        
        # Unknown error
        unknown_error = Exception("Some random error")
        category, severity = self.error_handler.categorize_error(unknown_error)
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM
    
    def test_get_user_friendly_message(self) -> Optional[Dict[str, Any]]:
        """Test user-friendly message generation."""
        # Validation error
        validation_error = ValidationError("Text too short")
        message = self.error_handler.get_user_friendly_message(
            validation_error, ErrorCategory.INPUT_VALIDATION
        )
        assert "check your input" in message.lower()
        
        # GPU memory error
        gpu_error = Exception("CUDA out of memory")
        message = self.error_handler.get_user_friendly_message(
            gpu_error, ErrorCategory.GPU_MEMORY
        )
        assert "gpu memory" in message.lower()
        
        # Network error
        network_error = Exception("Connection failed")
        message = self.error_handler.get_user_friendly_message(
            network_error, ErrorCategory.NETWORK
        )
        assert "network" in message.lower()
    
    def test_get_suggestions(self) -> Optional[Dict[str, Any]]:
        """Test suggestion generation."""
        suggestions = self.error_handler.get_suggestions(ErrorCategory.GPU_MEMORY)
        assert len(suggestions) > 0
        assert any("quality" in suggestion.lower() for suggestion in suggestions)
        
        suggestions = self.error_handler.get_suggestions(ErrorCategory.NETWORK)
        assert len(suggestions) > 0
        assert any("connection" in suggestion.lower() for suggestion in suggestions)
        
        suggestions = self.error_handler.get_suggestions(ErrorCategory.INPUT_VALIDATION)
        assert len(suggestions) > 0
        assert any("required" in suggestion.lower() for suggestion in suggestions)
    
    def test_format_error_for_gradio(self) -> Any:
        """Test error formatting for Gradio."""
        error = ValidationError("Test validation error")
        title, description, details = self.error_handler.format_error_for_gradio(error)
        
        assert "⚠️" in title
        assert "Invalid Input" in title
        assert "check your input" in description.lower()
        assert "Error ID:" in details
    
    def test_format_error_for_gradio_technical(self) -> Any:
        """Test error formatting with technical details."""
        error = ValidationError("Test validation error")
        title, description, details = self.error_handler.format_error_for_gradio(
            error, show_technical=True
        )
        
        assert "Test validation error" in details
        assert "Technical Details:" in details
    
    def test_store_error(self) -> Any:
        """Test error storage."""
        error = ValidationError("Test error")
        error_info = self.error_handler.create_error_info(
            error=error,
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="Test message",
            technical_message="Test technical"
        )
        
        self.error_handler._store_error(error_info)
        
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_counts["input_validation"] == 1
    
    def test_get_error_statistics(self) -> Optional[Dict[str, Any]]:
        """Test error statistics generation."""
        # Add some test errors
        for i in range(3):
            error = ValidationError(f"Test error {i}")
            error_info = self.error_handler.create_error_info(
                error=error,
                category=ErrorCategory.INPUT_VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                user_message=f"Test message {i}",
                technical_message=f"Test technical {i}"
            )
            self.error_handler._store_error(error_info)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts"]["input_validation"] == 3
        assert len(stats["recent_errors"]) == 3


class TestGradioInputValidator:
    """Test GradioInputValidator class."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.error_handler = GradioErrorHandler()
        self.validator = GradioInputValidator(self.error_handler)
    
    def test_validator_initialization(self) -> Any:
        """Test validator initialization."""
        assert self.validator.error_handler == self.error_handler
        assert len(self.validator.validation_rules) > 0
        assert "input_text" in self.validator.validation_rules
        assert "quality" in self.validator.validation_rules
        assert "duration" in self.validator.validation_rules
    
    def test_validate_text_input_success(self) -> bool:
        """Test successful text validation."""
        # Valid text
        self.validator.validate_text_input("This is a valid text input", "input_text")
        
        # Empty text with required=False
        rule = InputValidationRule(field_name="optional_text", required=False)
        self.validator.validation_rules["optional_text"] = rule
        self.validator.validate_text_input("", "optional_text")
    
    def test_validate_text_input_failure(self) -> bool:
        """Test text validation failures."""
        # Empty required text
        with pytest.raises(ValidationError, match="input_text is required"):
            self.validator.validate_text_input("", "input_text")
        
        # Text too short
        with pytest.raises(ValidationError, match="at least 10 characters"):
            self.validator.validate_text_input("Short", "input_text")
        
        # Text too long
        with pytest.raises(ValidationError, match="at most 2000 characters"):
            long_text = "a" * 2001
            self.validator.validate_text_input(long_text, "input_text")
    
    def test_validate_numeric_input_success(self) -> bool:
        """Test successful numeric validation."""
        # Valid duration
        self.validator.validate_numeric_input(30, "duration")
        
        # Valid quality
        self.validator.validate_numeric_input("medium", "quality")
    
    def test_validate_numeric_input_failure(self) -> bool:
        """Test numeric validation failures."""
        # Duration too low
        with pytest.raises(ValidationError, match="at least 5"):
            self.validator.validate_numeric_input(3, "duration")
        
        # Duration too high
        with pytest.raises(ValidationError, match="at most 300"):
            self.validator.validate_numeric_input(400, "duration")
        
        # Invalid quality
        with pytest.raises(ValidationError, match="one of"):
            self.validator.validate_numeric_input("invalid", "quality")
    
    def test_validate_file_input_success(self) -> bool:
        """Test successful file validation."""
        # Create temporary file for testing
        temp_file = Path("temp_test_file.txt")
        temp_file.write_text("test content")
        
        try:
            # Valid file
            self.validator.validate_file_input(str(temp_file), "uploaded_file")
            
            # None file (optional)
            self.validator.validate_file_input(None, "uploaded_file")
            
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_validate_file_input_failure(self) -> bool:
        """Test file validation failures."""
        # Non-existent file
        with pytest.raises(ValidationError, match="File not found"):
            self.validator.validate_file_input("nonexistent_file.txt", "uploaded_file")
        
        # Invalid file type
        temp_file = Path("temp_test_file.invalid")
        temp_file.write_text("test content")
        
        try:
            with pytest.raises(ValidationError, match="File type not supported"):
                self.validator.validate_file_input(str(temp_file), "uploaded_file")
        finally:
            temp_file.unlink(missing_ok=True)
    
    async def test_validate_video_request_success(self) -> bool:
        """Test successful video request validation."""
        request = self.validator.validate_video_request(
            input_text="This is a valid video prompt for testing purposes",
            quality="medium",
            duration=30,
            output_format="mp4"
        )
        
        assert isinstance(request, VideoRequest)
        assert request.input_text == "This is a valid video prompt for testing purposes"
        assert request.quality == "medium"
        assert request.duration == 30
        assert request.output_format == "mp4"
    
    async def test_validate_video_request_failure(self) -> bool:
        """Test video request validation failures."""
        # Invalid input text
        with pytest.raises(ValidationError, match="at least 10 characters"):
            self.validator.validate_video_request(
                input_text="Short",
                quality="medium",
                duration=30,
                output_format="mp4"
            )
        
        # Invalid quality
        with pytest.raises(ValidationError, match="one of"):
            self.validator.validate_video_request(
                input_text="Valid prompt text",
                quality="invalid",
                duration=30,
                output_format="mp4"
            )
        
        # Invalid duration
        with pytest.raises(ValidationError, match="at least 5"):
            self.validator.validate_video_request(
                input_text="Valid prompt text",
                quality="medium",
                duration=3,
                output_format="mp4"
            )


class TestDecorators:
    """Test decorator functions."""
    
    def test_gradio_error_handler_decorator(self) -> Any:
        """Test gradio_error_handler decorator."""
        @gradio_error_handler(show_technical=False, log_errors=True)
        def test_function():
            
    """test_function function."""
raise ValidationError("Test error")
        
        result = test_function()
        
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["success"] is False
        assert "title" in result
        assert "description" in result
        assert "details" in result
    
    def test_gradio_error_handler_success(self) -> Any:
        """Test gradio_error_handler with successful execution."""
        @gradio_error_handler(show_technical=False, log_errors=True)
        def test_function():
            
    """test_function function."""
return {"success": True, "data": "test"}
        
        result = test_function()
        
        assert result == {"success": True, "data": "test"}
    
    def test_gradio_input_validator_decorator(self) -> Any:
        """Test gradio_input_validator decorator."""
        @gradio_input_validator()
        def test_function(input_text: str, quality: str):
            
    """test_function function."""
return {"input_text": input_text, "quality": quality}
        
        # Valid inputs
        result = test_function("Valid input text", "medium")
        assert result == {"input_text": "Valid input text", "quality": "medium"}
        
        # Invalid input
        result = test_function("Short", "medium")
        assert isinstance(result, dict)
        assert result["error"] is True


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_handle_gradio_error(self) -> Any:
        """Test handle_gradio_error function."""
        error = ValidationError("Test validation error")
        result = handle_gradio_error(error, show_technical=False)
        
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["success"] is False
        assert "title" in result
        assert "description" in result
        assert "details" in result
    
    def test_validate_gradio_inputs_success(self) -> bool:
        """Test validate_gradio_inputs with valid inputs."""
        result = validate_gradio_inputs(
            input_text="Valid input text for video generation",
            quality="medium",
            duration=30,
            output_format="mp4"
        )
        
        assert isinstance(result, VideoRequest)
        assert result.input_text == "Valid input text for video generation"
        assert result.quality == "medium"
        assert result.duration == 30
        assert result.output_format == "mp4"
    
    def test_validate_gradio_inputs_failure(self) -> bool:
        """Test validate_gradio_inputs with invalid inputs."""
        result = validate_gradio_inputs(
            input_text="Short",
            quality="medium",
            duration=30,
            output_format="mp4"
        )
        
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["success"] is False


class TestGradioComponents:
    """Test Gradio component functions."""
    
    @patch('gradio.HTML')
    def test_create_gradio_error_components(self, mock_html) -> Any:
        """Test create_gradio_error_components function."""
        mock_html.side_effect = [
            Mock(spec=gr.HTML),
            Mock(spec=gr.HTML),
            Mock(spec=gr.HTML)
        ]
        
        error_title, error_description, error_details = create_gradio_error_components()
        
        assert mock_html.call_count == 3
        assert error_title is not None
        assert error_description is not None
        assert error_details is not None
    
    def test_update_error_display_error(self) -> Any:
        """Test update_error_display with error result."""
        result = {
            "error": True,
            "title": "Test Error",
            "description": "Test Description",
            "details": "Test Details"
        }
        
        mock_title = Mock(spec=gr.HTML)
        mock_description = Mock(spec=gr.HTML)
        mock_details = Mock(spec=gr.HTML)
        
        title, description, details, success_visible = update_error_display(
            result, mock_title, mock_description, mock_details
        )
        
        assert success_visible is False
    
    def test_update_error_display_success(self) -> Any:
        """Test update_error_display with success result."""
        result = {
            "error": False,
            "success": True,
            "data": "test"
        }
        
        mock_title = Mock(spec=gr.HTML)
        mock_description = Mock(spec=gr.HTML)
        mock_details = Mock(spec=gr.HTML)
        
        title, description, details, success_visible = update_error_display(
            result, mock_title, mock_description, mock_details
        )
        
        assert success_visible is True


# Integration tests
class TestIntegration:
    """Integration tests for the complete error handling system."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.error_handler = GradioErrorHandler()
        self.validator = GradioInputValidator(self.error_handler)
    
    def test_complete_error_flow(self) -> Any:
        """Test complete error handling flow."""
        # Create an error
        error = ValidationError("Input validation failed")
        
        # Categorize and format error
        category, severity = self.error_handler.categorize_error(error)
        user_message = self.error_handler.get_user_friendly_message(error, category)
        suggestions = self.error_handler.get_suggestions(category)
        
        # Create error info
        error_info = self.error_handler.create_error_info(
            error=error,
            category=category,
            severity=severity,
            user_message=user_message,
            technical_message=str(error),
            suggestions=suggestions
        )
        
        # Store error
        self.error_handler._store_error(error_info)
        
        # Format for Gradio
        title, description, details = self.error_handler.format_error_for_gradio(error)
        
        # Verify results
        assert category == ErrorCategory.INPUT_VALIDATION
        assert severity == ErrorSeverity.MEDIUM
        assert len(suggestions) > 0
        assert "⚠️" in title
        assert "Invalid Input" in title
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_counts["input_validation"] == 1
    
    def test_complete_validation_flow(self) -> Any:
        """Test complete validation flow."""
        # Test valid inputs
        try:
            self.validator.validate_text_input(
                "This is a valid input text for testing purposes", "input_text"
            )
            self.validator.validate_numeric_input(30, "duration")
            self.validator.validate_numeric_input("medium", "quality")
            
            request = self.validator.validate_video_request(
                input_text="Valid video generation prompt",
                quality="high",
                duration=60,
                output_format="mp4"
            )
            
            assert isinstance(request, VideoRequest)
            assert request.input_text == "Valid video generation prompt"
            assert request.quality == "high"
            assert request.duration == 60
            
        except ValidationError:
            pytest.fail("Validation should not fail for valid inputs")
        
        # Test invalid inputs
        with pytest.raises(ValidationError):
            self.validator.validate_text_input("Short", "input_text")
        
        with pytest.raises(ValidationError):
            self.validator.validate_numeric_input(400, "duration")
        
        with pytest.raises(ValidationError):
            self.validator.validate_video_request(
                input_text="Short",
                quality="invalid",
                duration=3,
                output_format="invalid"
            )


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 