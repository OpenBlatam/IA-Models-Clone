"""
Test suite for enhanced error logging and user-friendly error messages.

Tests the logging configuration, error message templates, and error handling
with proper logging and user-friendly messages.
"""

import pytest
import sys
import os
import json
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import (
    setup_logging,
    EnhancedLogger,
    ErrorMessages,
    log_error_with_context,
    log_validation_error,
    log_processing_error,
    log_resource_error,
    log_security_error,
    RequestTracker,
    create_logging_middleware
)

from error_handling import (
    ErrorHandler,
    ValidationError,
    ProcessingError,
    ExternalServiceError,
    ResourceError,
    CriticalSystemError,
    SecurityError,
    ConfigurationError,
    ErrorCode
)

# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_ERROR_CONTEXT = {
    "user_id": "user123",
    "operation": "video_processing",
    "video_id": "abc123",
    "processing_time": 15.5
}

SAMPLE_VALIDATION_ERROR = ValidationError(
    "Invalid YouTube URL format",
    "youtube_url",
    "invalid_url"
)

SAMPLE_PROCESSING_ERROR = ProcessingError(
    "Video processing failed",
    "video_encoding",
    {"video_id": "abc123", "error_type": "encoding_error"}
)

SAMPLE_RESOURCE_ERROR = ResourceError(
    "Insufficient GPU memory",
    "gpu_memory",
    available="2GB",
    required="4GB"
)

# =============================================================================
# LOGGING CONFIGURATION TESTS
# =============================================================================

class TestLoggingConfiguration:
    """Test logging configuration setup."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        # Should not raise any exceptions
        setup_logging(log_level="INFO", enable_console=True, enable_json=False)
        
        # Verify logger is configured
        logger = logging.getLogger()
        assert logger.level == logging.INFO
    
    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file output."""
        log_file = tmp_path / "test.log"
        
        setup_logging(
            log_level="DEBUG",
            log_file=str(log_file),
            enable_console=False,
            enable_json=True
        )
        
        # Verify log file was created
        assert log_file.exists()
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level."""
        # Should handle invalid level gracefully
        setup_logging(log_level="INVALID_LEVEL")
        
        # Should default to INFO
        logger = logging.getLogger()
        assert logger.level == logging.INFO

# =============================================================================
# ENHANCED LOGGER TESTS
# =============================================================================

class TestEnhancedLogger:
    """Test enhanced logger functionality."""
    
    def test_enhanced_logger_initialization(self):
        """Test enhanced logger initialization."""
        logger = EnhancedLogger("test_logger")
        
        assert logger.logger is not None
        assert logger.request_id is not None
        assert isinstance(logger.request_id, str)
    
    def test_set_request_id(self):
        """Test setting request ID."""
        logger = EnhancedLogger("test_logger")
        original_id = logger.request_id
        
        new_id = "test-request-123"
        logger.set_request_id(new_id)
        
        assert logger.request_id == new_id
        assert logger.request_id != original_id
    
    def test_format_user_message(self):
        """Test user message formatting."""
        logger = EnhancedLogger("test_logger")
        
        # Test with no details
        message = logger._format_user_message("Test message")
        assert message == "Test message"
        
        # Test with details
        details = {"field": "youtube_url", "value": "invalid_url"}
        message = logger._format_user_message("Validation failed", details)
        assert "youtube_url" in message
        assert "invalid_url" in message
    
    def test_format_technical_message(self):
        """Test technical message formatting."""
        logger = EnhancedLogger("test_logger")
        
        # Test with no error or details
        message = logger._format_technical_message("Test message")
        assert message == "Test message"
        
        # Test with error
        error = ValueError("Test error")
        message = logger._format_technical_message("Processing failed", error)
        assert "ValueError" in message
        assert "Test error" in message
        
        # Test with details
        details = {"operation": "video_processing", "video_id": "abc123"}
        message = logger._format_technical_message("Processing failed", error, details)
        assert "video_processing" in message
        assert "abc123" in message
    
    @patch('logging_config.structlog.get_logger')
    def test_info_logging(self, mock_get_logger):
        """Test info level logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = EnhancedLogger("test_logger")
        logger.info("Test info message", extra_data="test")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Test info message"
        assert "request_id" in call_args[1]
        assert "extra_data" in call_args[1]
    
    @patch('logging_config.structlog.get_logger')
    def test_error_logging(self, mock_get_logger):
        """Test error level logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = EnhancedLogger("test_logger")
        error = ValueError("Test error")
        logger.error("Test error message", error=error, details={"field": "test"})
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[0][0] == "Test error message"
        assert "error_type" in call_args[1]
        assert "error_message" in call_args[1]
        assert "stack_trace" in call_args[1]
        assert "request_id" in call_args[1]

# =============================================================================
# ERROR MESSAGES TESTS
# =============================================================================

class TestErrorMessages:
    """Test error message templates."""
    
    def test_get_user_message_validation_error(self):
        """Test getting user message for validation error."""
        message = ErrorMessages.get_user_message("youtube_url_invalid")
        assert "YouTube URL format is not valid" in message
        assert "check the URL" in message
    
    def test_get_user_message_with_parameters(self):
        """Test getting user message with parameters."""
        message = ErrorMessages.get_user_message("clip_length_invalid", min_length=10, max_length=600)
        assert "10" in message
        assert "600" in message
    
    def test_get_user_message_unknown_error(self):
        """Test getting user message for unknown error."""
        message = ErrorMessages.get_user_message("unknown_error_code")
        assert "unexpected error occurred" in message
    
    def test_get_suggestion_validation_error(self):
        """Test getting suggestion for validation error."""
        suggestion = ErrorMessages.get_suggestion("youtube_url_invalid")
        assert suggestion is not None
        assert "youtube.com" in suggestion
    
    def test_get_suggestion_unknown_error(self):
        """Test getting suggestion for unknown error."""
        suggestion = ErrorMessages.get_suggestion("unknown_error_code")
        assert suggestion is None
    
    def test_get_suggestion_resource_error(self):
        """Test getting suggestion for resource error."""
        suggestion = ErrorMessages.get_suggestion("insufficient_memory")
        assert suggestion is not None
        assert "shorter video" in suggestion
    
    def test_validation_error_messages(self):
        """Test all validation error messages."""
        validation_errors = [
            "youtube_url_required",
            "youtube_url_invalid",
            "language_required",
            "language_invalid",
            "clip_length_invalid",
            "caption_required",
            "caption_empty"
        ]
        
        for error_code in validation_errors:
            message = ErrorMessages.get_user_message(error_code)
            assert message is not None
            assert len(message) > 0
            assert "Please" in message or "cannot" in message
    
    def test_processing_error_messages(self):
        """Test all processing error messages."""
        processing_errors = [
            "video_processing_failed",
            "langchain_processing_failed",
            "viral_analysis_failed",
            "image_generation_failed"
        ]
        
        for error_code in processing_errors:
            message = ErrorMessages.get_user_message(error_code)
            assert message is not None
            assert len(message) > 0
            assert "try again" in message
    
    def test_resource_error_messages(self):
        """Test all resource error messages."""
        resource_errors = [
            "insufficient_memory",
            "gpu_not_available",
            "gpu_memory_full",
            "disk_space_full",
            "rate_limit_exceeded"
        ]
        
        for error_code in resource_errors:
            message = ErrorMessages.get_user_message(error_code)
            assert message is not None
            assert len(message) > 0
            assert "try again" in message or "later" in message

# =============================================================================
# ERROR LOGGING UTILITIES TESTS
# =============================================================================

class TestErrorLoggingUtilities:
    """Test error logging utility functions."""
    
    @patch('logging_config.EnhancedLogger')
    def test_log_error_with_context(self, mock_enhanced_logger):
        """Test logging error with context."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        error = ValueError("Test error")
        context = {"user_id": "user123", "operation": "test"}
        
        log_error_with_context(mock_logger, error, context, "User friendly message", "test_error")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[0][0] == "User friendly message"
        assert call_args[1]["error"] == error
        assert "user_id" in call_args[1]["details"]
    
    @patch('logging_config.EnhancedLogger')
    def test_log_validation_error(self, mock_enhanced_logger):
        """Test logging validation error."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        log_validation_error(mock_logger, "youtube_url", "invalid_url", "youtube_url_invalid", "Technical message")
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "field" in call_args[1]["details"]
        assert "value" in call_args[1]["details"]
        assert "error_code" in call_args[1]["details"]
    
    @patch('logging_config.EnhancedLogger')
    def test_log_processing_error(self, mock_enhanced_logger):
        """Test logging processing error."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        error = ProcessingError("Processing failed", "video_processing")
        context = {"video_id": "abc123"}
        
        log_processing_error(mock_logger, "video_processing", error, context)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[1]["error"] == error
        assert "operation" in call_args[1]["details"]
        assert "video_id" in call_args[1]["details"]
    
    @patch('logging_config.EnhancedLogger')
    def test_log_resource_error(self, mock_enhanced_logger):
        """Test logging resource error."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        log_resource_error(mock_logger, "gpu_memory", "2GB", "4GB")
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "resource" in call_args[1]["details"]
        assert "available" in call_args[1]["details"]
        assert "required" in call_args[1]["details"]
    
    @patch('logging_config.EnhancedLogger')
    def test_log_security_error(self, mock_enhanced_logger):
        """Test logging security error."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        log_security_error(mock_logger, "malicious_input", {"ip": "192.168.1.1"})
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "threat_type" in call_args[1]["details"]
        assert "ip" in call_args[1]["details"]

# =============================================================================
# REQUEST TRACKER TESTS
# =============================================================================

class TestRequestTracker:
    """Test request tracking functionality."""
    
    def test_request_tracker_initialization(self):
        """Test request tracker initialization."""
        tracker = RequestTracker()
        
        assert tracker.request_id is None
        assert tracker.user_id is None
        assert tracker.session_id is None
        assert tracker.start_time is None
    
    def test_start_request(self):
        """Test starting request tracking."""
        tracker = RequestTracker()
        
        request_id = "test-request-123"
        user_id = "user123"
        session_id = "session456"
        
        tracker.start_request(request_id, user_id, session_id)
        
        assert tracker.request_id == request_id
        assert tracker.user_id == user_id
        assert tracker.session_id == session_id
        assert tracker.start_time is not None
        assert isinstance(tracker.start_time, datetime)
    
    def test_get_context(self):
        """Test getting request context."""
        tracker = RequestTracker()
        
        # Test empty context
        context = tracker.get_context()
        assert context["request_id"] is None
        assert context["user_id"] is None
        assert context["session_id"] is None
        
        # Test with data
        tracker.start_request("test-123", "user123", "session456")
        context = tracker.get_context()
        
        assert context["request_id"] == "test-123"
        assert context["user_id"] == "user123"
        assert context["session_id"] == "session456"
        assert "duration" in context
    
    def test_clear(self):
        """Test clearing request context."""
        tracker = RequestTracker()
        
        tracker.start_request("test-123", "user123", "session456")
        assert tracker.request_id is not None
        
        tracker.clear()
        
        assert tracker.request_id is None
        assert tracker.user_id is None
        assert tracker.session_id is None
        assert tracker.start_time is None

# =============================================================================
# ERROR HANDLER TESTS
# =============================================================================

class TestErrorHandler:
    """Test enhanced error handler functionality."""
    
    @patch('error_handling.EnhancedLogger')
    def test_error_handler_initialization(self, mock_enhanced_logger):
        """Test error handler initialization."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        handler = ErrorHandler()
        
        assert handler.logger is not None
        assert handler.critical_error_count == 0
        assert "critical" in handler.error_thresholds
    
    @patch('error_handling.EnhancedLogger')
    @patch('error_handling.ErrorMessages')
    def test_handle_validation_error(self, mock_error_messages, mock_enhanced_logger):
        """Test handling validation error with user-friendly messages."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        mock_error_messages.get_user_message.return_value = "User friendly message"
        mock_error_messages.get_suggestion.return_value = "Try using a valid URL"
        
        handler = ErrorHandler()
        error = ValidationError("Invalid URL", "youtube_url", "invalid_url")
        
        response = handler.handle_validation_error(error, "request-123")
        
        # Verify logging
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "User friendly message"
        
        # Verify response
        assert response.message == "User friendly message"
        assert response.request_id == "request-123"
        assert "suggestion" in response.details
        assert "help_url" in response.details
    
    @patch('error_handling.EnhancedLogger')
    @patch('error_handling.ErrorMessages')
    def test_handle_processing_error(self, mock_error_messages, mock_enhanced_logger):
        """Test handling processing error with user-friendly messages."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        mock_error_messages.get_user_message.return_value = "Processing failed, please try again"
        mock_error_messages.get_suggestion.return_value = "Wait a moment and retry"
        
        handler = ErrorHandler()
        error = ProcessingError("Video processing failed", "video_encoding")
        
        response = handler.handle_processing_error(error, "request-123")
        
        # Verify logging
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[0][0] == "Processing failed, please try again"
        
        # Verify response
        assert response.message == "Processing failed, please try again"
        assert response.request_id == "request-123"
        assert "retry_after" in response.details
        assert response.details["retry_after"] == 30
    
    @patch('error_handling.EnhancedLogger')
    @patch('error_handling.ErrorMessages')
    def test_handle_critical_system_error(self, mock_error_messages, mock_enhanced_logger):
        """Test handling critical system error with alerting."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        mock_error_messages.get_user_message.return_value = "Critical system error"
        mock_error_messages.get_suggestion.return_value = "Contact support immediately"
        
        handler = ErrorHandler()
        error = CriticalSystemError("System crash", "database")
        
        response = handler.handle_critical_system_error(error, "request-123")
        
        # Verify logging
        mock_logger.critical.assert_called_once()
        call_args = mock_logger.critical.call_args
        assert call_args[0][0] == "Critical system error"
        
        # Verify response
        assert response.message == "Critical system error"
        assert response.details["contact_support"] is True
        assert response.details["critical"] is True
    
    @patch('error_handling.EnhancedLogger')
    @patch('error_handling.ErrorMessages')
    def test_handle_security_error(self, mock_error_messages, mock_enhanced_logger):
        """Test handling security error with threat detection."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        mock_error_messages.get_user_message.return_value = "Security violation detected"
        mock_error_messages.get_suggestion.return_value = "Check your input"
        
        handler = ErrorHandler()
        error = SecurityError("Malicious input", "injection")
        
        response = handler.handle_security_error(error, "request-123")
        
        # Verify logging
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "Security violation detected"
        
        # Verify response
        assert response.message == "Security violation detected"
        assert response.details["security_incident"] is True

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestErrorLoggingIntegration:
    """Test integration of error logging components."""
    
    @patch('logging_config.EnhancedLogger')
    @patch('error_handling.ErrorMessages')
    def test_full_error_handling_flow(self, mock_error_messages, mock_enhanced_logger):
        """Test complete error handling flow with logging."""
        mock_logger = Mock()
        mock_enhanced_logger.return_value = mock_logger
        
        mock_error_messages.get_user_message.return_value = "User friendly error message"
        mock_error_messages.get_suggestion.return_value = "Helpful suggestion"
        
        # Create error handler
        handler = ErrorHandler()
        
        # Create validation error
        error = ValidationError("Invalid YouTube URL", "youtube_url", "invalid_url")
        
        # Handle error
        response = handler.handle_validation_error(error, "request-123")
        
        # Verify complete flow
        assert response.message == "User friendly error message"
        assert response.request_id == "request-123"
        assert "suggestion" in response.details
        assert "help_url" in response.details
        
        # Verify logging
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "User friendly error message"
        assert "error_code" in call_args[1]["details"]
        assert "field" in call_args[1]["details"]
        assert "value" in call_args[1]["details"]
    
    def test_error_message_consistency(self):
        """Test consistency of error messages across different error types."""
        # Test that all error types have user-friendly messages
        error_types = [
            "youtube_url_invalid",
            "video_processing_failed",
            "insufficient_memory",
            "youtube_api_error",
            "unauthorized_access",
            "missing_config",
            "system_crash"
        ]
        
        for error_type in error_types:
            message = ErrorMessages.get_user_message(error_type)
            assert message is not None
            assert len(message) > 0
            assert isinstance(message, str)
    
    def test_suggestion_availability(self):
        """Test that common errors have helpful suggestions."""
        # Test that common errors have suggestions
        common_errors = [
            "youtube_url_invalid",
            "insufficient_memory",
            "rate_limit_exceeded",
            "gpu_not_available"
        ]
        
        for error_type in common_errors:
            suggestion = ErrorMessages.get_suggestion(error_type)
            assert suggestion is not None
            assert len(suggestion) > 0
            assert isinstance(suggestion, str)

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestErrorLoggingPerformance:
    """Test performance of error logging system."""
    
    def test_error_message_lookup_performance(self):
        """Test performance of error message lookups."""
        import time
        
        # Test lookup performance
        start_time = time.perf_counter()
        
        for _ in range(1000):
            ErrorMessages.get_user_message("youtube_url_invalid")
            ErrorMessages.get_suggestion("youtube_url_invalid")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should be very fast (less than 1 second for 1000 lookups)
        assert duration < 1.0
    
    def test_logger_initialization_performance(self):
        """Test performance of logger initialization."""
        import time
        
        start_time = time.perf_counter()
        
        for _ in range(100):
            logger = EnhancedLogger("test_logger")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should be very fast
        assert duration < 1.0

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestErrorLoggingEdgeCases:
    """Test edge cases in error logging system."""
    
    def test_error_message_with_missing_parameters(self):
        """Test error message formatting with missing parameters."""
        # Should handle missing parameters gracefully
        message = ErrorMessages.get_user_message("clip_length_invalid")
        assert message is not None
        assert len(message) > 0
    
    def test_error_message_with_extra_parameters(self):
        """Test error message formatting with extra parameters."""
        # Should handle extra parameters gracefully
        message = ErrorMessages.get_user_message("youtube_url_invalid", extra_param="test")
        assert message is not None
        assert len(message) > 0
    
    def test_logger_with_none_error(self):
        """Test logger with None error."""
        logger = EnhancedLogger("test_logger")
        
        # Should handle None error gracefully
        logger.error("Test message", error=None, details={"test": "value"})
    
    def test_logger_with_empty_details(self):
        """Test logger with empty details."""
        logger = EnhancedLogger("test_logger")
        
        # Should handle empty details gracefully
        logger.error("Test message", details={})
    
    def test_request_tracker_with_none_values(self):
        """Test request tracker with None values."""
        tracker = RequestTracker()
        
        # Should handle None values gracefully
        tracker.start_request("test-123", None, None)
        
        context = tracker.get_context()
        assert context["request_id"] == "test-123"
        assert context["user_id"] is None
        assert context["session_id"] is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 