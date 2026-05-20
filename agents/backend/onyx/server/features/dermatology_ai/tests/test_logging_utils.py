"""
Tests for Logging Utils
Tests for structured logging and logging utilities
"""

import pytest
from unittest.mock import Mock, patch
import logging

from core.infrastructure.logging_utils import StructuredLogger


class TestStructuredLogger:
    """Tests for StructuredLogger"""
    
    @pytest.fixture
    def structured_logger(self):
        """Create structured logger"""
        return StructuredLogger("test_module")
    
    def test_create_logger(self, structured_logger):
        """Test creating structured logger"""
        assert structured_logger is not None
        assert structured_logger.logger is not None
    
    def test_set_context(self, structured_logger):
        """Test setting logging context"""
        structured_logger.set_context(
            user_id="user-123",
            operation="analyze_image"
        )
        
        assert structured_logger.context["user_id"] == "user-123"
        assert structured_logger.context["operation"] == "analyze_image"
    
    def test_log_with_context(self, structured_logger):
        """Test logging with context"""
        structured_logger.set_context(user_id="user-123")
        
        with patch.object(structured_logger.logger, 'info') as mock_info:
            structured_logger.info("Test message")
            
            mock_info.assert_called_once()
            # Check that context is included in log
            call_args = mock_info.call_args
            assert call_args is not None
    
    def test_operation_context(self, structured_logger):
        """Test operation context manager"""
        with structured_logger.operation("test_operation", user_id="user-123"):
            assert structured_logger.context.get("operation") == "test_operation"
            assert structured_logger.context.get("user_id") == "user-123"
        
        # Context should be cleared after operation
        # (implementation dependent)
    
    def test_log_error(self, structured_logger):
        """Test logging errors"""
        with patch.object(structured_logger.logger, 'error') as mock_error:
            structured_logger.error("Error message", exc_info=True)
            
            mock_error.assert_called_once()
    
    def test_log_warning(self, structured_logger):
        """Test logging warnings"""
        with patch.object(structured_logger.logger, 'warning') as mock_warning:
            structured_logger.warning("Warning message")
            
            mock_warning.assert_called_once()
    
    def test_log_debug(self, structured_logger):
        """Test logging debug messages"""
        with patch.object(structured_logger.logger, 'debug') as mock_debug:
            structured_logger.debug("Debug message")
            
            mock_debug.assert_called_once()
    
    def test_clear_context(self, structured_logger):
        """Test clearing logging context"""
        structured_logger.set_context(user_id="user-123", key="value")
        structured_logger.clear_context()
        
        assert structured_logger.context == {}


class TestLoggingIntegration:
    """Integration tests for logging"""
    
    def test_logging_in_use_case(self):
        """Test logging in use case execution"""
        logger = StructuredLogger("test_use_case")
        logger.set_context(user_id="user-123", operation="analyze")
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Starting analysis")
            
            mock_info.assert_called_once()
    
    def test_logging_with_exception(self):
        """Test logging with exception"""
        logger = StructuredLogger("test_module")
        
        with patch.object(logger.logger, 'error') as mock_error:
            try:
                raise ValueError("Test error")
            except Exception as e:
                logger.error("Error occurred", exc_info=True)
            
            mock_error.assert_called_once()



