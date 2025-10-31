"""
Tests for the Centralized Logging Management System
==================================================

Test coverage for:
- Logger initialization and configuration
- Multiple log handlers and formatters
- Performance and specialized logging functions
- Log rotation and file management
"""

import pytest
import tempfile
import os
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import the logging system
from core.logger_manager import (
    StructuredFormatter, ColoredFormatter, LoggerManager,
    get_logger, log_performance, log_api_request, log_security_event,
    log_database_operation, log_cache_operation, log_error_with_context
)


class TestStructuredFormatter:
    """Test structured JSON formatter"""
    
    def test_structured_formatter_basic(self):
        """Test basic structured formatting"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert "timestamp" in log_data
    
    def test_structured_formatter_with_exception(self):
        """Test structured formatting with exception"""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Test error message",
                args=(),
                exc_info=sys.exc_info()
            )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "ERROR"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"


class TestColoredFormatter:
    """Test colored console formatter"""
    
    def test_colored_formatter_basic(self):
        """Test basic colored formatting"""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should contain color codes and basic structure
        assert "INFO" in formatted
        assert "test_logger" in formatted
        assert "Test message" in formatted
        assert "|" in formatted  # Separator


class TestLoggerManager:
    """Test logger manager"""
    
    @pytest.fixture
    def temp_logs_dir(self):
        """Create a temporary logs directory"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            yield Path(temp_dir)
            
            # Clean up any remaining handlers to avoid file permission issues
            import logging
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if hasattr(handler, 'close'):
                    try:
                        handler.close()
                    except:
                        pass
                try:
                    root_logger.removeHandler(handler)
                except:
                    pass
    
    @patch('core.logger_manager.get_config')
    def test_logger_manager_initialization(self, mock_get_config, temp_logs_dir):
        """Test logger manager initialization"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024  # 1MB
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        assert isinstance(logger_manager.logs_dir, Path)
        assert len(logger_manager.handlers) > 0
        assert len(logger_manager.loggers) > 0
    
    @patch('core.logger_manager.get_config')
    def test_get_logger(self, mock_get_config, temp_logs_dir):
        """Test getting logger"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        logger = logger_manager.get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    @patch('core.logger_manager.get_config')
    def test_log_performance(self, mock_get_config, temp_logs_dir):
        """Test performance logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test performance logging
        logger_manager.log_performance("test_operation", 0.125, table="users", rows=100)
        
        # Check that performance logger exists
        assert "performance" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_log_api_request(self, mock_get_config, temp_logs_dir):
        """Test API request logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test API request logging
        logger_manager.log_api_request("GET", "/api/users", 200, 0.045, user_id="123")
        
        # Check that API logger exists
        assert "api" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_log_security_event(self, mock_get_config, temp_logs_dir):
        """Test security event logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test security event logging
        logger_manager.log_security_event("login_failed", "Invalid credentials", ip_address="192.168.1.1")
        
        # Check that security logger exists
        assert "security" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_log_database_operation(self, mock_get_config, temp_logs_dir):
        """Test database operation logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test database operation logging
        logger_manager.log_database_operation("SELECT", "users", 0.125, rows_affected=100)
        
        # Check that database logger exists
        assert "database" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_log_cache_operation(self, mock_get_config, temp_logs_dir):
        """Test cache operation logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test cache operation logging
        logger_manager.log_cache_operation("GET", "user:123", True, 0.001)
        
        # Check that cache logger exists
        assert "cache" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_log_error_with_context(self, mock_get_config, temp_logs_dir):
        """Test error logging with context"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        # Test error logging with context
        error = ValueError("Test error")
        context = {"user_id": "123", "operation": "test"}
        logger_manager.log_error_with_context(error, context, logger_name="error")
        
        # Check that error logger exists
        assert "error" in logger_manager.loggers
    
    @patch('core.logger_manager.get_config')
    def test_get_log_stats(self, mock_get_config, temp_logs_dir):
        """Test getting log statistics"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.system.log_file = str(temp_logs_dir / "test.log")
        mock_config.system.max_log_size = 1024 * 1024
        mock_config.system.backup_logs = 3
        mock_config.monitoring.log_level = "INFO"
        mock_config.monitoring.enable_metrics = True
        mock_config.system.debug = False
        mock_get_config.return_value = mock_config
        
        logger_manager = LoggerManager()
        
        stats = logger_manager.get_log_stats()
        
        assert "total_loggers" in stats
        assert "total_handlers" in stats
        assert "log_level" in stats
        assert "log_file" in stats
        assert "max_log_size" in stats
        assert "backup_logs" in stats
        assert "environment" in stats
        assert "debug_mode" in stats


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('core.logger_manager.logger_manager')
    def test_get_logger(self, mock_logger_manager):
        """Test get_logger convenience function"""
        mock_logger = MagicMock()
        mock_logger_manager.get_logger.return_value = mock_logger
        
        logger = get_logger("test_logger")
        
        assert logger == mock_logger
        mock_logger_manager.get_logger.assert_called_once_with("test_logger")
    
    @patch('core.logger_manager.logger_manager')
    def test_log_performance(self, mock_logger_manager):
        """Test log_performance convenience function"""
        log_performance("test_operation", 0.125, table="users", rows=100)
        
        mock_logger_manager.log_performance.assert_called_once_with(
            "test_operation", 0.125, table="users", rows=100
        )
    
    @patch('core.logger_manager.logger_manager')
    def test_log_api_request(self, mock_logger_manager):
        """Test log_api_request convenience function"""
        log_api_request("GET", "/api/users", 200, 0.045, "123")

        mock_logger_manager.log_api_request.assert_called_once_with(
            "GET", "/api/users", 200, 0.045, "123"
        )
    
    @patch('core.logger_manager.logger_manager')
    def test_log_security_event(self, mock_logger_manager):
        """Test log_security_event convenience function"""
        log_security_event("login_failed", "Invalid credentials", ip_address="192.168.1.1")
        
        mock_logger_manager.log_security_event.assert_called_once_with(
            "login_failed", "Invalid credentials", None, "192.168.1.1"
        )
    
    @patch('core.logger_manager.logger_manager')
    def test_log_database_operation(self, mock_logger_manager):
        """Test log_database_operation convenience function"""
        log_database_operation("SELECT", "users", 0.125, rows_affected=100)
        
        mock_logger_manager.log_database_operation.assert_called_once_with(
            "SELECT", "users", 0.125, 100
        )
    
    @patch('core.logger_manager.logger_manager')
    def test_log_cache_operation(self, mock_logger_manager):
        """Test log_cache_operation convenience function"""
        log_cache_operation("GET", "user:123", True, 0.001)
        
        mock_logger_manager.log_cache_operation.assert_called_once_with(
            "GET", "user:123", True, 0.001
        )
    
    @patch('core.logger_manager.logger_manager')
    def test_log_error_with_context(self, mock_logger_manager):
        """Test log_error_with_context convenience function"""
        error = ValueError("Test error")
        context = {"user_id": "123", "operation": "test"}
        log_error_with_context(error, context, logger_name="error")
        
        mock_logger_manager.log_error_with_context.assert_called_once_with(
            error, context, "error"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
