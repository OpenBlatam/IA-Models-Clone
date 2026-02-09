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
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from ...utils.logger import (
from ...utils.performance import (
from ...utils.security import (
from typing import Any, List, Dict, Optional
import logging
"""
Unit tests for utility modules.
"""


    OnyxLogger, setup_logger, get_logger, log_function_call,
    log_performance, log_error, log_security_event, log_plugin_event,
    create_logger_context, format_log_message, get_log_level,
    set_log_level, create_rotating_logger, create_json_logger
)
    PerformanceMonitor, PerformanceMetrics, CacheManager,
    start_performance_monitoring, stop_performance_monitoring,
    get_performance_metrics, measure_function_performance,
    track_memory_usage, track_cpu_usage, track_gpu_usage,
    create_performance_report, analyze_performance_trends,
    get_system_resources, monitor_request_performance
)
    SecurityManager, SecurityConfig, SecurityEvent,
    validate_access, validate_input, check_rate_limit,
    encrypt_data, decrypt_data, hash_password, verify_password,
    generate_api_key, validate_api_key, sanitize_input,
    log_security_event, create_security_audit_log,
    check_permissions, validate_file_upload, scan_for_malware
)


class TestOnyxLogger:
    """Test Onyx Logger."""
    
    @pytest.mark.unit
    def test_logger_initialization(self, temp_dir) -> Any:
        """Test logger initialization."""
        log_file = temp_dir / "test.log"
        
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file),
            use_onyx_logging=False
        )
        
        assert logger.name == "test_logger"
        assert logger.level == "DEBUG"
        assert logger.log_file == str(log_file)
        assert logger.use_onyx_logging is False
        assert logger.logger is not None
    
    @pytest.mark.unit
    def test_logger_with_onyx(self) -> Any:
        """Test logger with Onyx integration."""
        with patch('onyx.utils.logger.setup_logger') as mock_onyx_logger:
            mock_onyx_logger.return_value = Mock()
            
            logger = OnyxLogger(
                name="test_logger",
                use_onyx_logging=True
            )
            
            assert logger.use_onyx_logging is True
            mock_onyx_logger.assert_called_once()
    
    @pytest.mark.unit
    def test_logging_methods(self, temp_dir) -> Any:
        """Test all logging methods."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        # Test all logging levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        logger.notice("Notice message")
        
        # Check that log file was created and contains messages
        assert log_file.exists()
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
            assert "Critical message" in content
            assert "Notice message" in content
    
    @pytest.mark.unit
    async def test_request_context(self, temp_dir) -> Any:
        """Test request context logging."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        logger.set_request_context("req123", "user456", "session789")
        logger.info("Request message")
        
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "req123" in content
            assert "user456" in content
            assert "session789" in content
    
    @pytest.mark.unit
    async def test_clear_request_context(self, temp_dir) -> Any:
        """Test clearing request context."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        logger.set_request_context("req123", "user456", "session789")
        logger.clear_request_context()
        logger.info("No context message")
        
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            # Should not contain context in the last message
            lines = content.strip().split('\n')
            last_line = lines[-1]
            assert "req123" not in last_line
            assert "user456" not in last_line
            assert "session789" not in last_line
    
    @pytest.mark.unit
    def test_log_level_validation(self) -> Any:
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            logger = OnyxLogger(name="test", level=level)
            assert logger.level == level
        
        # Test invalid level
        with pytest.raises(ValueError, match="Invalid log level"):
            OnyxLogger(name="test", level="INVALID")
    
    @pytest.mark.unit
    def test_log_file_rotation(self, temp_dir) -> Any:
        """Test log file rotation."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file),
            max_size=1024,  # 1KB
            backup_count=2
        )
        
        # Write enough data to trigger rotation
        large_message = "x" * 100
        for _ in range(20):  # 2KB of data
            logger.info(large_message)
        
        # Check that backup files were created
        backup_files = list(temp_dir.glob("test.log.*"))
        assert len(backup_files) > 0
    
    @pytest.mark.unit
    def test_json_logging(self, temp_dir) -> Any:
        """Test JSON logging format."""
        log_file = temp_dir / "test.json"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file),
            format_type="json"
        )
        
        logger.info("Test message", extra={"key": "value"})
        
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            line = f.readline().strip()
            log_entry = json.loads(line)
            
            assert log_entry["level"] == "INFO"
            assert log_entry["message"] == "Test message"
            assert log_entry["key"] == "value"
            assert "timestamp" in log_entry


class TestLoggerUtilities:
    """Test logger utility functions."""
    
    @pytest.mark.unit
    def test_setup_logger(self, temp_dir) -> Any:
        """Test setup_logger function."""
        log_file = temp_dir / "test.log"
        
        logger = setup_logger(
            name="test_logger",
            log_file=str(log_file),
            level="DEBUG"
        )
        
        assert logger is not None
        assert logger.name == "test_logger"
    
    @pytest.mark.unit
    def test_get_logger(self) -> Optional[Dict[str, Any]]:
        """Test get_logger function."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        assert logger1 is logger2
    
    @pytest.mark.unit
    def test_log_function_call(self, temp_dir) -> Any:
        """Test log_function_call decorator."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        @log_function_call(logger)
        def test_function(arg1, arg2) -> Any:
            return arg1 + arg2
        
        result = test_function(1, 2)
        
        assert result == 3
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "test_function" in content
            assert "arg1=1" in content
            assert "arg2=2" in content
    
    @pytest.mark.unit
    def test_log_performance(self, temp_dir) -> Any:
        """Test log_performance function."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        log_performance(logger, "test_operation", 1.5, {"key": "value"})
        
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "test_operation" in content
            assert "1.5" in content
            assert "key" in content
    
    @pytest.mark.unit
    def test_log_error(self, temp_dir) -> Any:
        """Test log_error function."""
        log_file = temp_dir / "test.log"
        logger = OnyxLogger(
            name="test_logger",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        error = ValueError("Test error")
        log_error(logger, error, "test_context", {"extra": "data"})
        
        with open(log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "ValueError" in content
            assert "Test error" in content
            assert "test_context" in content
            assert "extra" in content
    
    @pytest.mark.unit
    def test_create_logger_context(self) -> Any:
        """Test create_logger_context function."""
        context = create_logger_context(
            request_id="req123",
            user_id="user456",
            session_id="session789",
            operation="test_op"
        )
        
        assert context["request_id"] == "req123"
        assert context["user_id"] == "user456"
        assert context["session_id"] == "session789"
        assert context["operation"] == "test_op"
        assert "timestamp" in context


class TestPerformanceMonitor:
    """Test Performance Monitor."""
    
    @pytest.mark.unit
    def test_initialization(self) -> Any:
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(
            enable_monitoring=True,
            metrics_interval=10,
            cache_enabled=True,
            cache_size=100
        )
        
        assert monitor.enable_monitoring is True
        assert monitor.metrics_interval == 10
        assert monitor.cache_enabled is True
        assert monitor.cache_size == 100
        assert monitor.metrics is not None
        assert monitor.cache is not None
    
    @pytest.mark.unit
    def test_start_stop_operation(self) -> Any:
        """Test operation timing."""
        monitor = PerformanceMonitor()
        
        operation_id = monitor.start_operation("test_operation")
        time.sleep(0.1)  # Simulate work
        monitor.end_operation(operation_id)
        
        metrics = monitor.get_operation_metrics(operation_id)
        assert metrics is not None
        assert metrics["operation_name"] == "test_operation"
        assert metrics["duration"] > 0
    
    @pytest.mark.unit
    def test_get_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Test system metrics collection."""
        monitor = PerformanceMonitor()
        
        metrics = monitor.get_system_metrics()
        
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics
        assert "timestamp" in metrics
    
    @pytest.mark.unit
    def test_get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Test performance summary generation."""
        monitor = PerformanceMonitor()
        
        # Add some operations
        op1 = monitor.start_operation("op1")
        time.sleep(0.1)
        monitor.end_operation(op1)
        
        op2 = monitor.start_operation("op2")
        time.sleep(0.1)
        monitor.end_operation(op2)
        
        summary = monitor.get_performance_summary()
        
        assert summary["total_operations"] >= 2
        assert summary["avg_duration"] > 0
        assert "operations" in summary
    
    @pytest.mark.unit
    def test_cache_operations(self) -> Any:
        """Test cache operations."""
        monitor = PerformanceMonitor(cache_enabled=True, cache_size=5)
        
        # Test cache set/get
        monitor.cache_set("key1", "value1", ttl=60)
        value = monitor.cache_get("key1")
        assert value == "value1"
        
        # Test cache size limit
        for i in range(10):
            monitor.cache_set(f"key{i}", f"value{i}")
        
        # Should have evicted oldest entries
        assert len(monitor.cache) <= 5
    
    @pytest.mark.unit
    def test_cache_ttl(self) -> Any:
        """Test cache TTL functionality."""
        monitor = PerformanceMonitor(cache_enabled=True)
        
        monitor.cache_set("key1", "value1", ttl=0.1)  # 100ms TTL
        time.sleep(0.2)  # Wait for expiration
        
        value = monitor.cache_get("key1")
        assert value is None
    
    @pytest.mark.unit
    def test_memory_tracking(self) -> Any:
        """Test memory usage tracking."""
        monitor = PerformanceMonitor()
        
        memory_info = monitor.track_memory_usage()
        
        assert "used_memory" in memory_info
        assert "total_memory" in memory_info
        assert "memory_percentage" in memory_info
        assert memory_info["used_memory"] > 0
        assert memory_info["total_memory"] > 0
    
    @pytest.mark.unit
    def test_cpu_tracking(self) -> Any:
        """Test CPU usage tracking."""
        monitor = PerformanceMonitor()
        
        cpu_info = monitor.track_cpu_usage()
        
        assert "cpu_percentage" in cpu_info
        assert "cpu_count" in cpu_info
        assert cpu_info["cpu_percentage"] >= 0
        assert cpu_info["cpu_count"] > 0
    
    @pytest.mark.unit
    def test_gpu_tracking(self) -> Any:
        """Test GPU usage tracking."""
        monitor = PerformanceMonitor()
        
        gpu_info = monitor.track_gpu_usage()
        
        # GPU might not be available, so just check structure
        assert "gpu_available" in gpu_info
        if gpu_info["gpu_available"]:
            assert "gpu_usage" in gpu_info
            assert "gpu_memory" in gpu_info
    
    @pytest.mark.unit
    def test_create_performance_report(self) -> Any:
        """Test performance report creation."""
        monitor = PerformanceMonitor()
        
        # Add some operations
        op1 = monitor.start_operation("op1")
        time.sleep(0.1)
        monitor.end_operation(op1)
        
        report = monitor.create_performance_report()
        
        assert "summary" in report
        assert "operations" in report
        assert "system_metrics" in report
        assert "recommendations" in report
    
    @pytest.mark.unit
    def test_analyze_performance_trends(self) -> Any:
        """Test performance trend analysis."""
        monitor = PerformanceMonitor()
        
        # Add operations over time
        for i in range(5):
            op = monitor.start_operation(f"op{i}")
            time.sleep(0.1)
            monitor.end_operation(op)
        
        trends = monitor.analyze_performance_trends()
        
        assert "trend_analysis" in trends
        assert "performance_changes" in trends
        assert "recommendations" in trends


class TestPerformanceUtilities:
    """Test performance utility functions."""
    
    @pytest.mark.unit
    def test_start_stop_performance_monitoring(self) -> Any:
        """Test performance monitoring start/stop."""
        monitor = PerformanceMonitor()
        
        start_performance_monitoring(monitor)
        assert monitor.monitoring_active is True
        
        stop_performance_monitoring(monitor)
        assert monitor.monitoring_active is False
    
    @pytest.mark.unit
    def test_get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Test get_performance_metrics function."""
        monitor = PerformanceMonitor()
        
        metrics = get_performance_metrics(monitor)
        
        assert "system_metrics" in metrics
        assert "operation_metrics" in metrics
        assert "cache_metrics" in metrics
    
    @pytest.mark.unit
    def test_measure_function_performance(self) -> Any:
        """Test measure_function_performance decorator."""
        monitor = PerformanceMonitor()
        
        @measure_function_performance(monitor)
        def test_function():
            
    """test_function function."""
time.sleep(0.1)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        summary = monitor.get_performance_summary()
        assert summary["total_operations"] >= 1
    
    @pytest.mark.unit
    def test_track_memory_usage(self) -> Any:
        """Test track_memory_usage function."""
        memory_info = track_memory_usage()
        
        assert "used_memory" in memory_info
        assert "total_memory" in memory_info
        assert "memory_percentage" in memory_info
    
    @pytest.mark.unit
    def test_track_cpu_usage(self) -> Any:
        """Test track_cpu_usage function."""
        cpu_info = track_cpu_usage()
        
        assert "cpu_percentage" in cpu_info
        assert "cpu_count" in cpu_info
    
    @pytest.mark.unit
    def test_track_gpu_usage(self) -> Any:
        """Test track_gpu_usage function."""
        gpu_info = track_gpu_usage()
        
        assert "gpu_available" in gpu_info
        if gpu_info["gpu_available"]:
            assert "gpu_usage" in gpu_info
    
    @pytest.mark.unit
    def test_get_system_resources(self) -> Optional[Dict[str, Any]]:
        """Test get_system_resources function."""
        resources = get_system_resources()
        
        assert "cpu" in resources
        assert "memory" in resources
        assert "disk" in resources
        assert "network" in resources
    
    @pytest.mark.unit
    async def test_monitor_request_performance(self) -> Any:
        """Test monitor_request_performance function."""
        monitor = PerformanceMonitor()
        
        def request_handler():
            
    """request_handler function."""
time.sleep(0.1)
            return {"status": "success"}
        
        result = monitor_request_performance(monitor, request_handler)
        
        assert result["status"] == "success"
        assert "performance_metrics" in result


class TestSecurityManager:
    """Test Security Manager."""
    
    @pytest.mark.unit
    def test_initialization(self) -> Any:
        """Test security manager initialization."""
        config = SecurityConfig(
            enable_encryption=True,
            encryption_key="test-key-32-chars-long-key",
            validate_input=True,
            max_input_length=1000,
            rate_limit_enabled=True,
            rate_limit_requests=10,
            rate_limit_window=60
        )
        
        manager = SecurityManager(config)
        
        assert manager.config == config
        assert manager.rate_limit_cache is not None
        assert manager.security_events is not None
    
    @pytest.mark.unit
    async def test_validate_access(self) -> bool:
        """Test access validation."""
        manager = SecurityManager()
        
        # Test valid access
        result = await manager.validate_access("user123", "video_generation")
        assert result is True
        
        # Test invalid access
        with patch.object(manager, '_check_permissions', return_value=False):
            result = await manager.validate_access("user123", "admin_access")
            assert result is False
    
    @pytest.mark.unit
    def test_validate_input(self) -> bool:
        """Test input validation."""
        manager = SecurityManager()
        
        # Test valid input
        is_valid, cleaned_input = manager.validate_input("Valid input text")
        assert is_valid is True
        assert cleaned_input == "Valid input text"
        
        # Test input too long
        long_input = "x" * 2000
        is_valid, _ = manager.validate_input(long_input)
        assert is_valid is False
    
    @pytest.mark.unit
    def test_check_rate_limit(self) -> Any:
        """Test rate limiting."""
        config = SecurityConfig(
            rate_limit_enabled=True,
            rate_limit_requests=2,
            rate_limit_window=60
        )
        manager = SecurityManager(config)
        
        # First request
        allowed, info = manager.check_rate_limit("user123")
        assert allowed is True
        assert info["remaining"] == 1
        
        # Second request
        allowed, info = manager.check_rate_limit("user123")
        assert allowed is True
        assert info["remaining"] == 0
        
        # Third request (should be blocked)
        allowed, info = manager.check_rate_limit("user123")
        assert allowed is False
        assert info["remaining"] == 0
    
    @pytest.mark.unit
    def test_encrypt_decrypt_data(self) -> Any:
        """Test data encryption and decryption."""
        config = SecurityConfig(
            enable_encryption=True,
            encryption_key="test-key-32-chars-long-key"
        )
        manager = SecurityManager(config)
        
        original_data = "sensitive data"
        encrypted = manager.encrypt_data(original_data)
        decrypted = manager.decrypt_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    @pytest.mark.unit
    def test_hash_password(self) -> Any:
        """Test password hashing."""
        manager = SecurityManager()
        
        password = "test_password"
        hashed = manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > len(password)
    
    @pytest.mark.unit
    def test_verify_password(self) -> Any:
        """Test password verification."""
        manager = SecurityManager()
        
        password = "test_password"
        hashed = manager.hash_password(password)
        
        assert manager.verify_password(password, hashed) is True
        assert manager.verify_password("wrong_password", hashed) is False
    
    @pytest.mark.unit
    async def test_generate_api_key(self) -> Any:
        """Test API key generation."""
        manager = SecurityManager()
        
        api_key = manager.generate_api_key()
        
        assert len(api_key) == 32
        assert api_key.isalnum()
    
    @pytest.mark.unit
    async def test_validate_api_key(self) -> bool:
        """Test API key validation."""
        manager = SecurityManager()
        
        api_key = manager.generate_api_key()
        
        assert manager.validate_api_key(api_key) is True
        assert manager.validate_api_key("invalid_key") is False
    
    @pytest.mark.unit
    def test_sanitize_input(self) -> Any:
        """Test input sanitization."""
        manager = SecurityManager()
        
        # Test script injection
        malicious_input = "<script>alert('xss')</script>"
        sanitized = manager.sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        
        # Test SQL injection
        sql_input = "'; DROP TABLE users; --"
        sanitized = manager.sanitize_input(sql_input)
        
        assert "DROP TABLE" not in sanitized
        assert ";" not in sanitized
    
    @pytest.mark.unit
    def test_log_security_event(self) -> Any:
        """Test security event logging."""
        manager = SecurityManager()
        
        event = SecurityEvent(
            event_type="access_denied",
            user_id="user123",
            ip_address="192.168.1.1",
            details={"reason": "rate_limit_exceeded"}
        )
        
        manager.log_security_event(event)
        
        assert len(manager.security_events) == 1
        assert manager.security_events[0].event_type == "access_denied"
    
    @pytest.mark.unit
    def test_create_security_audit_log(self) -> Any:
        """Test security audit log creation."""
        manager = SecurityManager()
        
        # Add some events
        for i in range(5):
            event = SecurityEvent(
                event_type="access_attempt",
                user_id=f"user{i}",
                ip_address=f"192.168.1.{i}",
                details={"operation": "video_generation"}
            )
            manager.log_security_event(event)
        
        audit_log = manager.create_security_audit_log()
        
        assert "total_events" in audit_log
        assert "events_by_type" in audit_log
        assert "events_by_user" in audit_log
        assert audit_log["total_events"] == 5
    
    @pytest.mark.unit
    async def test_check_permissions(self) -> Any:
        """Test permission checking."""
        manager = SecurityManager()
        
        # Mock permission check
        with patch.object(manager, '_get_user_permissions', return_value=["video_generation"]):
            has_permission = await manager._check_permissions("user123", "video_generation")
            assert has_permission is True
            
            has_permission = await manager._check_permissions("user123", "admin_access")
            assert has_permission is False
    
    @pytest.mark.unit
    async def test_validate_file_upload(self) -> bool:
        """Test file upload validation."""
        manager = SecurityManager()
        
        # Test valid file
        valid_file = Mock()
        valid_file.filename = "test.mp4"
        valid_file.content_type = "video/mp4"
        valid_file.size = 1024 * 1024  # 1MB
        
        is_valid, error = manager.validate_file_upload(valid_file)
        assert is_valid is True
        assert error is None
        
        # Test invalid file type
        invalid_file = Mock()
        invalid_file.filename = "test.exe"
        invalid_file.content_type = "application/x-executable"
        invalid_file.size = 1024
        
        is_valid, error = manager.validate_file_upload(invalid_file)
        assert is_valid is False
        assert "executable" in error.lower()
    
    @pytest.mark.unit
    def test_scan_for_malware(self) -> Any:
        """Test malware scanning."""
        manager = SecurityManager()
        
        # Mock malware scan
        with patch.object(manager, '_perform_malware_scan', return_value=False):
            is_clean = manager.scan_for_malware(b"test_file_content")
            assert is_clean is True
        
        with patch.object(manager, '_perform_malware_scan', return_value=True):
            is_clean = manager.scan_for_malware(b"malicious_content")
            assert is_clean is False


class TestSecurityUtilities:
    """Test security utility functions."""
    
    @pytest.mark.unit
    async def test_validate_access_function(self) -> bool:
        """Test validate_access function."""
        config = SecurityConfig()
        manager = SecurityManager(config)
        
        result = await validate_access(manager, "user123", "video_generation")
        assert result is True
    
    @pytest.mark.unit
    def test_validate_input_function(self) -> bool:
        """Test validate_input function."""
        config = SecurityConfig()
        manager = SecurityManager(config)
        
        is_valid, cleaned = validate_input(manager, "Valid input")
        assert is_valid is True
        assert cleaned == "Valid input"
    
    @pytest.mark.unit
    def test_check_rate_limit_function(self) -> Any:
        """Test check_rate_limit function."""
        config = SecurityConfig(rate_limit_enabled=True, rate_limit_requests=1)
        manager = SecurityManager(config)
        
        allowed, info = check_rate_limit(manager, "user123")
        assert allowed is True
        
        allowed, info = check_rate_limit(manager, "user123")
        assert allowed is False
    
    @pytest.mark.unit
    def test_encrypt_decrypt_functions(self) -> Any:
        """Test encrypt/decrypt functions."""
        config = SecurityConfig(enable_encryption=True, encryption_key="test-key-32-chars-long-key")
        manager = SecurityManager(config)
        
        data = "sensitive data"
        encrypted = encrypt_data(manager, data)
        decrypted = decrypt_data(manager, encrypted)
        
        assert decrypted == data
    
    @pytest.mark.unit
    def test_hash_verify_password_functions(self) -> Any:
        """Test password hash/verify functions."""
        manager = SecurityManager()
        
        password = "test_password"
        hashed = hash_password(manager, password)
        
        assert verify_password(manager, password, hashed) is True
        assert verify_password(manager, "wrong", hashed) is False
    
    @pytest.mark.unit
    async def test_generate_validate_api_key_functions(self) -> bool:
        """Test API key generation/validation functions."""
        manager = SecurityManager()
        
        api_key = generate_api_key(manager)
        assert validate_api_key(manager, api_key) is True
        assert validate_api_key(manager, "invalid") is False
    
    @pytest.mark.unit
    def test_sanitize_input_function(self) -> Any:
        """Test sanitize_input function."""
        manager = SecurityManager()
        
        malicious = "<script>alert('xss')</script>"
        sanitized = sanitize_input(manager, malicious)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
    
    @pytest.mark.unit
    def test_log_security_event_function(self) -> Any:
        """Test log_security_event function."""
        manager = SecurityManager()
        
        event = SecurityEvent(
            event_type="test_event",
            user_id="user123",
            details={"test": "data"}
        )
        
        log_security_event(manager, event)
        assert len(manager.security_events) == 1
    
    @pytest.mark.unit
    def test_create_security_audit_log_function(self) -> Any:
        """Test create_security_audit_log function."""
        manager = SecurityManager()
        
        audit_log = create_security_audit_log(manager)
        
        assert "total_events" in audit_log
        assert "events_by_type" in audit_log
        assert "events_by_user" in audit_log
    
    @pytest.mark.unit
    async def test_check_permissions_function(self) -> Any:
        """Test check_permissions function."""
        manager = SecurityManager()
        
        with patch.object(manager, '_get_user_permissions', return_value=["video_generation"]):
            has_permission = await check_permissions(manager, "user123", "video_generation")
            assert has_permission is True
    
    @pytest.mark.unit
    async def test_validate_file_upload_function(self) -> bool:
        """Test validate_file_upload function."""
        manager = SecurityManager()
        
        file = Mock()
        file.filename = "test.mp4"
        file.content_type = "video/mp4"
        file.size = 1024
        
        is_valid, error = validate_file_upload(manager, file)
        assert is_valid is True
        assert error is None
    
    @pytest.mark.unit
    def test_scan_for_malware_function(self) -> Any:
        """Test scan_for_malware function."""
        manager = SecurityManager()
        
        with patch.object(manager, '_perform_malware_scan', return_value=False):
            is_clean = scan_for_malware(manager, b"test_content")
            assert is_clean is True 