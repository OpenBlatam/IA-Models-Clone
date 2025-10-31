from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any
from .conftest import (
from cybersecurity.middleware import (
        import psutil
        import os
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Tests for Middleware
Comprehensive testing with pytest and pytest-asyncio, including network layer mocking.
"""


# Import test fixtures
    mock_socket, mock_ssl_context, mock_aiohttp_session, mock_httpx_client,
    mock_network_responses, temp_file, temp_dir
)

# Import modules to test
    MetricsData, LogEntry, MetricsCollector, CentralizedLogger,
    ExceptionHandler, MiddlewareManager, logging_middleware,
    metrics_middleware, exception_handling_middleware, apply_middleware,
    get_middleware_manager, get_metrics_summary, get_logs,
    get_exception_summary, clear_metrics
)

class TestMetricsData:
    """Test MetricsData class."""
    
    def test_metrics_data_creation(self) -> Any:
        """Test MetricsData creation."""
        metric = MetricsData(
            operation="test_operation",
            duration=1.5,
            success=True,
            error_code=None,
            metadata={"key": "value"},
            timestamp=time.time()
        )
        
        assert metric.operation == "test_operation"
        assert metric.duration == 1.5
        assert metric.success is True
        assert metric.error_code is None
        assert metric.metadata == {"key": "value"}
        assert metric.timestamp is not None
    
    def test_metrics_data_defaults(self) -> Any:
        """Test MetricsData with default values."""
        metric = MetricsData(operation="test", duration=1.0, success=True)
        
        assert metric.operation == "test"
        assert metric.duration == 1.0
        assert metric.success is True
        assert metric.error_code is None
        assert metric.metadata is None
        assert metric.timestamp is None

class TestLogEntry:
    """Test LogEntry class."""
    
    def test_log_entry_creation(self) -> Any:
        """Test LogEntry creation."""
        entry = LogEntry(
            level="INFO",
            message="Test log message",
            operation="test_operation",
            user="test_user",
            target="test_target",
            metadata={"key": "value"},
            timestamp=time.time(),
            traceback="Test traceback"
        )
        
        assert entry.level == "INFO"
        assert entry.message == "Test log message"
        assert entry.operation == "test_operation"
        assert entry.user == "test_user"
        assert entry.target == "test_target"
        assert entry.metadata == {"key": "value"}
        assert entry.timestamp is not None
        assert entry.traceback == "Test traceback"
    
    def test_log_entry_defaults(self) -> Any:
        """Test LogEntry with default values."""
        entry = LogEntry(level="ERROR", message="Test error")
        
        assert entry.level == "ERROR"
        assert entry.message == "Test error"
        assert entry.operation is None
        assert entry.user is None
        assert entry.target is None
        assert entry.metadata is None
        assert entry.timestamp is None
        assert entry.traceback is None

class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_metrics_collector_creation(self) -> Any:
        """Test MetricsCollector creation."""
        collector = MetricsCollector(max_history=100)
        
        assert collector.metrics is not None
        assert collector.counters is not None
        assert collector.timers is not None
        assert collector.errors is not None
    
    def test_record_metric(self) -> Any:
        """Test recording metrics."""
        collector = MetricsCollector()
        
        metric = MetricsData(
            operation="test_operation",
            duration=1.0,
            success=True
        )
        
        collector.record_metric(metric)
        
        assert collector.counters["test_operation"] == 1
        assert len(collector.timers["test_operation"]) == 1
        assert collector.errors["test_operation"] == 0
    
    def test_record_metric_failure(self) -> Any:
        """Test recording failed metrics."""
        collector = MetricsCollector()
        
        metric = MetricsData(
            operation="test_operation",
            duration=1.0,
            success=False,
            error_code="timeout"
        )
        
        collector.record_metric(metric)
        
        assert collector.counters["test_operation"] == 1
        assert len(collector.timers["test_operation"]) == 0
        assert collector.errors["test_operation"] == 1
        assert collector.errors["test_operation_timeout"] == 1
    
    def test_get_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Test getting metrics summary."""
        collector = MetricsCollector()
        
        # Record some metrics
        for i in range(5):
            metric = MetricsData(
                operation="test_operation",
                duration=float(i + 1),
                success=True
            )
            collector.record_metric(metric)
        
        # Record one failure
        failure_metric = MetricsData(
            operation="test_operation",
            duration=0.5,
            success=False,
            error_code="timeout"
        )
        collector.record_metric(failure_metric)
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_operations"] == 6
        assert "test_operation" in summary["operations"]
        assert summary["operations"]["test_operation"]["count"] == 6
        assert summary["operations"]["test_operation"]["error_count"] == 1
        assert summary["operations"]["test_operation"]["success_rate"] == 5/6
        assert summary["operations"]["test_operation"]["avg_duration"] == 3.0
        assert summary["operations"]["test_operation"]["min_duration"] == 1.0
        assert summary["operations"]["test_operation"]["max_duration"] == 5.0
    
    def test_clear_metrics(self) -> Any:
        """Test clearing metrics."""
        collector = MetricsCollector()
        
        # Record some metrics
        metric = MetricsData(operation="test", duration=1.0, success=True)
        collector.record_metric(metric)
        
        # Clear metrics
        collector.clear_metrics()
        
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.timers) == 0
        assert len(collector.errors) == 0

class TestCentralizedLogger:
    """Test CentralizedLogger class."""
    
    def test_centralized_logger_creation(self) -> Any:
        """Test CentralizedLogger creation."""
        logger = CentralizedLogger()
        
        assert logger.logs is not None
        assert logger.max_entries == 10000
    
    def test_log_entry(self) -> Any:
        """Test logging entries."""
        logger = CentralizedLogger()
        
        logger.log(
            level="INFO",
            message="Test message",
            operation="test_operation",
            user="test_user",
            target="test_target"
        )
        
        assert len(logger.logs) == 1
        entry = logger.logs[0]
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.operation == "test_operation"
        assert entry.user == "test_user"
        assert entry.target == "test_target"
    
    def test_log_with_exception(self) -> Any:
        """Test logging with exception."""
        logger = CentralizedLogger()
        
        try:
            raise ValueError("Test exception")
        except Exception as e:
            logger.log(
                level="ERROR",
                message="Exception occurred",
                operation="test_operation",
                exception=e
            )
        
        assert len(logger.logs) == 1
        entry = logger.logs[0]
        assert entry.level == "ERROR"
        assert entry.message == "Exception occurred"
        assert entry.traceback is not None
        assert "ValueError" in entry.traceback
    
    def test_get_logs_filtered(self) -> Optional[Dict[str, Any]]:
        """Test getting filtered logs."""
        logger = CentralizedLogger()
        
        # Add some logs
        logger.log(level="INFO", message="Info message", operation="op1")
        logger.log(level="ERROR", message="Error message", operation="op2")
        logger.log(level="INFO", message="Another info", operation="op1")
        
        # Filter by level
        info_logs = logger.get_logs(level="INFO")
        assert len(info_logs) == 2
        
        # Filter by operation
        op1_logs = logger.get_logs(operation="op1")
        assert len(op1_logs) == 2
        
        # Filter by both
        op1_info_logs = logger.get_logs(level="INFO", operation="op1")
        assert len(op1_info_logs) == 2
    
    def test_get_logs_limit(self) -> Optional[Dict[str, Any]]:
        """Test getting logs with limit."""
        logger = CentralizedLogger()
        
        # Add many logs
        for i in range(10):
            logger.log(level="INFO", message=f"Message {i}")
        
        # Get limited logs
        limited_logs = logger.get_logs(limit=5)
        assert len(limited_logs) == 5

class TestExceptionHandler:
    """Test ExceptionHandler class."""
    
    def test_exception_handler_creation(self) -> Any:
        """Test ExceptionHandler creation."""
        logger = CentralizedLogger()
        metrics = MetricsCollector()
        handler = ExceptionHandler(logger, metrics)
        
        assert handler.logger == logger
        assert handler.metrics == metrics
        assert handler.exceptions is not None
    
    def test_handle_exception(self) -> Any:
        """Test handling exceptions."""
        logger = CentralizedLogger()
        metrics = MetricsCollector()
        handler = ExceptionHandler(logger, metrics)
        
        exception = ValueError("Test exception")
        
        handler.handle_exception(
            exception=exception,
            operation="test_operation",
            user="test_user",
            target="test_target",
            metadata={"key": "value"}
        )
        
        # Check that exception was logged
        assert len(logger.logs) == 1
        log_entry = logger.logs[0]
        assert log_entry.level == "ERROR"
        assert log_entry.operation == "test_operation"
        assert log_entry.user == "test_user"
        assert log_entry.target == "test_target"
        assert log_entry.metadata == {"key": "value"}
        assert log_entry.traceback is not None
        
        # Check that metrics were recorded
        assert handler.exceptions["ValueError"] == 1
    
    def test_get_exception_summary(self) -> Optional[Dict[str, Any]]:
        """Test getting exception summary."""
        logger = CentralizedLogger()
        metrics = MetricsCollector()
        handler = ExceptionHandler(logger, metrics)
        
        # Handle some exceptions
        handler.handle_exception(ValueError("Test 1"), "op1")
        handler.handle_exception(ValueError("Test 2"), "op1")
        handler.handle_exception(TypeError("Test 3"), "op2")
        
        summary = handler.get_exception_summary()
        
        assert summary["total_exceptions"] == 3
        assert summary["exception_types"]["ValueError"] == 2
        assert summary["exception_types"]["TypeError"] == 1
        assert summary["operations"]["op1"] == 2
        assert summary["operations"]["op2"] == 1

class TestMiddlewareManager:
    """Test MiddlewareManager class."""
    
    def test_middleware_manager_creation(self) -> Any:
        """Test MiddlewareManager creation."""
        manager = MiddlewareManager()
        
        assert manager.logger is not None
        assert manager.metrics is not None
        assert manager.exception_handler is not None
        assert manager.middleware_functions is not None
    
    def test_add_middleware(self) -> Any:
        """Test adding middleware functions."""
        manager = MiddlewareManager()
        
        def test_middleware(func) -> Any:
            def wrapper(*args, **kwargs) -> Any:
                return func(*args, **kwargs)
            return wrapper
        
        manager.add_middleware(test_middleware)
        
        assert len(manager.middleware_functions) == 1
    
    def test_apply_middleware(self) -> Any:
        """Test applying middleware to functions."""
        manager = MiddlewareManager()
        
        def test_middleware(func) -> Any:
            def wrapper(*args, **kwargs) -> Any:
                return func(*args, **kwargs)
            return wrapper
        
        manager.add_middleware(test_middleware)
        
        def test_function():
            
    """test_function function."""
return "test"
        
        wrapped_function = manager.apply_middleware(test_function)
        
        assert wrapped_function() == "test"

class TestMiddlewareDecorators:
    """Test middleware decorators."""
    
    @pytest.mark.asyncio
    async def test_logging_middleware(self) -> Any:
        """Test logging middleware decorator."""
        @logging_middleware("test_operation")
        async def test_function():
            
    """test_function function."""
await asyncio.sleep(0.01)
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        
        # Check that log was created
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="test_operation")
        assert len(logs) == 1
    
    @pytest.mark.asyncio
    async def test_metrics_middleware(self) -> Any:
        """Test metrics middleware decorator."""
        @metrics_middleware("test_operation")
        async def test_function():
            
    """test_function function."""
await asyncio.sleep(0.01)
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        
        # Check that metrics were recorded
        metrics = get_metrics_summary()
        assert "test_operation" in metrics["operations"]
        assert metrics["operations"]["test_operation"]["count"] == 1
        assert metrics["operations"]["test_operation"]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_exception_handling_middleware(self) -> Any:
        """Test exception handling middleware decorator."""
        @exception_handling_middleware("test_operation")
        async def test_function():
            
    """test_function function."""
raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            await test_function()
        
        # Check that exception was handled
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="test_operation")
        assert len(logs) == 1
        assert logs[0].level == "ERROR"
    
    @pytest.mark.asyncio
    async def test_apply_middleware_function(self) -> Any:
        """Test apply_middleware function."""
        async def test_function():
            
    """test_function function."""
await asyncio.sleep(0.01)
            return "success"
        
        wrapped_function = apply_middleware(test_function, "test_operation")
        
        result = await wrapped_function()
        
        assert result == "success"
        
        # Check that all middleware was applied
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="test_operation")
        metrics = get_metrics_summary()
        
        assert len(logs) == 1
        assert "test_operation" in metrics["operations"]

class TestNetworkIntegration:
    """Test middleware with network operations."""
    
    @pytest.mark.asyncio
    async def test_network_operation_with_middleware(self) -> Any:
        """Test network operations with middleware."""
        @apply_middleware(operation_name="network_test")
        async def network_operation():
            
    """network_operation function."""
# Simulate network operation
            await asyncio.sleep(0.01)
            return {"status": "success", "data": "test"}
        
        result = await network_operation()
        
        assert result["status"] == "success"
        assert result["data"] == "test"
        
        # Check middleware applied
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="network_test")
        metrics = get_metrics_summary()
        
        assert len(logs) == 1
        assert "network_test" in metrics["operations"]
    
    @pytest.mark.asyncio
    async def test_network_error_with_middleware(self) -> Any:
        """Test network errors with middleware."""
        @apply_middleware(operation_name="network_error_test")
        async def network_operation_with_error():
            
    """network_operation_with_error function."""
# Simulate network error
            await asyncio.sleep(0.01)
            raise ConnectionError("Network timeout")
        
        with pytest.raises(ConnectionError):
            await network_operation_with_error()
        
        # Check error handling
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="network_error_test")
        metrics = get_metrics_summary()
        
        assert len(logs) == 1
        assert logs[0].level == "ERROR"
        assert "network_error_test" in metrics["operations"]
        assert metrics["operations"]["network_error_test"]["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_network_operations(self) -> Any:
        """Test concurrent network operations with middleware."""
        @apply_middleware(operation_name="concurrent_network")
        async def network_operation(i) -> Any:
            await asyncio.sleep(0.01)
            return f"result_{i}"
        
        # Run concurrent operations
        tasks = [network_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert results == ["result_0", "result_1", "result_2", "result_3", "result_4"]
        
        # Check middleware applied to all operations
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="concurrent_network")
        metrics = get_metrics_summary()
        
        assert len(logs) == 5
        assert "concurrent_network" in metrics["operations"]
        assert metrics["operations"]["concurrent_network"]["count"] == 5

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_middleware_with_long_operation(self) -> Any:
        """Test middleware with long-running operation."""
        @apply_middleware(operation_name="long_operation")
        async def long_operation():
            
    """long_operation function."""
await asyncio.sleep(0.1)  # Simulate long operation
            return "completed"
        
        result = await long_operation()
        
        assert result == "completed"
        
        # Check metrics for long operation
        metrics = get_metrics_summary()
        assert "long_operation" in metrics["operations"]
        assert metrics["operations"]["long_operation"]["avg_duration"] > 0.1
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_middleware_with_fast_operation(self) -> Any:
        """Test middleware with very fast operation."""
        @apply_middleware(operation_name="fast_operation")
        async def fast_operation():
            
    """fast_operation function."""
return "fast"
        
        result = await fast_operation()
        
        assert result == "fast"
        
        # Check metrics for fast operation
        metrics = get_metrics_summary()
        assert "fast_operation" in metrics["operations"]
        assert metrics["operations"]["fast_operation"]["avg_duration"] < 0.01
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_middleware_with_exception_types(self) -> Any:
        """Test middleware with different exception types."""
        @apply_middleware(operation_name="exception_test")
        async def operation_with_exception(exception_type) -> Any:
            raise exception_type("Test exception")
        
        exception_types = [ValueError, TypeError, RuntimeError, ConnectionError]
        
        for exc_type in exception_types:
            with pytest.raises(exc_type):
                await operation_with_exception(exc_type)
        
        # Check that all exceptions were handled
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="exception_test")
        metrics = get_metrics_summary()
        
        assert len(logs) == len(exception_types)
        assert "exception_test" in metrics["operations"]
        assert metrics["operations"]["exception_test"]["error_count"] == len(exception_types)
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_middleware_with_nested_exceptions(self) -> Any:
        """Test middleware with nested exceptions."""
        @apply_middleware(operation_name="nested_exception")
        async def nested_operation():
            
    """nested_operation function."""
try:
                raise ValueError("Inner exception")
            except ValueError:
                raise RuntimeError("Outer exception")
        
        with pytest.raises(RuntimeError):
            await nested_operation()
        
        # Check exception handling
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="nested_exception")
        
        assert len(logs) == 1
        assert logs[0].level == "ERROR"
        assert "RuntimeError" in logs[0].traceback

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_middleware_performance_overhead(self) -> Any:
        """Test middleware performance overhead."""
        # Test without middleware
        async def operation_without_middleware():
            
    """operation_without_middleware function."""
await asyncio.sleep(0.01)
            return "result"
        
        start_time = time.time()
        result = await operation_without_middleware()
        time_without_middleware = time.time() - start_time
        
        # Test with middleware
        @apply_middleware(operation_name="performance_test")
        async def operation_with_middleware():
            
    """operation_with_middleware function."""
await asyncio.sleep(0.01)
            return "result"
        
        start_time = time.time()
        result = await operation_with_middleware()
        time_with_middleware = time.time() - start_time
        
        assert result == "result"
        # Middleware overhead should be minimal
        assert time_with_middleware < time_without_middleware + 0.01
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_middleware_performance(self) -> Any:
        """Test concurrent middleware performance."""
        @apply_middleware(operation_name="concurrent_performance")
        async def concurrent_operation(i) -> Any:
            await asyncio.sleep(0.01)
            return f"result_{i}"
        
        start_time = time.time()
        
        # Run many concurrent operations
        tasks = [concurrent_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        assert len(results) == 50
        assert duration < 1.0  # Should complete quickly
        
        # Check metrics
        metrics = get_metrics_summary()
        assert "concurrent_performance" in metrics["operations"]
        assert metrics["operations"]["concurrent_performance"]["count"] == 50
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self) -> Any:
        """Test memory efficiency of middleware."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many operations with middleware
        @apply_middleware(operation_name="memory_test")
        async def memory_operation(i) -> Any:
            await asyncio.sleep(0.001)
            return f"result_{i}"
        
        tasks = [memory_operation(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert len(results) == 100
        assert memory_increase < 50 * 1024 * 1024  # Should not increase by more than 50MB

class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_middleware_workflow(self) -> Any:
        """Test complete middleware workflow."""
        @apply_middleware(operation_name="full_workflow")
        async def full_workflow_operation():
            
    """full_workflow_operation function."""
# Simulate complex operation
            await asyncio.sleep(0.01)
            
            # Simulate some processing
            data = {"processed": True, "timestamp": time.time()}
            
            # Simulate potential error
            if data["timestamp"] % 2 == 0:
                raise ValueError("Simulated error")
            
            return data
        
        results = []
        errors = []
        
        # Run multiple operations
        for i in range(10):
            try:
                result = await full_workflow_operation()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Check results
        assert len(results) + len(errors) == 10
        
        # Check middleware applied
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="full_workflow")
        metrics = get_metrics_summary()
        
        assert len(logs) == 10
        assert "full_workflow" in metrics["operations"]
        assert metrics["operations"]["full_workflow"]["count"] == 10
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_middleware_with_external_services(self) -> Any:
        """Test middleware with external service calls."""
        @apply_middleware(operation_name="external_service")
        async def external_service_call():
            
    """external_service_call function."""
# Simulate external service call
            await asyncio.sleep(0.02)
            
            # Simulate service response
            return {
                "status": "success",
                "data": {"service": "test", "response_time": time.time()}
            }
        
        # Run multiple service calls
        tasks = [external_service_call() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result["status"] == "success"
            assert "data" in result
        
        # Check middleware metrics
        metrics = get_metrics_summary()
        assert "external_service" in metrics["operations"]
        assert metrics["operations"]["external_service"]["count"] == 5
        assert metrics["operations"]["external_service"]["avg_duration"] > 0.02
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_middleware_error_recovery(self) -> Any:
        """Test middleware error recovery."""
        @apply_middleware(operation_name="error_recovery")
        async def operation_with_retry():
            
    """operation_with_retry function."""
# Simulate operation that might fail
            await asyncio.sleep(0.01)
            
            # Simulate random failure
            if time.time() % 3 == 0:
                raise ConnectionError("Temporary failure")
            
            return "success"
        
        results = []
        errors = []
        
        # Run operations with potential failures
        for i in range(10):
            try:
                result = await operation_with_retry()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Check that some succeeded and some failed
        assert len(results) > 0
        assert len(errors) > 0
        assert len(results) + len(errors) == 10
        
        # Check middleware handled all cases
        manager = get_middleware_manager()
        logs = manager.logger.get_logs(operation="error_recovery")
        metrics = get_metrics_summary()
        
        assert len(logs) == 10
        assert "error_recovery" in metrics["operations"]
        assert metrics["operations"]["error_recovery"]["count"] == 10
        assert metrics["operations"]["error_recovery"]["error_count"] > 0 