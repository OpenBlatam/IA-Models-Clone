"""
Tests for service lifecycle management.
Modular test structure for better organization.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from core.base_service import ServiceStatus
from core.dependency_manager import ServicePriority, ServiceLifecycle, ServiceInfo


# ============================================================================
# MODULE: Service Lifecycle Management
# ============================================================================

class TestServiceLifecycle:
    """Test ServiceLifecycle class - Core lifecycle management"""
    
    def test_service_lifecycle_initialization(self):
        """Test ServiceLifecycle initialization"""
        lifecycle = ServiceLifecycle("test-service", "test", ServicePriority.NORMAL)
        assert lifecycle.name == "test-service"
        assert lifecycle.service_type == "test"
        assert lifecycle.priority == ServicePriority.NORMAL
        assert lifecycle.status == ServiceStatus.UNKNOWN
        assert lifecycle.start_time is None
        assert lifecycle.stop_time is None
        assert lifecycle.error_count == 0
        assert lifecycle.last_error is None
        assert lifecycle.dependencies == []
        assert lifecycle.metadata == {}
    
    def test_add_remove_dependency(self):
        """Test adding and removing dependencies"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        lifecycle.add_dependency("dep1")
        assert "dep1" in lifecycle.dependencies
        
        lifecycle.add_dependency("dep2")
        assert "dep2" in lifecycle.dependencies
        
        lifecycle.remove_dependency("dep1")
        assert "dep1" not in lifecycle.dependencies
        assert "dep2" in lifecycle.dependencies
    
    def test_dependency_duplicates(self):
        """Test that duplicate dependencies are not added"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        lifecycle.add_dependency("dep1")
        lifecycle.add_dependency("dep1")  # Duplicate
        
        assert lifecycle.dependencies.count("dep1") == 1
        assert len(lifecycle.dependencies) == 1
    
    def test_dependency_removal_nonexistent(self):
        """Test removing non-existent dependency"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Should not raise error
        lifecycle.remove_dependency("non_existent")
        assert lifecycle.dependencies == []
    
    def test_lifecycle_callbacks(self):
        """Test lifecycle callback registration"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        start_callback = Mock()
        stop_callback = Mock()
        error_callback = Mock()
        
        lifecycle.on_start(start_callback)
        lifecycle.on_stop(stop_callback)
        lifecycle.on_error(error_callback)
        
        assert lifecycle._on_start == start_callback
        assert lifecycle._on_stop == stop_callback
        assert lifecycle._on_error == error_callback
    
    def test_callback_overwrite(self):
        """Test that callbacks can be overwritten"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        callback1 = Mock()
        callback2 = Mock()
        
        lifecycle.on_start(callback1)
        assert lifecycle._on_start == callback1
        
        lifecycle.on_start(callback2)
        assert lifecycle._on_start == callback2
        assert lifecycle._on_start != callback1
    
    @pytest.mark.asyncio
    async def test_service_start_success(self):
        """Test successful service start"""
        lifecycle = ServiceLifecycle("test", "test_type")
        start_callback = AsyncMock()
        lifecycle.on_start(start_callback)
        
        await lifecycle.start()
        
        assert lifecycle.status == ServiceStatus.RUNNING
        assert lifecycle.start_time is not None
        start_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_start_no_callback(self):
        """Test service start without callback"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Should not raise error if no callback is set
        await lifecycle.start()
        
        assert lifecycle.status == ServiceStatus.RUNNING
        assert lifecycle.start_time is not None
    
    @pytest.mark.asyncio
    async def test_service_start_failure(self):
        """Test service start failure"""
        lifecycle = ServiceLifecycle("test", "test_type")
        error_callback = AsyncMock()
        lifecycle.on_error(error_callback)
        
        # Mock start callback to raise exception
        lifecycle._on_start = AsyncMock(side_effect=Exception("Start failed"))
        
        with pytest.raises(Exception):
            await lifecycle.start()
        
        assert lifecycle.status == ServiceStatus.ERROR
        assert lifecycle.error_count == 1
        assert lifecycle.last_error == "Start failed"
        error_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_start_failure_no_error_callback(self):
        """Test service start failure without error callback"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Mock start callback to raise exception
        lifecycle._on_start = AsyncMock(side_effect=Exception("Start failed"))
        
        with pytest.raises(Exception):
            await lifecycle.start()
        
        assert lifecycle.status == ServiceStatus.ERROR
        assert lifecycle.error_count == 1
        assert lifecycle.last_error == "Start failed"
    
    @pytest.mark.asyncio
    async def test_service_stop_success(self):
        """Test successful service stop"""
        lifecycle = ServiceLifecycle("test", "test_type")
        lifecycle.status = ServiceStatus.RUNNING
        lifecycle.start_time = 123.0
        
        stop_callback = AsyncMock()
        lifecycle.on_stop(stop_callback)
        
        await lifecycle.stop()
        
        assert lifecycle.status == ServiceStatus.STOPPED
        assert lifecycle.stop_time is not None
        stop_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_stop_no_callback(self):
        """Test service stop without callback"""
        lifecycle = ServiceLifecycle("test", "test_type")
        lifecycle.status = ServiceStatus.RUNNING
        lifecycle.start_time = 123.0
        
        # Should not raise error if no callback is set
        await lifecycle.stop()
        
        assert lifecycle.status == ServiceStatus.STOPPED
        assert lifecycle.stop_time is not None
    
    @pytest.mark.asyncio
    async def test_service_stop_from_unknown_state(self):
        """Test service stop from unknown state"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Should handle stop from unknown state gracefully
        await lifecycle.stop()
        
        assert lifecycle.status == ServiceStatus.STOPPED
        assert lifecycle.stop_time is not None
    
    def test_to_info_conversion(self):
        """Test conversion to ServiceInfo"""
        lifecycle = ServiceLifecycle("test", "test_type", ServicePriority.HIGH)
        lifecycle.status = ServiceStatus.RUNNING
        lifecycle.start_time = 123.0
        lifecycle.add_dependency("dep1")
        
        info = lifecycle.to_info()
        
        assert isinstance(info, ServiceInfo)
        assert info.name == "test"
        assert info.service_type == "test_type"
        assert info.priority == ServicePriority.HIGH
        assert info.status == ServiceStatus.RUNNING
        assert info.start_time == 123.0
        assert info.dependencies == ["dep1"]


# ============================================================================
# MODULE: Lifecycle State Transitions
# ============================================================================

class TestLifecycleStateTransitions:
    """Test lifecycle state transitions and validation"""
    
    def test_valid_state_transitions(self):
        """Test valid state transitions"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # UNKNOWN -> STARTING -> RUNNING -> STOPPING -> STOPPED
        assert lifecycle.status == ServiceStatus.UNKNOWN
        
        # Start the service
        lifecycle.status = ServiceStatus.STARTING
        assert lifecycle.status == ServiceStatus.STARTING
        
        lifecycle.status = ServiceStatus.RUNNING
        assert lifecycle.status == ServiceStatus.RUNNING
        
        lifecycle.status = ServiceStatus.STOPPING
        assert lifecycle.status == ServiceStatus.STOPPING
        
        lifecycle.status = ServiceStatus.STOPPED
        assert lifecycle.status == ServiceStatus.STOPPED
    
    def test_error_state_transition(self):
        """Test error state transition"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Can transition to ERROR from any state
        lifecycle.status = ServiceStatus.ERROR
        assert lifecycle.status == ServiceStatus.ERROR
        
        # Can transition back to normal states
        lifecycle.status = ServiceStatus.STOPPED
        assert lifecycle.status == ServiceStatus.STOPPED
    
    def test_lifecycle_metadata_operations(self):
        """Test lifecycle metadata operations"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Test setting metadata
        lifecycle.metadata["key1"] = "value1"
        lifecycle.metadata["key2"] = "value2"
        
        assert lifecycle.metadata["key1"] == "value1"
        assert lifecycle.metadata["key2"] == "value2"
        
        # Test updating metadata
        lifecycle.metadata["key1"] = "updated_value"
        assert lifecycle.metadata["key1"] == "updated_value"
        
        # Test removing metadata
        del lifecycle.metadata["key1"]
        assert "key1" not in lifecycle.metadata


# ============================================================================
# MODULE: Lifecycle Error Handling
# ============================================================================

class TestLifecycleErrorHandling:
    """Test lifecycle error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_start_callback_exception_handling(self):
        """Test handling of start callback exceptions"""
        lifecycle = ServiceLifecycle("test", "test_type")
        error_callback = AsyncMock()
        lifecycle.on_error(error_callback)
        
        # Create a callback that raises different types of exceptions
        async def failing_callback():
            raise RuntimeError("Runtime error occurred")
        
        lifecycle._on_start = failing_callback
        
        with pytest.raises(RuntimeError):
            await lifecycle.start()
        
        assert lifecycle.status == ServiceStatus.ERROR
        assert lifecycle.error_count == 1
        assert "Runtime error occurred" in lifecycle.last_error
        error_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_callback_exception_handling(self):
        """Test handling of stop callback exceptions"""
        lifecycle = ServiceLifecycle("test", "test_type")
        lifecycle.status = ServiceStatus.RUNNING
        lifecycle.start_time = 123.0
        
        error_callback = AsyncMock()
        lifecycle.on_error(error_callback)
        
        # Create a callback that raises an exception
        async def failing_callback():
            raise ValueError("Stop failed")
        
        lifecycle._on_stop = failing_callback
        
        with pytest.raises(ValueError):
            await lifecycle.stop()
        
        assert lifecycle.status == ServiceStatus.ERROR
        assert lifecycle.error_count == 1
        assert "Stop failed" in lifecycle.last_error
        error_callback.assert_called_once()
    
    def test_error_count_increment(self):
        """Test that error count increments correctly"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        assert lifecycle.error_count == 0
        
        # Simulate multiple errors
        lifecycle.error_count += 1
        assert lifecycle.error_count == 1
        
        lifecycle.error_count += 1
        assert lifecycle.error_count == 2
        
        # Reset error count
        lifecycle.error_count = 0
        assert lifecycle.error_count == 0
    
    def test_last_error_tracking(self):
        """Test that last error is tracked correctly"""
        lifecycle = ServiceLifecycle("test", "test_type")
        
        assert lifecycle.last_error is None
        
        # Set error messages
        lifecycle.last_error = "First error"
        assert lifecycle.last_error == "First error"
        
        lifecycle.last_error = "Second error"
        assert lifecycle.last_error == "Second error"
        
        # Clear error
        lifecycle.last_error = None
        assert lifecycle.last_error is None


# ============================================================================
# MODULE: Lifecycle Performance and Scalability
# ============================================================================

class TestLifecyclePerformance:
    """Test lifecycle performance characteristics"""
    
    def test_dependency_operations_performance(self):
        """Test performance of dependency operations"""
        import time
        
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Test adding many dependencies
        start_time = time.time()
        for i in range(1000):
            lifecycle.add_dependency(f"dep_{i}")
        add_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert add_time < 0.1
        assert len(lifecycle.dependencies) == 1000
        
        # Test removing dependencies
        start_time = time.time()
        for i in range(1000):
            lifecycle.remove_dependency(f"dep_{i}")
        remove_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert remove_time < 0.1
        assert len(lifecycle.dependencies) == 0
    
    def test_metadata_operations_performance(self):
        """Test performance of metadata operations"""
        import time
        
        lifecycle = ServiceLifecycle("test", "test_type")
        
        # Test setting many metadata items
        start_time = time.time()
        for i in range(1000):
            lifecycle.metadata[f"key_{i}"] = f"value_{i}"
        set_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert set_time < 0.1
        assert len(lifecycle.metadata) == 1000
        
        # Test accessing metadata
        start_time = time.time()
        for i in range(1000):
            _ = lifecycle.metadata[f"key_{i}"]
        access_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert access_time < 0.1
