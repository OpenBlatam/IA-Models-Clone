"""
Tests for the centralized dependency management system.
Modular test structure for better organization and maintainability.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, Mock
from typing import Dict, Any, Optional

from core.dependency_manager import (
    ServiceStatus, ServicePriority, ServiceInfo, ServiceLifecycle, DependencyManager,
    start_all_services, stop_all_services, get_dependency_manager,
    register_service, get_service, has_service
)


# ============================================================================
# MODULE 1: Core Enums and Data Structures
# ============================================================================

class TestServiceStatus:
    """Test ServiceStatus enum - Core data structure"""
    
    def test_service_status_values(self):
        """Test ServiceStatus enum values"""
        assert ServiceStatus.UNKNOWN.value == "unknown"
        assert ServiceStatus.STARTING.value == "starting"
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.STOPPING.value == "stopping"
        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.ERROR.value == "error"
    
    def test_service_status_immutability(self):
        """Test that ServiceStatus values are immutable"""
        # Should not be able to modify enum values
        assert ServiceStatus.UNKNOWN.value == "unknown"
        assert ServiceStatus.RUNNING.value == "running"


class TestServicePriority:
    """Test ServicePriority enum - Core data structure"""
    
    def test_service_priority_values(self):
        """Test ServicePriority enum values"""
        assert ServicePriority.CRITICAL.value == 0
        assert ServicePriority.HIGH.value == 1
        assert ServicePriority.NORMAL.value == 2
        assert ServicePriority.LOW.value == 3
    
    def test_service_priority_comparison(self):
        """Test ServicePriority comparison"""
        assert ServicePriority.CRITICAL.value < ServicePriority.HIGH.value
        assert ServicePriority.HIGH.value < ServicePriority.NORMAL.value
        assert ServicePriority.NORMAL.value < ServicePriority.LOW.value
    
    def test_service_priority_ordering(self):
        """Test ServicePriority ordering consistency"""
        priorities = [
            ServicePriority.CRITICAL,
            ServicePriority.HIGH,
            ServicePriority.NORMAL,
            ServicePriority.LOW
        ]
        
        for i in range(len(priorities) - 1):
            assert priorities[i].value < priorities[i + 1].value


class TestServiceInfo:
    """Test ServiceInfo dataclass - Core data structure"""
    
    def test_service_info_creation(self):
        """Test ServiceInfo creation with defaults"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        assert info.name == "test-service"
        assert info.service_type == "test"
        assert info.priority == ServicePriority.NORMAL
        assert info.status == ServiceStatus.STOPPED
        assert info.start_time is None
        assert info.stop_time is None
        assert info.error_count == 0
        assert info.last_error is None
        assert info.dependencies == []
        assert info.metadata == {}
    
    def test_service_info_with_all_fields(self):
        """Test ServiceInfo with all fields populated"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.HIGH,
            status=ServiceStatus.RUNNING,
            start_time=123.0,
            stop_time=None,
            error_count=0,
            last_error=None,
            dependencies=["dep1", "dep2"],
            metadata={"key": "value"}
        )
        assert info.name == "test-service"
        assert info.service_type == "test"
        assert info.priority == ServicePriority.HIGH
        assert info.status == ServiceStatus.RUNNING
        assert info.start_time == 123.0
        assert info.dependencies == ["dep1", "dep2"]
        assert info.metadata == {"key": "value"}
    
    def test_service_info_immutability(self):
        """Test that ServiceInfo fields maintain data integrity"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        # Test that dependencies and metadata are separate instances
        original_deps = info.dependencies
        original_metadata = info.metadata
        
        assert original_deps is not info.dependencies
        assert original_metadata is not info.metadata


# ============================================================================
# MODULE 2: Service Lifecycle Management
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
# MODULE 3: Dependency Manager Core
# ============================================================================

class TestDependencyManager:
    """Test DependencyManager class - Core dependency management"""
    
    @pytest.fixture
    def dependency_manager(self):
        """Create a DependencyManager instance"""
        return DependencyManager()
    
    def test_dependency_manager_initialization(self, dependency_manager):
        """Test DependencyManager initialization"""
        assert len(dependency_manager.services) == 0
        assert len(dependency_manager.service_instances) == 0
        assert len(dependency_manager.service_factories) == 0
        assert dependency_manager.is_running is False
    
    def test_register_service(self, dependency_manager):
        """Test service registration"""
        factory = Mock()
        
        dependency_manager.register_service(
            "test_service",
            "test_type",
            factory,
            ServicePriority.HIGH,
            ["dep1", "dep2"]
        )
        
        assert "test_service" in dependency_manager.services
        assert "test_service" in dependency_manager.service_factories
        assert dependency_manager.service_factories["test_service"] == factory
        
        service = dependency_manager.services["test_service"]
        assert service.name == "test_service"
        assert service.service_type == "test_type"
        assert service.priority == ServicePriority.HIGH
        assert service.dependencies == ["dep1", "dep2"]
    
    def test_register_service_overwrite(self, dependency_manager):
        """Test that registering existing service overwrites it"""
        factory1 = Mock()
        factory2 = Mock()
        
        dependency_manager.register_service("test_service", "test_type", factory1)
        dependency_manager.register_service("test_service", "test_type", factory2)
        
        assert dependency_manager.service_factories["test_service"] == factory2
    
    def test_unregister_service(self, dependency_manager):
        """Test service unregistration"""
        factory = Mock()
        dependency_manager.register_service("test_service", "test_type", factory)
        
        dependency_manager.unregister_service("test_service")
        
        assert "test_service" not in dependency_manager.services
        assert "test_service" not in dependency_manager.service_factories
    
    def test_unregister_nonexistent_service(self, dependency_manager):
        """Test unregistering non-existent service"""
        # Should not raise error
        dependency_manager.unregister_service("non_existent")
    
    def test_get_service_status(self, dependency_manager):
        """Test getting service status"""
        factory = Mock()
        dependency_manager.register_service("test_service", "test_type", factory)
        
        status = dependency_manager.get_service_status("test_service")
        assert status == ServiceStatus.UNKNOWN
        
        # Test non-existent service
        status = dependency_manager.get_service_status("non_existent")
        assert status is None
    
    def test_get_service_info(self, dependency_manager):
        """Test getting service information"""
        factory = Mock()
        dependency_manager.register_service("test_service", "test_type", factory)
        
        info = dependency_manager.get_service_info("test_service")
        assert isinstance(info, ServiceInfo)
        assert info.name == "test_service"
        
        # Test non-existent service
        info = dependency_manager.get_service_info("non_existent")
        assert info is None
    
    def test_get_all_services(self, dependency_manager):
        """Test getting all services"""
        dependency_manager.register_service("service1", "type1", Mock())
        dependency_manager.register_service("service2", "type2", Mock())
        
        services = dependency_manager.get_all_services()
        assert len(services) == 2
        assert any(s.name == "service1" for s in services)
        assert any(s.name == "service2" for s in services)
    
    def test_get_services_by_status(self, dependency_manager):
        """Test getting services by status"""
        factory = Mock()
        dependency_manager.register_service("service1", "type1", factory)
        dependency_manager.register_service("service2", "type2", factory)
        
        # Set different statuses
        dependency_manager.services["service1"].status = ServiceStatus.RUNNING
        dependency_manager.services["service2"].status = ServiceStatus.STOPPED
        
        running_services = dependency_manager.get_services_by_status(ServiceStatus.RUNNING)
        assert len(running_services) == 1
        assert running_services[0].name == "service1"
        
        stopped_services = dependency_manager.get_services_by_status(ServiceStatus.STOPPED)
        assert len(stopped_services) == 1
        assert stopped_services[0].name == "service2"
    
    def test_check_dependencies(self, dependency_manager):
        """Test dependency checking"""
        dependency_manager.register_service("dep_service", "type1", Mock())
        dependency_manager.register_service("main_service", "type2", Mock(), dependencies=["dep_service"])
        
        # Set dependency service to running
        dependency_manager.services["dep_service"].status = ServiceStatus.RUNNING
        
        # Check dependencies
        assert dependency_manager.check_dependencies("main_service") is True
        
        # Set dependency service to stopped
        dependency_manager.services["dep_service"].status = ServiceStatus.STOPPED
        
        # Check dependencies
        assert dependency_manager.check_dependencies("main_service") is False
    
    def test_get_startup_order(self, dependency_manager):
        """Test startup order calculation"""
        dependency_manager.register_service("dep1", "type1", Mock())
        dependency_manager.register_service("dep2", "type2", Mock(), dependencies=["dep1"])
        dependency_manager.register_service("main", "type3", Mock(), dependencies=["dep2"])
        
        startup_order = dependency_manager.get_startup_order()
        
        # Check that dependencies come before dependents
        assert startup_order.index("dep1") < startup_order.index("dep2")
        assert startup_order.index("dep2") < startup_order.index("main")
    
    @pytest.mark.asyncio
    async def test_start_all_services(self, dependency_manager):
        """Test starting all services"""
        factory1 = Mock(return_value="instance1")
        factory2 = Mock(return_value="instance2")
        
        dependency_manager.register_service("service1", "type1", factory1)
        dependency_manager.register_service("service2", "type2", factory2)
        
        await dependency_manager.start_all_services()
        
        assert dependency_manager.is_running is True
        assert dependency_manager.services["service1"].status == ServiceStatus.RUNNING
        assert dependency_manager.services["service2"].status == ServiceStatus.RUNNING
        assert "service1" in dependency_manager.service_instances
        assert "service2" in dependency_manager.service_instances
    
    @pytest.mark.asyncio
    async def test_stop_all_services(self, dependency_manager):
        """Test stopping all services"""
        factory = Mock(return_value="instance")
        dependency_manager.register_service("service1", "type1", factory)
        
        await dependency_manager.start_all_services()
        assert dependency_manager.is_running is True
        
        await dependency_manager.stop_all_services()
        
        assert dependency_manager.is_running is False
        assert dependency_manager.services["service1"].status == ServiceStatus.STOPPED
    
    def test_get_health_summary(self, dependency_manager):
        """Test health summary generation"""
        dependency_manager.register_service("service1", "type1", Mock())
        dependency_manager.register_service("service2", "type2", Mock())
        
        # Set different statuses
        dependency_manager.services["service1"].status = ServiceStatus.RUNNING
        dependency_manager.services["service2"].status = ServiceStatus.ERROR
        
        health = dependency_manager.get_health_summary()
        
        assert health["total_services"] == 2
        assert health["running_services"] == 1
        assert health["error_services"] == 1
        assert health["health_percentage"] == 50.0
        assert "timestamp" in health
        assert len(health["services"]) == 2


# ============================================================================
# MODULE 4: Global Functions and API
# ============================================================================

class TestGlobalFunctions:
    """Test global convenience functions - Public API layer"""
    
    @patch('core.dependency_manager.dependency_manager')
    def test_get_dependency_manager(self, mock_dependency_manager):
        """Test get_dependency_manager function"""
        manager = get_dependency_manager()
        assert manager == mock_dependency_manager
    
    @patch('core.dependency_manager.dependency_manager')
    def test_register_service(self, mock_dependency_manager):
        """Test register_service function"""
        factory = Mock()
        register_service("test", "type", factory, ServicePriority.HIGH, ["dep1"])
        
        mock_dependency_manager.register_service.assert_called_once_with(
            "test", "type", factory, ServicePriority.HIGH, ["dep1"]
        )
    
    @patch('core.dependency_manager.dependency_manager')
    def test_get_service(self, mock_dependency_manager):
        """Test get_service function"""
        mock_dependency_manager.get_service.return_value = "test_instance"
        
        service = get_service("test")
        
        assert service == "test_instance"
        mock_dependency_manager.get_service.assert_called_once_with("test")
    
    @patch('core.dependency_manager.dependency_manager')
    def test_has_service(self, mock_dependency_manager):
        """Test has_service function"""
        mock_dependency_manager.has_service.return_value = True
        
        exists = has_service("test")
        
        assert exists is True
        mock_dependency_manager.has_service.assert_called_once_with("test")
    
    @patch('core.dependency_manager.dependency_manager')
    @pytest.mark.asyncio
    async def test_start_all_services(self, mock_dependency_manager):
        """Test start_all_services function"""
        mock_dependency_manager.start_all_services = AsyncMock()
        
        await start_all_services()
        
        mock_dependency_manager.start_all_services.assert_called_once()
    
    @patch('core.dependency_manager.dependency_manager')
    @pytest.mark.asyncio
    async def test_stop_all_services(self, mock_dependency_manager):
        """Test stop_all_services function"""
        mock_dependency_manager.stop_all_services = AsyncMock()
        
        await stop_all_services()
        
        mock_dependency_manager.stop_all_services.assert_called_once()


# ============================================================================
# MODULE 5: Integration and End-to-End Testing
# ============================================================================

class TestIntegration:
    """Integration tests for the dependency management system"""
    
    @pytest.fixture
    def dependency_manager(self):
        """Create a DependencyManager instance for integration tests"""
        return DependencyManager()
    
    @pytest.mark.asyncio
    async def test_complete_service_lifecycle(self, dependency_manager):
        """Test complete service lifecycle with dependencies"""
        # Register services with dependencies
        dependency_manager.register_service(
            "database",
            "database",
            lambda: "database_instance",
            ServicePriority.CRITICAL
        )
        
        dependency_manager.register_service(
            "cache",
            "cache",
            lambda: "cache_instance",
            ServicePriority.HIGH,
            dependencies=["database"]
        )
        
        dependency_manager.register_service(
            "api",
            "api",
            lambda: "api_instance",
            ServicePriority.NORMAL,
            dependencies=["cache", "database"]
        )
        
        # Start all services
        await dependency_manager.start_all_services()
        
        # Verify all services are running
        assert dependency_manager.services["database"].status == ServiceStatus.RUNNING
        assert dependency_manager.services["cache"].status == ServiceStatus.RUNNING
        assert dependency_manager.services["api"].status == ServiceStatus.RUNNING
        
        # Get health summary
        summary = dependency_manager.get_health_summary()
        assert summary["total_services"] == 3
        assert summary["health_percentage"] == 100.0
        
        # Stop all services
        await dependency_manager.stop_all_services()
        
        # Verify all services are stopped
        assert dependency_manager.services["database"].status == ServiceStatus.STOPPED
        assert dependency_manager.services["cache"].status == ServiceStatus.STOPPED
        assert dependency_manager.services["api"].status == ServiceStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_managed_services_context(self, dependency_manager):
        """Test managed_services context manager"""
        dependency_manager.register_service(
            "test_service",
            "test",
            lambda: "test_instance"
        )
        
        # Use context manager
        async with dependency_manager.managed_services():
            # Service should be running
            assert dependency_manager.services["test_service"].status == ServiceStatus.RUNNING
            assert dependency_manager.is_running is True
        
        # Service should be stopped after context exit
        assert dependency_manager.services["test_service"].status == ServiceStatus.STOPPED
        assert dependency_manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_service_dependency_chain(self, dependency_manager):
        """Test complex service dependency chain"""
        # Create a chain: A -> B -> C -> D
        dependency_manager.register_service("service_d", "type", lambda: "d", dependencies=[])
        dependency_manager.register_service("service_c", "type", lambda: "c", dependencies=["service_d"])
        dependency_manager.register_service("service_b", "type", lambda: "b", dependencies=["service_c"])
        dependency_manager.register_service("service_a", "type", lambda: "a", dependencies=["service_b"])
        
        # Start services
        await dependency_manager.start_all_services()
        
        # Verify startup order
        startup_order = dependency_manager.get_startup_order()
        assert startup_order.index("service_d") < startup_order.index("service_c")
        assert startup_order.index("service_c") < startup_order.index("service_b")
        assert startup_order.index("service_b") < startup_order.index("service_a")
        
        # All should be running
        for service_name in ["service_a", "service_b", "service_c", "service_d"]:
            assert dependency_manager.services[service_name].status == ServiceStatus.RUNNING


# ============================================================================
# MODULE 6: Error Handling and Edge Cases
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def dependency_manager(self):
        """Create a DependencyManager instance for error testing"""
        return DependencyManager()
    
    def test_circular_dependency_detection(self, dependency_manager):
        """Test that circular dependencies are detected"""
        dependency_manager.register_service("service_a", "type", Mock(), dependencies=["service_b"])
        dependency_manager.register_service("service_b", "type", Mock(), dependencies=["service_a"])
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            dependency_manager.get_startup_order()
    
    def test_missing_dependency(self, dependency_manager):
        """Test handling of missing dependencies"""
        dependency_manager.register_service("main_service", "type", Mock(), dependencies=["missing_dep"])
        
        # Should not raise error, but should log warning
        assert dependency_manager.check_dependencies("main_service") is False
    
    @pytest.mark.asyncio
    async def test_service_start_failure_handling(self, dependency_manager):
        """Test handling of service start failures"""
        # Create a factory that raises an exception
        def failing_factory():
            raise Exception("Factory failed")
        
        dependency_manager.register_service("failing_service", "type", failing_factory)
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            await dependency_manager.start_all_services()
        
        # Service should be in error state
        assert dependency_manager.services["failing_service"].status == ServiceStatus.ERROR


# ============================================================================
# MODULE 7: Performance and Scalability
# ============================================================================

class TestPerformance:
    """Test performance characteristics of the dependency management system"""
    
    @pytest.fixture
    def large_dependency_manager(self):
        """Create a DependencyManager with many services"""
        dm = DependencyManager()
        
        # Register 100 services with simple dependencies
        for i in range(100):
            deps = [f"service_{j}" for j in range(max(0, i-5), i)]
            dm.register_service(f"service_{i}", "type", lambda: f"instance_{i}", dependencies=deps)
        
        return dm
    
    def test_large_service_registration(self, large_dependency_manager):
        """Test registration of many services"""
        assert len(large_dependency_manager.services) == 100
        assert len(large_dependency_manager.service_factories) == 100
    
    def test_startup_order_calculation_performance(self, large_dependency_manager):
        """Test startup order calculation performance"""
        import time
        
        start_time = time.time()
        startup_order = large_dependency_manager.get_startup_order()
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert len(startup_order) == 100
    
    def test_health_summary_performance(self, large_dependency_manager):
        """Test health summary generation performance"""
        import time
        
        start_time = time.time()
        health = large_dependency_manager.get_health_summary()
        end_time = time.time()
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert end_time - start_time < 0.1
        assert health["total_services"] == 100
