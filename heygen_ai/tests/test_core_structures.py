"""
Tests for core data structures used across the system.
Modular test structure for better organization.
"""

import pytest
from core.base_service import ServiceStatus, ServiceType, HealthCheckResult
from core.dependency_manager import ServicePriority, ServiceInfo


# ============================================================================
# MODULE: Core Enums and Data Structures
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
    
    def test_service_status_string_representation(self):
        """Test string representation of ServiceStatus"""
        assert str(ServiceStatus.UNKNOWN) == "ServiceStatus.UNKNOWN"
        assert str(ServiceStatus.RUNNING) == "ServiceStatus.RUNNING"
    
    def test_service_status_comparison(self):
        """Test ServiceStatus comparison operations"""
        # Should be able to compare enum members
        assert ServiceStatus.UNKNOWN != ServiceStatus.RUNNING
        assert ServiceStatus.RUNNING == ServiceStatus.RUNNING


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
    
    def test_service_priority_immutability(self):
        """Test that ServicePriority values are immutable"""
        assert ServicePriority.CRITICAL.value == 0
        assert ServicePriority.HIGH.value == 1
    
    def test_service_priority_string_representation(self):
        """Test string representation of ServicePriority"""
        assert str(ServicePriority.CRITICAL) == "ServicePriority.CRITICAL"
        assert str(ServicePriority.NORMAL) == "ServicePriority.NORMAL"


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
    
    def test_service_info_data_integrity(self):
        """Test that ServiceInfo fields maintain data integrity"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        # Test that we can modify the collections without affecting the original
        original_deps = info.dependencies.copy()
        original_metadata = info.metadata.copy()
        
        # Modify the collections
        info.dependencies.append("new_dep")
        info.metadata["new_key"] = "new_value"
        
        # Original collections should remain unchanged
        assert original_deps == []
        assert original_metadata == {}
        
        # New collections should have the new values
        assert "new_dep" in info.dependencies
        assert "new_key" in info.metadata
    
    def test_service_info_equality(self):
        """Test ServiceInfo equality comparison"""
        info1 = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        info2 = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        # Should be equal if all fields are the same
        assert info1 == info2
        
        # Should not be equal if fields differ
        info3 = ServiceInfo(
            name="different-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        assert info1 != info3
    
    def test_service_info_repr(self):
        """Test ServiceInfo string representation"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        repr_str = repr(info)
        assert "test-service" in repr_str
        assert "test" in repr_str
        assert "ServicePriority.NORMAL" in repr_str
        assert "ServiceStatus.STOPPED" in repr_str


# ============================================================================
# MODULE: Data Structure Validation
# ============================================================================

class TestDataStructureValidation:
    """Test validation of data structures"""
    
    def test_service_info_field_types(self):
        """Test that ServiceInfo fields have correct types"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        assert isinstance(info.name, str)
        assert isinstance(info.service_type, str)
        assert isinstance(info.priority, ServicePriority)
        assert isinstance(info.status, ServiceStatus)
        assert isinstance(info.dependencies, list)
        assert isinstance(info.metadata, dict)
    
    def test_service_info_default_values(self):
        """Test that ServiceInfo has appropriate default values"""
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.STOPPED
        )
        
        assert info.start_time is None
        assert info.stop_time is None
        assert info.error_count == 0
        assert info.last_error is None
        assert info.dependencies == []
        assert info.metadata == {}
    
    def test_enum_value_consistency(self):
        """Test that enum values are consistent and meaningful"""
        # ServiceStatus values should be descriptive strings
        for status in ServiceStatus:
            assert isinstance(status.value, str)
            assert len(status.value) > 0
        
        # ServicePriority values should be ordered integers
        for priority in ServicePriority:
            assert isinstance(priority.value, int)
            assert priority.value >= 0
        
        # Priority values should be in ascending order
        priority_values = [p.value for p in ServicePriority]
        assert priority_values == sorted(priority_values)
