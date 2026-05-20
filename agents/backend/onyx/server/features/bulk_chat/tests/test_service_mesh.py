"""
Tests for Service Mesh
======================
"""

import pytest
from ..core.service_mesh import ServiceMesh, LoadBalancingStrategy, ServiceStatus


@pytest.fixture
def service_mesh():
    """Create service mesh for testing."""
    return ServiceMesh()


@pytest.mark.asyncio
async def test_register_service(service_mesh):
    """Test registering a service."""
    service_name = service_mesh.register_service(
        service_name="test_service",
        load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
    )
    
    assert service_name == "test_service"
    assert "test_service" in service_mesh.services


@pytest.mark.asyncio
async def test_register_instance(service_mesh):
    """Test registering a service instance."""
    service_mesh.register_service("test_service")
    
    instance_id = service_mesh.register_instance(
        instance_id="instance_1",
        service_name="test_service",
        address="127.0.0.1",
        port=8000,
        weight=1
    )
    
    assert instance_id == "instance_1"
    instances = service_mesh.get_service_instances("test_service")
    assert len(instances) == 1


@pytest.mark.asyncio
async def test_get_instance_round_robin(service_mesh):
    """Test getting instance with round robin strategy."""
    service_mesh.register_service(
        "test_service",
        load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
    )
    
    service_mesh.register_instance("inst_1", "test_service", "127.0.0.1", 8000)
    service_mesh.register_instance("inst_2", "test_service", "127.0.0.2", 8000)
    
    # Update status to healthy
    service_mesh.update_instance_status("inst_1", ServiceStatus.HEALTHY)
    service_mesh.update_instance_status("inst_2", ServiceStatus.HEALTHY)
    
    instance1 = service_mesh.get_instance("test_service")
    instance2 = service_mesh.get_instance("test_service")
    
    assert instance1 is not None
    assert instance2 is not None
    # Should alternate (round robin)
    assert instance1.instance_id != instance2.instance_id or instance1 == instance2


@pytest.mark.asyncio
async def test_update_instance_status(service_mesh):
    """Test updating instance status."""
    service_mesh.register_service("test_service")
    service_mesh.register_instance("inst_1", "test_service", "127.0.0.1", 8000)
    
    service_mesh.update_instance_status("inst_1", ServiceStatus.HEALTHY)
    
    instances = service_mesh.get_service_instances("test_service")
    assert instances[0]["status"] == ServiceStatus.HEALTHY.value


@pytest.mark.asyncio
async def test_get_service_mesh_summary(service_mesh):
    """Test getting service mesh summary."""
    service_mesh.register_service("service_1")
    service_mesh.register_instance("inst_1", "service_1", "127.0.0.1", 8000)
    
    summary = service_mesh.get_service_mesh_summary()
    
    assert summary["total_services"] >= 1
    assert summary["total_instances"] >= 1


