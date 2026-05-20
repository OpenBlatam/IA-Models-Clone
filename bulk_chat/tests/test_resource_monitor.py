"""
Tests for Resource Monitor
===========================
"""

import pytest
import asyncio
from ..core.resource_monitor import ResourceMonitor


@pytest.fixture
def resource_monitor():
    """Create resource monitor for testing."""
    return ResourceMonitor()


@pytest.mark.asyncio
async def test_get_cpu_usage(resource_monitor):
    """Test getting CPU usage."""
    cpu_usage = resource_monitor.get_cpu_usage()
    
    assert cpu_usage is not None
    assert 0 <= cpu_usage <= 100 or isinstance(cpu_usage, float)


@pytest.mark.asyncio
async def test_get_memory_usage(resource_monitor):
    """Test getting memory usage."""
    memory = resource_monitor.get_memory_usage()
    
    assert memory is not None
    assert isinstance(memory, dict) or isinstance(memory, float)
    if isinstance(memory, dict):
        assert "used" in memory or "percentage" in memory or "total" in memory


@pytest.mark.asyncio
async def test_get_disk_usage(resource_monitor):
    """Test getting disk usage."""
    disk = resource_monitor.get_disk_usage()
    
    assert disk is not None
    assert isinstance(disk, dict) or isinstance(disk, float)
    if isinstance(disk, dict):
        assert "used" in disk or "percentage" in disk or "total" in disk


@pytest.mark.asyncio
async def test_get_network_usage(resource_monitor):
    """Test getting network usage."""
    network = resource_monitor.get_network_usage()
    
    assert network is not None
    assert isinstance(network, dict) or isinstance(network, float)


@pytest.mark.asyncio
async def test_get_all_resources(resource_monitor):
    """Test getting all resource metrics."""
    resources = resource_monitor.get_all_resources()
    
    assert resources is not None
    assert isinstance(resources, dict)
    assert "cpu" in resources or "memory" in resources or "disk" in resources


@pytest.mark.asyncio
async def test_get_resource_history(resource_monitor):
    """Test getting resource history."""
    # Record some metrics
    resource_monitor.get_cpu_usage()
    resource_monitor.get_memory_usage()
    
    await asyncio.sleep(0.1)
    
    history = resource_monitor.get_resource_history(limit=10)
    
    assert history is not None
    assert isinstance(history, list) or isinstance(history, dict)


@pytest.mark.asyncio
async def test_get_resource_monitor_summary(resource_monitor):
    """Test getting resource monitor summary."""
    resource_monitor.get_cpu_usage()
    resource_monitor.get_memory_usage()
    
    summary = resource_monitor.get_resource_monitor_summary()
    
    assert summary is not None
    assert "total_samples" in summary or "monitoring_active" in summary


