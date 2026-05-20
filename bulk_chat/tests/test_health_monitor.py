"""
Tests for Health Monitor
========================
"""

import pytest
import asyncio
from ..core.health_monitor import HealthMonitor


@pytest.fixture
def health_monitor():
    """Create health monitor for testing."""
    return HealthMonitor()


@pytest.mark.asyncio
async def test_check_health(health_monitor):
    """Test health check."""
    health = await health_monitor.check_health()
    
    assert health is not None
    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy", "degraded"]


@pytest.mark.asyncio
async def test_check_component_health(health_monitor):
    """Test component health check."""
    health = await health_monitor.check_component_health("database")
    
    assert health is not None
    assert "component" in health
    assert health["component"] == "database"


@pytest.mark.asyncio
async def test_get_system_metrics(health_monitor):
    """Test getting system metrics."""
    metrics = health_monitor.get_system_metrics()
    
    assert metrics is not None
    assert "cpu" in metrics or "memory" in metrics or "disk" in metrics


@pytest.mark.asyncio
async def test_register_health_check(health_monitor):
    """Test registering a custom health check."""
    def custom_check():
        return {"status": "healthy", "message": "Custom check passed"}
    
    health_monitor.register_health_check("custom_component", custom_check)
    
    health = await health_monitor.check_component_health("custom_component")
    
    assert health is not None
    assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_get_health_summary(health_monitor):
    """Test getting health summary."""
    summary = health_monitor.get_health_summary()
    
    assert summary is not None
    assert "overall_status" in summary or "components" in summary


