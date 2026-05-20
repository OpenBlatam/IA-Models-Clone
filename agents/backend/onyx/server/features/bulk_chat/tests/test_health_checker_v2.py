"""
Tests for Health Checker V2
============================
"""

import pytest
import asyncio
from ..core.health_checker_v2 import HealthCheckerV2, HealthStatus


@pytest.fixture
def health_checker_v2():
    """Create health checker V2 for testing."""
    return HealthCheckerV2()


@pytest.mark.asyncio
async def test_register_health_check(health_checker_v2):
    """Test registering a health check."""
    async def custom_check():
        return {"status": "healthy", "message": "OK"}
    
    health_checker_v2.register_health_check(
        "custom_component",
        custom_check,
        timeout=5.0
    )
    
    assert "custom_component" in health_checker_v2.health_checks


@pytest.mark.asyncio
async def test_check_health(health_checker_v2):
    """Test checking health."""
    async def simple_check():
        return {"status": "healthy"}
    
    health_checker_v2.register_health_check("test_component", simple_check)
    
    health = await health_checker_v2.check_health("test_component")
    
    assert health is not None
    assert health["status"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_check_all_health(health_checker_v2):
    """Test checking all components health."""
    async def check1():
        return {"status": "healthy"}
    
    async def check2():
        return {"status": "healthy"}
    
    health_checker_v2.register_health_check("component1", check1)
    health_checker_v2.register_health_check("component2", check2)
    
    all_health = await health_checker_v2.check_all_health()
    
    assert all_health is not None
    assert len(all_health) >= 2


@pytest.mark.asyncio
async def test_get_health_summary(health_checker_v2):
    """Test getting health summary."""
    async def check():
        return {"status": "healthy"}
    
    health_checker_v2.register_health_check("test_component", check)
    
    summary = health_checker_v2.get_health_summary()
    
    assert summary is not None
    assert "total_components" in summary or "healthy" in summary or "unhealthy" in summary


