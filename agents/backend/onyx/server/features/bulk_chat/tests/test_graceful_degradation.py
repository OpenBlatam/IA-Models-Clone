"""
Tests for Graceful Degradation Manager
=======================================
"""

import pytest
import asyncio
from ..core.graceful_degradation_manager import GracefulDegradationManager, DegradationLevel


@pytest.fixture
def degradation_manager():
    """Create graceful degradation manager for testing."""
    return GracefulDegradationManager()


def test_register_fallback(degradation_manager):
    """Test registering a fallback."""
    def fallback_func():
        return "fallback_result"
    
    degradation_manager.register_fallback(
        component_id="test_component",
        fallback=fallback_func,
        degradation_level=DegradationLevel.PARTIAL
    )
    
    assert "test_component" in degradation_manager.fallbacks


@pytest.mark.asyncio
async def test_execute_with_fallback(degradation_manager):
    """Test executing with fallback."""
    def fallback():
        return "fallback_result"
    
    async def main_operation():
        raise Exception("Main operation failed")
    
    degradation_manager.register_fallback("test_component", fallback, DegradationLevel.PARTIAL)
    
    result = await degradation_manager.execute_with_fallback(
        "test_component",
        main_operation
    )
    
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_set_degradation_level(degradation_manager):
    """Test setting degradation level."""
    degradation_manager.set_degradation_level("test_component", DegradationLevel.PARTIAL)
    
    level = degradation_manager.get_degradation_level("test_component")
    
    assert level == DegradationLevel.PARTIAL


@pytest.mark.asyncio
async def test_get_degradation_status(degradation_manager):
    """Test getting degradation status."""
    degradation_manager.set_degradation_level("component1", DegradationLevel.PARTIAL)
    degradation_manager.set_degradation_level("component2", DegradationLevel.FULL)
    
    status = degradation_manager.get_degradation_status()
    
    assert status is not None
    assert "components" in status or "total_components" in status


@pytest.mark.asyncio
async def test_get_graceful_degradation_summary(degradation_manager):
    """Test getting graceful degradation summary."""
    degradation_manager.set_degradation_level("component1", DegradationLevel.PARTIAL)
    
    summary = degradation_manager.get_graceful_degradation_summary()
    
    assert summary is not None
    assert "total_components" in summary or "components_by_level" in summary


