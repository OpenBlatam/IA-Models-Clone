"""
Tests for Backpressure Manager
===============================
"""

import pytest
import asyncio
from ..core.backpressure_manager import BackpressureManager, BackpressureLevel


@pytest.fixture
def backpressure_manager():
    """Create backpressure manager for testing."""
    return BackpressureManager()


@pytest.mark.asyncio
async def test_add_rule(backpressure_manager):
    """Test adding a backpressure rule."""
    rule_id = backpressure_manager.add_rule(
        rule_id="test_rule",
        component_id="test_component",
        queue_threshold=100,
        error_rate_threshold=0.1,
        latency_threshold=2.0
    )
    
    assert rule_id == "test_rule"
    assert "test_rule" in backpressure_manager.rules


@pytest.mark.asyncio
async def test_record_metric(backpressure_manager):
    """Test recording metrics."""
    backpressure_manager.add_rule("test_rule", "test_component")
    
    backpressure_manager.record_metric(
        component_id="test_component",
        queue_size=50,
        error_rate=0.05,
        latency=1.0,
        system_load=0.6
    )
    
    # Wait for async evaluation
    await asyncio.sleep(0.1)
    
    metrics = backpressure_manager.metrics.get("test_component")
    assert len(metrics) > 0


@pytest.mark.asyncio
async def test_get_backpressure_level(backpressure_manager):
    """Test getting backpressure level."""
    backpressure_manager.add_rule("test_rule", "test_component")
    
    level = backpressure_manager.get_backpressure_level("test_component")
    
    assert level in BackpressureLevel


@pytest.mark.asyncio
async def test_should_accept_request(backpressure_manager):
    """Test should accept request."""
    backpressure_manager.add_rule("test_rule", "test_component")
    
    # Set to low level
    backpressure_manager.current_levels["test_component"] = BackpressureLevel.LOW
    
    allowed = backpressure_manager.should_accept_request("test_component")
    assert allowed is True
    
    # Set to critical
    backpressure_manager.current_levels["test_component"] = BackpressureLevel.CRITICAL
    
    allowed = backpressure_manager.should_accept_request("test_component")
    assert allowed is False


@pytest.mark.asyncio
async def test_get_backpressure_status(backpressure_manager):
    """Test getting backpressure status."""
    backpressure_manager.add_rule("test_rule", "test_component")
    
    status = backpressure_manager.get_backpressure_status("test_component")
    
    assert status["component_id"] == "test_component"
    assert "level" in status


@pytest.mark.asyncio
async def test_get_backpressure_manager_summary(backpressure_manager):
    """Test getting backpressure manager summary."""
    backpressure_manager.add_rule("rule_1", "component_1")
    backpressure_manager.add_rule("rule_2", "component_2")
    
    summary = backpressure_manager.get_backpressure_manager_summary()
    
    assert summary["total_rules"] >= 2

