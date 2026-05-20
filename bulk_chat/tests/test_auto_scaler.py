"""
Tests for Auto Scaler
=====================
"""

import pytest
import asyncio
from ..core.auto_scaler import AutoScaler, ScalingAction


@pytest.fixture
def auto_scaler():
    """Create auto scaler for testing."""
    return AutoScaler(
        min_instances=1,
        max_instances=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3
    )


@pytest.mark.asyncio
async def test_evaluate_scaling(auto_scaler):
    """Test evaluating scaling needs."""
    # Record high load
    auto_scaler.record_metric("cpu_usage", 0.9)
    auto_scaler.record_metric("memory_usage", 0.85)
    
    action = await auto_scaler.evaluate_scaling()
    
    assert action is not None
    assert action in ScalingAction


@pytest.mark.asyncio
async def test_scale_up(auto_scaler):
    """Test scale up action."""
    auto_scaler.current_instances = 2
    
    result = await auto_scaler.scale_up(target_instances=5)
    
    assert result is True
    assert auto_scaler.current_instances >= 2


@pytest.mark.asyncio
async def test_scale_down(auto_scaler):
    """Test scale down action."""
    auto_scaler.current_instances = 5
    
    result = await auto_scaler.scale_down(target_instances=2)
    
    assert result is True
    assert auto_scaler.current_instances <= 5


@pytest.mark.asyncio
async def test_get_scaling_status(auto_scaler):
    """Test getting scaling status."""
    status = auto_scaler.get_scaling_status()
    
    assert status is not None
    assert "current_instances" in status
    assert "min_instances" in status
    assert "max_instances" in status


@pytest.mark.asyncio
async def test_get_auto_scaler_summary(auto_scaler):
    """Test getting auto scaler summary."""
    auto_scaler.record_metric("cpu_usage", 0.7)
    
    summary = auto_scaler.get_auto_scaler_summary()
    
    assert summary is not None
    assert "current_instances" in summary


