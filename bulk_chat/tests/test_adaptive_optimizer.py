"""
Tests for Adaptive Optimizer
=============================
"""

import pytest
import asyncio
from ..core.adaptive_optimizer import AdaptiveOptimizer


@pytest.fixture
def adaptive_optimizer():
    """Create adaptive optimizer for testing."""
    return AdaptiveOptimizer()


@pytest.mark.asyncio
async def test_optimize_parameter(adaptive_optimizer):
    """Test optimizing a parameter."""
    optimization_id = adaptive_optimizer.optimize_parameter(
        parameter_name="batch_size",
        current_value=32,
        target_metric="throughput",
        optimization_target="maximize"
    )
    
    assert optimization_id is not None
    assert optimization_id in adaptive_optimizer.optimizations


@pytest.mark.asyncio
async def test_record_performance(adaptive_optimizer):
    """Test recording performance metrics."""
    optimization_id = adaptive_optimizer.optimize_parameter(
        "batch_size", 32, "throughput", "maximize"
    )
    
    adaptive_optimizer.record_performance(
        optimization_id,
        parameter_value=64,
        metric_value=100.0
    )
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    optimization = adaptive_optimizer.optimizations.get(optimization_id)
    assert optimization is not None


@pytest.mark.asyncio
async def test_get_optimal_value(adaptive_optimizer):
    """Test getting optimal parameter value."""
    optimization_id = adaptive_optimizer.optimize_parameter(
        "batch_size", 32, "throughput", "maximize"
    )
    
    adaptive_optimizer.record_performance(optimization_id, 32, 50.0)
    adaptive_optimizer.record_performance(optimization_id, 64, 100.0)
    adaptive_optimizer.record_performance(optimization_id, 128, 80.0)
    
    await asyncio.sleep(0.1)
    
    optimal = adaptive_optimizer.get_optimal_value(optimization_id)
    
    assert optimal is not None
    assert optimal in [32, 64, 128]


@pytest.mark.asyncio
async def test_get_adaptive_optimizer_summary(adaptive_optimizer):
    """Test getting adaptive optimizer summary."""
    adaptive_optimizer.optimize_parameter("param1", 10, "metric1", "maximize")
    adaptive_optimizer.optimize_parameter("param2", 20, "metric2", "minimize")
    
    summary = adaptive_optimizer.get_adaptive_optimizer_summary()
    
    assert summary is not None
    assert "total_optimizations" in summary or "active_optimizations" in summary


