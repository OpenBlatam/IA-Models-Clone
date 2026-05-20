"""
Tests for Performance Monitor
==============================
"""

import pytest
import asyncio
from ..core.performance_monitor import PerformanceMonitor, MetricType


@pytest.fixture
def performance_monitor():
    """Create performance monitor for testing."""
    return PerformanceMonitor()


@pytest.mark.asyncio
async def test_record_metric(performance_monitor):
    """Test recording a metric."""
    performance_monitor.record_metric(
        metric_name="test_metric",
        value=100.0,
        metric_type=MetricType.GAUGE,
        tags={"service": "test"}
    )
    
    metrics = performance_monitor.get_metrics("test_metric")
    
    assert len(metrics) >= 1


@pytest.mark.asyncio
async def test_record_latency(performance_monitor):
    """Test recording latency."""
    performance_monitor.record_latency(
        operation_name="test_operation",
        latency_ms=50.5,
        tags={"endpoint": "/api/test"}
    )
    
    latency_stats = performance_monitor.get_latency_stats("test_operation")
    
    assert latency_stats is not None
    assert "count" in latency_stats or "avg" in latency_stats or "p95" in latency_stats


@pytest.mark.asyncio
async def test_get_slow_operations(performance_monitor):
    """Test getting slow operations."""
    performance_monitor.record_latency("fast_op", 10.0)
    performance_monitor.record_latency("slow_op", 5000.0)
    
    slow_ops = performance_monitor.get_slow_operations(threshold_ms=100.0, limit=10)
    
    assert len(slow_ops) >= 1
    assert any(op["operation"] == "slow_op" for op in slow_ops)


@pytest.mark.asyncio
async def test_get_performance_summary(performance_monitor):
    """Test getting performance summary."""
    performance_monitor.record_metric("metric1", 100.0, MetricType.COUNTER)
    performance_monitor.record_latency("op1", 50.0)
    
    summary = performance_monitor.get_performance_summary()
    
    assert summary is not None
    assert "total_metrics" in summary or "operations" in summary


