"""
Tests for Metrics
=================
"""

import pytest
import asyncio
from ..core.metrics import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Create metrics collector for testing."""
    return MetricsCollector()


@pytest.mark.asyncio
async def test_increment_counter(metrics_collector):
    """Test incrementing a counter."""
    metrics_collector.increment("test_counter")
    metrics_collector.increment("test_counter", value=5)
    
    value = metrics_collector.get_counter("test_counter")
    
    assert value == 6


@pytest.mark.asyncio
async def test_set_gauge(metrics_collector):
    """Test setting a gauge."""
    metrics_collector.set_gauge("test_gauge", 100)
    
    value = metrics_collector.get_gauge("test_gauge")
    
    assert value == 100


@pytest.mark.asyncio
async def test_record_histogram(metrics_collector):
    """Test recording histogram values."""
    metrics_collector.record_histogram("test_histogram", 1.5)
    metrics_collector.record_histogram("test_histogram", 2.5)
    metrics_collector.record_histogram("test_histogram", 3.5)
    
    stats = metrics_collector.get_histogram_stats("test_histogram")
    
    assert stats is not None
    assert "count" in stats
    assert stats["count"] == 3


@pytest.mark.asyncio
async def test_get_all_metrics(metrics_collector):
    """Test getting all metrics."""
    metrics_collector.increment("counter1")
    metrics_collector.set_gauge("gauge1", 50)
    metrics_collector.record_histogram("hist1", 1.0)
    
    all_metrics = metrics_collector.get_all_metrics()
    
    assert "counters" in all_metrics
    assert "gauges" in all_metrics
    assert "histograms" in all_metrics


@pytest.mark.asyncio
async def test_reset_metrics(metrics_collector):
    """Test resetting metrics."""
    metrics_collector.increment("test_counter")
    metrics_collector.set_gauge("test_gauge", 100)
    
    metrics_collector.reset()
    
    assert metrics_collector.get_counter("test_counter") == 0
    assert metrics_collector.get_gauge("test_gauge") == 0


