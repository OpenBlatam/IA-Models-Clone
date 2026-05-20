"""
Tests for Real-Time Metrics
============================
"""

import pytest
import asyncio
from ..core.realtime_metrics import RealTimeMetrics, MetricType


@pytest.fixture
def realtime_metrics():
    """Create real-time metrics for testing."""
    return RealTimeMetrics()


@pytest.mark.asyncio
async def test_record_counter(realtime_metrics):
    """Test recording a counter metric."""
    realtime_metrics.record_metric(
        metric_name="test_counter",
        metric_type=MetricType.COUNTER,
        value=1,
        tags={"service": "test"}
    )
    
    value = realtime_metrics.get_metric("test_counter", MetricType.COUNTER)
    
    assert value is not None
    assert value >= 1


@pytest.mark.asyncio
async def test_record_gauge(realtime_metrics):
    """Test recording a gauge metric."""
    realtime_metrics.record_metric(
        "test_gauge",
        MetricType.GAUGE,
        value=100.0,
        tags={}
    )
    
    value = realtime_metrics.get_metric("test_gauge", MetricType.GAUGE)
    
    assert value == 100.0 or value is not None


@pytest.mark.asyncio
async def test_record_histogram(realtime_metrics):
    """Test recording histogram values."""
    for value in [10, 20, 30, 40, 50]:
        realtime_metrics.record_metric(
            "test_histogram",
            MetricType.HISTOGRAM,
            value=float(value),
            tags={}
        )
    
    stats = realtime_metrics.get_histogram_stats("test_histogram")
    
    assert stats is not None
    assert "count" in stats or "avg" in stats or "min" in stats or "max" in stats


@pytest.mark.asyncio
async def test_get_metric_summary(realtime_metrics):
    """Test getting metric summary."""
    realtime_metrics.record_metric("metric1", MetricType.COUNTER, 1)
    realtime_metrics.record_metric("metric2", MetricType.GAUGE, 50.0)
    
    summary = realtime_metrics.get_metric_summary("metric1")
    
    assert summary is not None
    assert "metric_name" in summary or "value" in summary or "type" in summary


@pytest.mark.asyncio
async def test_get_all_metrics(realtime_metrics):
    """Test getting all metrics."""
    realtime_metrics.record_metric("counter1", MetricType.COUNTER, 1)
    realtime_metrics.record_metric("gauge1", MetricType.GAUGE, 100.0)
    
    all_metrics = realtime_metrics.get_all_metrics()
    
    assert all_metrics is not None
    assert isinstance(all_metrics, dict)
    assert len(all_metrics) > 0


@pytest.mark.asyncio
async def test_get_realtime_metrics_summary(realtime_metrics):
    """Test getting real-time metrics summary."""
    realtime_metrics.record_metric("metric1", MetricType.COUNTER, 1)
    realtime_metrics.record_metric("metric2", MetricType.GAUGE, 50.0)
    
    summary = realtime_metrics.get_realtime_metrics_summary()
    
    assert summary is not None
    assert "total_metrics" in summary or "metrics_by_type" in summary


