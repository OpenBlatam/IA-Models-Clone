"""
Tests for RealtimeMonitor utility
"""

import pytest
import asyncio
from unittest.mock import patch

from ..utils.realtime_monitoring import RealtimeMonitor


class TestRealtimeMonitor:
    """Test suite for RealtimeMonitor"""

    def test_init(self):
        """Test RealtimeMonitor initialization"""
        monitor = RealtimeMonitor()
        assert len(monitor.metrics_buffer) == 0
        assert monitor.alerts == []
        assert monitor.monitoring_active is False

    def test_collect_metrics(self):
        """Test collecting system metrics"""
        monitor = RealtimeMonitor()
        
        metrics = monitor._collect_metrics()
        
        assert "timestamp" in metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "cpu_count" in metrics
        assert "memory_total" in metrics
        assert "memory_available" in metrics

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        monitor = RealtimeMonitor()
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor.start_monitoring(interval_seconds=0.1))
        
        # Wait a bit
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Wait for task to finish
        await asyncio.sleep(0.2)
        
        # Should have collected some metrics
        assert len(monitor.metrics_buffer) > 0
        assert monitor.monitoring_active is False

    def test_check_alerts_high_cpu(self):
        """Test alert generation for high CPU"""
        monitor = RealtimeMonitor()
        
        metrics = {
            "timestamp": "2024-01-01T00:00:00",
            "cpu_percent": 95.0,
            "memory_percent": 50.0,
        }
        
        monitor._check_alerts(metrics)
        
        assert len(monitor.alerts) > 0
        assert any(alert["type"] == "high_cpu" for alert in monitor.alerts)

    def test_check_alerts_high_memory(self):
        """Test alert generation for high memory"""
        monitor = RealtimeMonitor()
        
        metrics = {
            "timestamp": "2024-01-01T00:00:00",
            "cpu_percent": 50.0,
            "memory_percent": 95.0,
        }
        
        monitor._check_alerts(metrics)
        
        assert len(monitor.alerts) > 0
        assert any(alert["type"] == "high_memory" for alert in monitor.alerts)

    def test_get_current_metrics(self):
        """Test getting current metrics"""
        monitor = RealtimeMonitor()
        
        # Add some metrics
        metrics1 = monitor._collect_metrics()
        monitor.metrics_buffer.append(metrics1)
        
        current = monitor.get_current_metrics()
        
        assert current is not None
        assert "timestamp" in current

    def test_get_metrics_history(self):
        """Test getting metrics history"""
        monitor = RealtimeMonitor()
        
        # Add multiple metrics
        for i in range(5):
            metrics = monitor._collect_metrics()
            monitor.metrics_buffer.append(metrics)
        
        history = monitor.get_metrics_history(limit=3)
        
        assert len(history) == 3

    def test_get_recent_alerts(self):
        """Test getting recent alerts"""
        monitor = RealtimeMonitor()
        
        # Generate some alerts
        for i in range(3):
            metrics = {
                "timestamp": f"2024-01-01T00:00:0{i}",
                "cpu_percent": 95.0,
                "memory_percent": 50.0,
            }
            monitor._check_alerts(metrics)
        
        alerts = monitor.get_recent_alerts(limit=2)
        
        assert len(alerts) <= 2

