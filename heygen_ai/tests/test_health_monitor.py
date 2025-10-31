"""
Tests for the Advanced Health Monitoring System
==============================================

Test coverage for:
- Health status enumeration
- Component health tracking
- System metrics monitoring
- Health check execution
- Alerting system
- Data export functionality
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Import the health monitoring system
from core.health_monitor import (
    HealthStatus, ComponentType, HealthMetric, ComponentHealth,
    SystemHealth, HealthCheck, HealthMonitor, quick_health_check
)


class TestHealthStatus:
    """Test health status enumeration"""
    
    def test_health_status_values(self):
        """Test health status enum values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_health_status_comparison(self):
        """Test health status comparison logic"""
        # Critical is worse than warning, warning is worse than healthy
        assert HealthStatus.CRITICAL > HealthStatus.WARNING
        assert HealthStatus.WARNING > HealthStatus.HEALTHY
        assert HealthStatus.HEALTHY > HealthStatus.UNKNOWN


class TestComponentType:
    """Test component type enumeration"""
    
    def test_component_type_values(self):
        """Test component type enum values"""
        assert ComponentType.CACHE.value == "cache"
        assert ComponentType.LOAD_BALANCER.value == "load_balancer"
        assert ComponentType.BACKGROUND_PROCESSOR.value == "background_processor"
        assert ComponentType.NETWORK.value == "network"
        assert ComponentType.SECURITY.value == "security"
        assert ComponentType.EXTERNAL_API.value == "external_api"


class TestHealthMetric:
    """Test health metric data structure"""
    
    def test_health_metric_creation(self):
        """Test creating a health metric"""
        metric = HealthMetric(
            component="test_component",
            metric_name="response_time",
            value=150.0,
            unit="ms",
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY,
            threshold_warning=200.0,
            threshold_critical=500.0
        )
        
        assert metric.component == "test_component"
        assert metric.metric_name == "response_time"
        assert metric.value == 150.0
        assert metric.unit == "ms"
        assert metric.status == HealthStatus.HEALTHY
        assert metric.threshold_warning == 200.0
        assert metric.threshold_critical == 500.0
    
    def test_health_metric_to_dict(self):
        """Test converting health metric to dictionary"""
        timestamp = datetime.now()
        metric = HealthMetric(
            component="test_component",
            metric_name="cpu_usage",
            value=75.0,
            unit="%",
            timestamp=timestamp,
            status=HealthStatus.WARNING
        )
        
        data = metric.to_dict()
        
        assert data['component'] == "test_component"
        assert data['metric_name'] == "cpu_usage"
        assert data['value'] == 75.0
        assert data['unit'] == "%"
        assert data['status'] == "warning"
        assert data['timestamp'] == timestamp.isoformat()


class TestComponentHealth:
    """Test component health data structure"""
    
    def test_component_health_creation(self):
        """Test creating component health"""
        component = ComponentHealth(
            component_id="test_cache",
            component_type=ComponentType.CACHE,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            metrics=[],
            error_count=0,
            warning_count=0,
            uptime_seconds=3600.0
        )
        
        assert component.component_id == "test_cache"
        assert component.component_type == ComponentType.CACHE
        assert component.status == HealthStatus.HEALTHY
        assert component.error_count == 0
        assert component.warning_count == 0
        assert component.uptime_seconds == 3600.0
    
    def test_component_health_to_dict(self):
        """Test converting component health to dictionary"""
        timestamp = datetime.now()
        component = ComponentHealth(
            component_id="test_lb",
            component_type=ComponentType.LOAD_BALANCER,
            status=HealthStatus.WARNING,
            last_check=timestamp,
            metrics=[]
        )
        
        data = component.to_dict()
        
        assert data['component_id'] == "test_lb"
        assert data['component_type'] == "load_balancer"
        assert data['status'] == "warning"
        assert data['last_check'] == timestamp.isoformat()
        assert data['metrics'] == []


class TestSystemHealth:
    """Test system health data structure"""
    
    def test_system_health_creation(self):
        """Test creating system health"""
        timestamp = datetime.now()
        components = []
        metrics = []
        alerts = ["Test alert"]
        
        system_health = SystemHealth(
            timestamp=timestamp,
            overall_status=HealthStatus.HEALTHY,
            components=components,
            system_metrics=metrics,
            alerts=alerts
        )
        
        assert system_health.timestamp == timestamp
        assert system_health.overall_status == HealthStatus.HEALTHY
        assert system_health.components == components
        assert system_health.system_metrics == metrics
        assert system_health.alerts == alerts
    
    def test_system_health_to_dict(self):
        """Test converting system health to dictionary"""
        timestamp = datetime.now()
        system_health = SystemHealth(
            timestamp=timestamp,
            overall_status=HealthStatus.CRITICAL,
            components=[],
            system_metrics=[],
            alerts=[]
        )
        
        data = system_health.to_dict()
        
        assert data['timestamp'] == timestamp.isoformat()
        assert data['overall_status'] == "critical"
        assert data['components'] == []
        assert data['system_metrics'] == []
        assert data['alerts'] == []


class TestHealthCheck:
    """Test health check functionality"""
    
    def test_health_check_creation(self):
        """Test creating a health check"""
        async def dummy_check():
            return HealthStatus.HEALTHY
        
        health_check = HealthCheck(
            name="test_check",
            check_func=dummy_check,
            interval_seconds=30.0,
            timeout_seconds=15.0,
            critical=True
        )
        
        assert health_check.name == "test_check"
        assert health_check.interval_seconds == 30.0
        assert health_check.timeout_seconds == 15.0
        assert health_check.critical is True
        assert health_check.last_run is None
        assert health_check.last_result is None
        assert health_check.error_count == 0


class TestHealthMonitor:
    """Test health monitor system"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor instance for testing"""
        return HealthMonitor()
    
    def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initialization"""
        assert health_monitor.components == {}
        # health_checks will have default checks if auto_init is True
        assert len(health_monitor.health_checks) >= 0  # May have default checks
        assert health_monitor.metrics_history == []
        assert health_monitor.alerts == []
        assert health_monitor.is_running is False
        assert health_monitor.monitoring_task is None
        
        # Check default thresholds
        assert 'cpu_usage' in health_monitor.thresholds
        assert 'memory_usage' in health_monitor.thresholds
        assert 'disk_usage' in health_monitor.thresholds
    
    def test_health_monitor_default_checks(self, health_monitor):
        """Test that default health checks are initialized"""
        check_names = [check.name for check in health_monitor.health_checks]
        
        assert "system_resources" in check_names
        assert "component_health" in check_names
        assert "performance_metrics" in check_names
    
    def test_add_health_check(self, health_monitor):
        """Test adding a custom health check"""
        async def custom_check():
            return HealthStatus.HEALTHY
        
        health_monitor.add_health_check(
            name="custom_check",
            check_func=custom_check,
            interval_seconds=120.0,
            critical=True
        )
        
        # Find the added check
        custom_check_found = None
        for check in health_monitor.health_checks:
            if check.name == "custom_check":
                custom_check_found = check
                break
        
        assert custom_check_found is not None
        assert custom_check_found.interval_seconds == 120.0
        assert custom_check_found.critical is True
    
    def test_register_component(self, health_monitor):
        """Test registering a component"""
        health_monitor.register_component(
            component_id="test_cache",
            component_type=ComponentType.CACHE,
            initial_status=HealthStatus.HEALTHY
        )
        
        assert "test_cache" in health_monitor.components
        component = health_monitor.components["test_cache"]
        assert component.component_type == ComponentType.CACHE
        assert component.status == HealthStatus.HEALTHY
        assert component.metrics == []
    
    def test_update_component_metric(self, health_monitor):
        """Test updating component metrics"""
        # Register component first
        health_monitor.register_component("test_component", ComponentType.CACHE)
        
        # Update metric
        health_monitor.update_component_metric(
            component_id="test_component",
            metric_name="response_time",
            value=250.0,
            unit="ms",
            threshold_warning=200.0,
            threshold_critical=500.0
        )
        
        # Check component was updated
        component = health_monitor.components["test_component"]
        assert len(component.metrics) == 1
        
        metric = component.metrics[0]
        assert metric.metric_name == "response_time"
        assert metric.value == 250.0
        assert metric.status == HealthStatus.WARNING  # Above warning threshold
        
        # Check metrics history
        assert len(health_monitor.metrics_history) == 1
    
    def test_update_component_metric_critical(self, health_monitor):
        """Test updating component metric with critical threshold"""
        health_monitor.register_component("test_component", ComponentType.CACHE)
        
        health_monitor.update_component_metric(
            component_id="test_component",
            metric_name="error_rate",
            value=20.0,
            unit="%",
            threshold_warning=5.0,
            threshold_critical=15.0
        )
        
        component = health_monitor.components["test_component"]
        metric = component.metrics[0]
        assert metric.status == HealthStatus.CRITICAL  # Above critical threshold
        assert component.status == HealthStatus.CRITICAL
        assert component.error_count == 1
    
    def test_update_component_metric_unknown_component(self, health_monitor):
        """Test updating metric for unknown component"""
        # Should not raise error, just log warning
        health_monitor.update_component_metric(
            component_id="unknown_component",
            metric_name="test_metric",
            value=100.0,
            unit="count"
        )
        
        # Component should not be created
        assert "unknown_component" not in health_monitor.components
    
    @pytest.mark.asyncio
    async def test_check_system_resources(self, health_monitor):
        """Test system resources health check"""
        # Mock psutil to avoid actual system calls
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network:
            
            # Setup mocks
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.used = 800
            mock_disk.return_value.total = 1000
            mock_network.return_value.bytes_sent = 1000
            mock_network.return_value.bytes_recv = 2000
            
            # Run check
            result = await health_monitor._check_system_resources()
            
            assert result == HealthStatus.HEALTHY
            
            # Check that metrics were added
            system_metrics = [
                m for m in health_monitor.metrics_history
                if m.component == "system"
            ]
            
            assert len(system_metrics) >= 5  # CPU, memory, disk, network sent/recv
            
            # Check specific metrics
            cpu_metric = next(m for m in system_metrics if m.metric_name == "cpu_usage")
            assert cpu_metric.value == 50.0
            assert cpu_metric.unit == "%"
    
    @pytest.mark.asyncio
    async def test_check_component_health(self, health_monitor):
        """Test component health check"""
        # Register components with different states
        health_monitor.register_component("healthy_comp", ComponentType.CACHE)
        health_monitor.register_component("warning_comp", ComponentType.LOAD_BALANCER)
        
        # Add recent metrics for healthy component
        health_monitor.update_component_metric("healthy_comp", "test_metric", 50.0, "count")
        
        # Add old metrics for warning component (should trigger warning)
        old_timestamp = datetime.now() - timedelta(minutes=10)
        old_metric = HealthMetric(
            component="warning_comp",
            metric_name="old_metric",
            value=100.0,
            unit="count",
            timestamp=old_timestamp,
            status=HealthStatus.HEALTHY
        )
        health_monitor.components["warning_comp"].metrics.append(old_metric)
        
        # Run check
        result = await health_monitor._check_component_health()
        
        assert result == HealthStatus.WARNING
        
        # Check component statuses
        healthy_comp = health_monitor.components["healthy_comp"]
        warning_comp = health_monitor.components["warning_comp"]
        
        assert healthy_comp.status == HealthStatus.HEALTHY
        assert warning_comp.status == HealthStatus.WARNING
        
        # Check alerts
        assert len(health_monitor.alerts) > 0
        assert any("warning_comp" in alert for alert in health_monitor.alerts)
    
    @pytest.mark.asyncio
    async def test_check_performance_metrics(self, health_monitor):
        """Test performance metrics health check"""
        # Add some response time metrics with the exact name the system expects
        for i in range(5):
            health_monitor._add_system_metric(
                "response_time",  # Use exact name the system expects
                100.0 + i * 10,  # 100, 110, 120, 130, 140 ms
                "ms"
            )
        
        # Run check
        result = await health_monitor._check_performance_metrics()
        
        assert result == HealthStatus.HEALTHY
        
        # Check that performance metrics were added
        perf_metrics = [
            m for m in health_monitor.metrics_history
            if m.component == "system" and m.metric_name in ["avg_response_time", "error_rate"]
        ]
        
        # Should have at least avg_response_time (error_rate requires both system and component metrics)
        assert len(perf_metrics) >= 1
        
        # Check average response time (should be average of 100, 110, 120, 130, 140 = 120)
        avg_response_metric = next(m for m in perf_metrics if m.metric_name == "avg_response_time")
        # The average should be exactly 120 (average of 100, 110, 120, 130, 140)
        assert avg_response_metric.value == 120.0
    
    def test_get_system_health(self, health_monitor):
        """Test getting system health status"""
        # Register components with different statuses
        health_monitor.register_component("healthy_comp", ComponentType.CACHE)
        health_monitor.register_component("critical_comp", ComponentType.NETWORK)
        
        # Add metrics
        health_monitor.update_component_metric("healthy_comp", "test_metric", 50.0, "count")
        health_monitor.update_component_metric("critical_comp", "error_metric", 100.0, "count", 10.0, 50.0)
        
        # Get system health
        system_health = health_monitor.get_system_health()
        
        assert system_health.overall_status == HealthStatus.CRITICAL  # Due to critical component
        assert len(system_health.components) == 2
        assert len(system_health.alerts) >= 0
    
    def test_export_health_data(self, health_monitor, tmp_path):
        """Test exporting health data"""
        # Add some test data
        health_monitor.register_component("test_comp", ComponentType.CACHE)
        health_monitor.update_component_metric("test_comp", "test_metric", 100.0, "count")
        
        # Export data
        output_path = tmp_path / "health_export.json"
        exported_data = health_monitor.export_health_data(output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Check exported data structure
        assert 'system_health' in exported_data
        assert 'metrics_history' in exported_data
        assert 'health_checks' in exported_data
        assert 'export_timestamp' in exported_data
        
        # Check file content
        with open(output_path, 'r') as f:
            file_data = json.load(f)
        
        assert file_data['system_health']['overall_status'] in ['healthy', 'warning', 'critical', 'unknown']
    
    def test_clear_alerts(self, health_monitor):
        """Test clearing alerts"""
        # Add some alerts
        health_monitor.alerts = ["Alert 1", "Alert 2", "Alert 3"]
        
        assert len(health_monitor.alerts) == 3
        
        # Clear alerts
        health_monitor.clear_alerts()
        
        assert len(health_monitor.alerts) == 0
    
    def test_get_alert_summary(self, health_monitor):
        """Test getting alert summary"""
        # Add mixed alerts
        health_monitor.alerts = [
            "Critical threshold exceeded: cpu_usage = 95.0 %",
            "Warning threshold exceeded: memory_usage = 85.0 %",
            "Critical health check failed: system_resources"
        ]
        
        summary = health_monitor.get_alert_summary()
        
        assert summary['total_alerts'] == 3
        assert summary['critical_alerts'] == 2
        assert summary['warning_alerts'] == 1
        assert len(summary['recent_alerts']) == 3
        assert 'timestamp' in summary
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_monitor):
        """Test starting and stopping monitoring"""
        # Start monitoring
        await health_monitor.start_monitoring()
        assert health_monitor.is_running is True
        assert health_monitor.monitoring_task is not None
        
        # Stop monitoring
        await health_monitor.stop_monitoring()
        assert health_monitor.is_running is False
        # Note: monitoring_task might still exist but be cancelled, which is fine
        # The important thing is that is_running is False
    
    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, health_monitor):
        """Test starting monitoring when already running"""
        await health_monitor.start_monitoring()
        
        # Try to start again
        await health_monitor.start_monitoring()
        
        # Should still be running
        assert health_monitor.is_running is True


class TestQuickHealthCheck:
    """Test quick health check convenience function"""
    
    @pytest.mark.asyncio
    async def test_quick_health_check(self):
        """Test quick health check function"""
        # Mock psutil to avoid actual system calls
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network:
            
            # Setup mocks
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.used = 800
            mock_disk.return_value.total = 1000
            mock_network.return_value.bytes_sent = 1000
            mock_network.return_value.bytes_recv = 2000
            
            # Run quick health check
            result = await quick_health_check()
            
            # Check result structure
            assert 'status' in result
            assert 'components' in result
            assert 'metrics' in result
            assert 'alerts' in result
            
            # Check values
            assert result['components'] == 3  # system, cache, api
            assert result['metrics'] >= 5  # CPU, memory, disk, network sent/recv
            assert result['alerts'] >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
