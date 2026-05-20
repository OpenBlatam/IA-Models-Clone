"""
Unified Monitoring System for Color Grading AI
==============================================

Consolidates monitoring services:
- HealthMonitor (health monitoring)
- MonitoringDashboard (monitoring dashboard)
- PerformanceMonitor (performance monitoring)
- UnifiedPerformanceSystem (unified performance)

Features:
- Unified monitoring interface
- Health status tracking
- Performance monitoring
- Real-time dashboard
- Alert management
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .health_monitor import HealthMonitor, HealthStatus, HealthCheck
from .monitoring_dashboard import MonitoringDashboard, SystemHealth, DashboardMetric
from .performance_monitor import PerformanceMonitor, PerformanceMetric as MonitorMetric
from .unified_performance_system import UnifiedPerformanceSystem, ProfilerMode

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """Monitoring modes."""
    BASIC = "basic"  # Basic monitoring
    STANDARD = "standard"  # Standard monitoring
    DETAILED = "detailed"  # Detailed monitoring
    FULL = "full"  # Full monitoring with profiling


@dataclass
class UnifiedMonitoringReport:
    """Unified monitoring report."""
    timestamp: datetime
    system_health: SystemHealth
    performance_metrics: Dict[str, Any]
    dashboard_data: Dict[str, Any]
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class UnifiedMonitoringSystem:
    """
    Unified monitoring system.
    
    Consolidates:
    - HealthMonitor: Health monitoring
    - MonitoringDashboard: Monitoring dashboard
    - PerformanceMonitor: Performance monitoring
    - UnifiedPerformanceSystem: Unified performance
    
    Features:
    - Unified monitoring interface
    - Health status tracking
    - Performance monitoring
    - Real-time dashboard
    """
    
    def __init__(
        self,
        monitoring_mode: MonitoringMode = MonitoringMode.STANDARD,
        profiler_mode: ProfilerMode = ProfilerMode.SIMPLE
    ):
        """
        Initialize unified monitoring system.
        
        Args:
            monitoring_mode: Monitoring mode
            profiler_mode: Profiler mode
        """
        self.monitoring_mode = monitoring_mode
        
        # Initialize components
        self.health_monitor = HealthMonitor()
        self.monitoring_dashboard = MonitoringDashboard()
        self.performance_monitor = PerformanceMonitor()
        self.unified_performance_system = UnifiedPerformanceSystem(
            profiler_mode=profiler_mode
        )
        
        logger.info(f"Initialized UnifiedMonitoringSystem (mode={monitoring_mode.value})")
    
    def check_health(self, service_name: Optional[str] = None) -> HealthStatus:
        """
        Check health status.
        
        Args:
            service_name: Optional service name
            
        Returns:
            Health status
        """
        if service_name:
            return self.health_monitor.check_service(service_name)
        else:
            return self.health_monitor.get_overall_health()
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            metadata: Optional metadata
        """
        # Record in dashboard
        self.monitoring_dashboard.record_metric(name, value, unit, metadata)
        
        # Record in performance monitor
        self.performance_monitor.record_metric(
            operation=name,
            duration=value,
            success=True
        )
    
    def record_performance(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ):
        """
        Record performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
        """
        # Record in performance monitor
        self.performance_monitor.record_metric(operation, duration, success)
        
        # Record in unified performance system
        self.unified_performance_system.record_metric(
            operation=operation,
            duration=duration,
            success=success
        )
        
        # Record in dashboard
        self.monitoring_dashboard.record_metric(
            name=f"operation.{operation}",
            value=duration,
            unit="seconds"
        )
    
    def update_service_status(self, service_name: str, status: str):
        """
        Update service status.
        
        Args:
            service_name: Service name
            status: Status
        """
        # Update in dashboard
        self.monitoring_dashboard.update_service_status(service_name, status)
        
        # Update in health monitor
        health_status = HealthStatus.HEALTHY
        if status == "unhealthy":
            health_status = HealthStatus.UNHEALTHY
        elif status == "degraded":
            health_status = HealthStatus.DEGRADED
        
        self.health_monitor.update_service_status(service_name, health_status)
    
    def add_alert(
        self,
        severity: str,
        message: str,
        service: Optional[str] = None
    ):
        """
        Add alert.
        
        Args:
            severity: Alert severity
            message: Alert message
            service: Optional service name
        """
        self.monitoring_dashboard.add_alert(severity, message, service)
    
    def generate_report(self) -> UnifiedMonitoringReport:
        """
        Generate unified monitoring report.
        
        Returns:
            Unified monitoring report
        """
        # Get system health
        system_health = self.monitoring_dashboard.get_system_health()
        
        # Get performance metrics
        performance_metrics = self.unified_performance_system.get_all_stats()
        
        # Get dashboard data
        dashboard_data = self.monitoring_dashboard.get_dashboard_data()
        
        # Collect alerts
        alerts = dashboard_data.get("alerts", [])
        
        # Generate recommendations
        recommendations = []
        if system_health.overall != "healthy":
            recommendations.append(f"System health is {system_health.overall}. Review service statuses.")
        
        if alerts:
            critical_alerts = [a for a in alerts if a.get("severity") in ["error", "critical"]]
            if critical_alerts:
                recommendations.append(f"{len(critical_alerts)} critical alerts require attention.")
        
        return UnifiedMonitoringReport(
            timestamp=datetime.now(),
            system_health=system_health,
            performance_metrics=performance_metrics,
            dashboard_data=dashboard_data,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "monitoring_mode": self.monitoring_mode.value,
            "health_monitor": {
                "services_monitored": len(self.health_monitor._services),
            },
            "dashboard": self.monitoring_dashboard.get_statistics(),
            "performance": self.unified_performance_system.get_statistics(),
        }


