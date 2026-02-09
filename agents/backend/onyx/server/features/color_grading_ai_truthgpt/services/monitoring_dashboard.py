"""
Monitoring Dashboard for Color Grading AI
==========================================

Real-time monitoring and dashboard for system health and performance.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Dashboard metric."""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """System health status."""
    overall: str  # healthy, degraded, unhealthy
    services: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard.
    
    Features:
    - Real-time metrics
    - System health monitoring
    - Service status tracking
    - Resource usage monitoring
    - Alert management
    - Historical data
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize monitoring dashboard.
        
        Args:
            history_size: Size of metric history
        """
        self.history_size = history_size
        self._metrics: Dict[str, deque] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._service_status: Dict[str, str] = {}
        self._resource_usage: Dict[str, float] = {}
    
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
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.history_size)
        
        metric = DashboardMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self._metrics[name].append(metric)
        logger.debug(f"Recorded metric: {name}={value}{unit}")
    
    def get_metric(
        self,
        name: str,
        window_minutes: Optional[int] = None
    ) -> Optional[DashboardMetric]:
        """
        Get latest metric value.
        
        Args:
            name: Metric name
            window_minutes: Optional time window
            
        Returns:
            Latest metric or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self._metrics[name]
                if m.timestamp > cutoff
            ]
            if recent_metrics:
                return recent_metrics[-1]
            return None
        
        return self._metrics[name][-1]
    
    def get_metric_history(
        self,
        name: str,
        limit: int = 100
    ) -> List[DashboardMetric]:
        """
        Get metric history.
        
        Args:
            name: Metric name
            limit: Result limit
            
        Returns:
            List of metrics
        """
        if name not in self._metrics:
            return []
        
        return list(self._metrics[name])[-limit:]
    
    def update_service_status(self, service_name: str, status: str):
        """
        Update service status.
        
        Args:
            service_name: Service name
            status: Status (healthy, degraded, unhealthy)
        """
        self._service_status[service_name] = status
        logger.debug(f"Service {service_name} status: {status}")
    
    def update_resource_usage(
        self,
        resource_name: str,
        usage: float
    ):
        """
        Update resource usage.
        
        Args:
            resource_name: Resource name
            usage: Usage percentage (0-100)
        """
        self._resource_usage[resource_name] = usage
    
    def add_alert(
        self,
        severity: str,
        message: str,
        service: Optional[str] = None
    ):
        """
        Add alert.
        
        Args:
            severity: Alert severity (info, warning, error, critical)
            message: Alert message
            service: Optional service name
        """
        alert = {
            "severity": severity,
            "message": message,
            "service": service,
            "timestamp": datetime.now().isoformat()
        }
        self._alerts.append(alert)
        logger.warning(f"Alert [{severity}]: {message}")
    
    def get_system_health(self) -> SystemHealth:
        """
        Get overall system health.
        
        Returns:
            System health status
        """
        # Determine overall health
        unhealthy_services = [
            name for name, status in self._service_status.items()
            if status == "unhealthy"
        ]
        degraded_services = [
            name for name, status in self._service_status.items()
            if status == "degraded"
        ]
        
        if unhealthy_services:
            overall = "unhealthy"
        elif degraded_services:
            overall = "degraded"
        else:
            overall = "healthy"
        
        # Collect alerts
        recent_alerts = [
            alert["message"] for alert in self._alerts[-10:]
            if alert["severity"] in ["error", "critical"]
        ]
        
        return SystemHealth(
            overall=overall,
            services=self._service_status.copy(),
            resources=self._resource_usage.copy(),
            alerts=recent_alerts
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        health = self.get_system_health()
        
        # Get latest metrics
        latest_metrics = {
            name: {
                "value": metrics[-1].value if metrics else 0.0,
                "unit": metrics[-1].unit if metrics else "",
                "timestamp": metrics[-1].timestamp.isoformat() if metrics else None
            }
            for name, metrics in self._metrics.items()
        }
        
        return {
            "health": {
                "overall": health.overall,
                "services": health.services,
                "resources": health.resources,
            },
            "metrics": latest_metrics,
            "alerts": self._alerts[-20:],  # Last 20 alerts
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "total_metrics": len(self._metrics),
            "total_alerts": len(self._alerts),
            "monitored_services": len(self._service_status),
            "monitored_resources": len(self._resource_usage),
        }


