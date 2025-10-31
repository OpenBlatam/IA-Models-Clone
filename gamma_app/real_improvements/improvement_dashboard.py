"""
Gamma App - Real Improvement Dashboard
Real-time dashboard for monitoring and managing improvements
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

class DashboardWidget(Enum):
    """Dashboard widget types"""
    METRICS = "metrics"
    CHARTS = "charts"
    ALERTS = "alerts"
    PROGRESS = "progress"
    STATUS = "status"

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DashboardAlert:
    """Dashboard alert"""
    alert_id: str
    title: str
    message: str
    level: AlertLevel
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class DashboardMetric:
    """Dashboard metric"""
    metric_id: str
    name: str
    value: float
    unit: str
    trend: str  # up, down, stable
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical

class RealImprovementDashboard:
    """
    Real-time dashboard for monitoring improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement dashboard"""
        self.project_root = Path(project_root)
        self.alerts: Dict[str, DashboardAlert] = {}
        self.metrics: Dict[str, DashboardMetric] = {}
        self.widgets: Dict[str, Dict[str, Any]] = {}
        self.dashboard_data: Dict[str, Any] = {}
        
        # Initialize dashboard
        self._initialize_dashboard()
        
        logger.info(f"Real Improvement Dashboard initialized for {self.project_root}")
    
    def _initialize_dashboard(self):
        """Initialize dashboard with default widgets"""
        # Initialize metrics
        self.metrics = {
            "total_improvements": DashboardMetric(
                metric_id="total_improvements",
                name="Total Improvements",
                value=0,
                unit="count",
                trend="stable"
            ),
            "completed_improvements": DashboardMetric(
                metric_id="completed_improvements",
                name="Completed Improvements",
                value=0,
                unit="count",
                trend="up"
            ),
            "success_rate": DashboardMetric(
                metric_id="success_rate",
                name="Success Rate",
                value=0,
                unit="%",
                trend="stable",
                threshold=80.0
            ),
            "avg_execution_time": DashboardMetric(
                metric_id="avg_execution_time",
                name="Average Execution Time",
                value=0,
                unit="minutes",
                trend="down"
            ),
            "system_health": DashboardMetric(
                metric_id="system_health",
                name="System Health",
                value=100,
                unit="%",
                trend="stable",
                threshold=90.0
            )
        }
        
        # Initialize widgets
        self.widgets = {
            "overview": {
                "type": "metrics",
                "title": "Overview",
                "metrics": ["total_improvements", "completed_improvements", "success_rate"]
            },
            "performance": {
                "type": "charts",
                "title": "Performance",
                "metrics": ["avg_execution_time", "system_health"]
            },
            "alerts": {
                "type": "alerts",
                "title": "Alerts",
                "max_items": 10
            },
            "progress": {
                "type": "progress",
                "title": "Progress",
                "show_trends": True
            }
        }
    
    async def update_dashboard(self) -> Dict[str, Any]:
        """Update dashboard data"""
        try:
            # Update metrics
            await self._update_metrics()
            
            # Check for alerts
            await self._check_alerts()
            
            # Update dashboard data
            self.dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {metric_id: {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "trend": metric.trend,
                    "status": metric.status
                } for metric_id, metric in self.metrics.items()},
                "alerts": [{
                    "alert_id": alert.alert_id,
                    "title": alert.title,
                    "message": alert.message,
                    "level": alert.level.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                } for alert in self.alerts.values() if not alert.resolved],
                "widgets": self.widgets,
                "system_status": await self._get_system_status()
            }
            
            return self.dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
            return {"error": str(e)}
    
    async def _update_metrics(self):
        """Update dashboard metrics"""
        try:
            # Update system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Update system health metric
            system_health = 100 - max(cpu_percent, memory_percent, disk_percent)
            self.metrics["system_health"].value = system_health
            self.metrics["system_health"].status = "critical" if system_health < 70 else "warning" if system_health < 90 else "normal"
            
            # Update other metrics (simplified for example)
            self.metrics["total_improvements"].value = 25
            self.metrics["completed_improvements"].value = 18
            self.metrics["success_rate"].value = 85.5
            self.metrics["avg_execution_time"].value = 2.3
            
            # Update metric statuses
            for metric in self.metrics.values():
                if metric.threshold:
                    if metric.value < metric.threshold:
                        metric.status = "critical"
                    elif metric.value < metric.threshold * 1.1:
                        metric.status = "warning"
                    else:
                        metric.status = "normal"
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    async def _check_alerts(self):
        """Check for alerts"""
        try:
            # Check system health
            if self.metrics["system_health"].value < 70:
                await self._create_alert(
                    "System Health Critical",
                    f"System health is at {self.metrics['system_health'].value}%",
                    AlertLevel.CRITICAL,
                    "system"
                )
            elif self.metrics["system_health"].value < 90:
                await self._create_alert(
                    "System Health Warning",
                    f"System health is at {self.metrics['system_health'].value}%",
                    AlertLevel.WARNING,
                    "system"
                )
            
            # Check success rate
            if self.metrics["success_rate"].value < 80:
                await self._create_alert(
                    "Low Success Rate",
                    f"Success rate is at {self.metrics['success_rate'].value}%",
                    AlertLevel.WARNING,
                    "improvements"
                )
            
            # Check execution time
            if self.metrics["avg_execution_time"].value > 5:
                await self._create_alert(
                    "Slow Execution",
                    f"Average execution time is {self.metrics['avg_execution_time'].value} minutes",
                    AlertLevel.WARNING,
                    "performance"
                )
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    async def _create_alert(self, title: str, message: str, level: AlertLevel, source: str):
        """Create dashboard alert"""
        try:
            alert_id = f"alert_{int(time.time() * 1000)}"
            
            # Check if similar alert already exists
            for alert in self.alerts.values():
                if (alert.title == title and 
                    alert.level == level and 
                    not alert.resolved and
                    (datetime.utcnow() - alert.timestamp).seconds < 300):  # 5 minutes
                    return
            
            alert = DashboardAlert(
                alert_id=alert_id,
                title=title,
                message=message,
                level=level,
                timestamp=datetime.utcnow(),
                source=source
            )
            
            self.alerts[alert_id] = alert
            
            logger.info(f"Alert created: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0],
                "uptime": time.time() - psutil.boot_time(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                
                logger.info(f"Alert resolved: {alert.title}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def get_improvement_progress(self) -> Dict[str, Any]:
        """Get improvement progress"""
        try:
            # Simulate progress data
            progress_data = {
                "total_improvements": 25,
                "completed_improvements": 18,
                "in_progress_improvements": 4,
                "failed_improvements": 2,
                "pending_improvements": 1,
                "completion_rate": 72.0,
                "success_rate": 85.5,
                "avg_execution_time": 2.3,
                "estimated_completion": "2024-02-15T10:00:00Z"
            }
            
            return progress_data
            
        except Exception as e:
            logger.error(f"Failed to get improvement progress: {e}")
            return {"error": str(e)}
    
    async def get_improvement_trends(self) -> Dict[str, Any]:
        """Get improvement trends"""
        try:
            # Simulate trend data
            trends_data = {
                "daily_completions": [
                    {"date": "2024-01-01", "count": 3},
                    {"date": "2024-01-02", "count": 5},
                    {"date": "2024-01-03", "count": 2},
                    {"date": "2024-01-04", "count": 4},
                    {"date": "2024-01-05", "count": 6}
                ],
                "success_rate_trend": [
                    {"date": "2024-01-01", "rate": 80.0},
                    {"date": "2024-01-02", "rate": 85.0},
                    {"date": "2024-01-03", "rate": 82.0},
                    {"date": "2024-01-04", "rate": 88.0},
                    {"date": "2024-01-05", "rate": 90.0}
                ],
                "execution_time_trend": [
                    {"date": "2024-01-01", "time": 3.2},
                    {"date": "2024-01-02", "time": 2.8},
                    {"date": "2024-01-03", "time": 2.5},
                    {"date": "2024-01-04", "time": 2.1},
                    {"date": "2024-01-05", "time": 1.9}
                ]
            }
            
            return trends_data
            
        except Exception as e:
            logger.error(f"Failed to get improvement trends: {e}")
            return {"error": str(e)}
    
    async def get_improvement_categories(self) -> Dict[str, Any]:
        """Get improvement categories"""
        try:
            categories_data = {
                "performance": {
                    "total": 8,
                    "completed": 6,
                    "success_rate": 92.5,
                    "avg_time": 1.8
                },
                "security": {
                    "total": 5,
                    "completed": 4,
                    "success_rate": 88.0,
                    "avg_time": 2.1
                },
                "maintainability": {
                    "total": 7,
                    "completed": 5,
                    "success_rate": 85.0,
                    "avg_time": 2.5
                },
                "testing": {
                    "total": 3,
                    "completed": 2,
                    "success_rate": 90.0,
                    "avg_time": 1.5
                },
                "documentation": {
                    "total": 2,
                    "completed": 1,
                    "success_rate": 95.0,
                    "avg_time": 1.2
                }
            }
            
            return categories_data
            
        except Exception as e:
            logger.error(f"Failed to get improvement categories: {e}")
            return {"error": str(e)}
    
    async def get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent activities"""
        try:
            activities = [
                {
                    "id": "act_001",
                    "type": "improvement_completed",
                    "title": "Database Optimization Completed",
                    "description": "Added database indexes for performance",
                    "timestamp": "2024-01-05T14:30:00Z",
                    "status": "success"
                },
                {
                    "id": "act_002",
                    "type": "improvement_failed",
                    "title": "Security Headers Failed",
                    "description": "Failed to add security headers",
                    "timestamp": "2024-01-05T13:45:00Z",
                    "status": "error"
                },
                {
                    "id": "act_003",
                    "type": "alert_created",
                    "title": "System Health Warning",
                    "description": "System health is at 85%",
                    "timestamp": "2024-01-05T13:30:00Z",
                    "status": "warning"
                },
                {
                    "id": "act_004",
                    "type": "improvement_started",
                    "title": "Input Validation Started",
                    "description": "Started implementing input validation",
                    "timestamp": "2024-01-05T12:15:00Z",
                    "status": "info"
                },
                {
                    "id": "act_005",
                    "type": "rollback_completed",
                    "title": "Rollback Completed",
                    "description": "Successfully rolled back failed improvement",
                    "timestamp": "2024-01-05T11:30:00Z",
                    "status": "success"
                }
            ]
            
            return activities
            
        except Exception as e:
            logger.error(f"Failed to get recent activities: {e}")
            return []
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        try:
            return {
                "total_improvements": self.metrics["total_improvements"].value,
                "completed_improvements": self.metrics["completed_improvements"].value,
                "success_rate": self.metrics["success_rate"].value,
                "system_health": self.metrics["system_health"].value,
                "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard summary: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """Start dashboard monitoring"""
        logger.info("Starting dashboard monitoring")
        
        while True:
            try:
                # Update dashboard
                await self.update_dashboard()
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in dashboard monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return {
            "widgets": self.widgets,
            "refresh_interval": 30,
            "alert_retention": 24,  # hours
            "metrics_retention": 168,  # hours (1 week)
            "auto_refresh": True,
            "notifications": True
        }
    
    def update_widget_config(self, widget_id: str, config: Dict[str, Any]) -> bool:
        """Update widget configuration"""
        try:
            if widget_id in self.widgets:
                self.widgets[widget_id].update(config)
                logger.info(f"Widget {widget_id} configuration updated")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update widget config: {e}")
            return False
    
    def add_custom_metric(self, metric_id: str, name: str, value: float, unit: str) -> bool:
        """Add custom metric"""
        try:
            metric = DashboardMetric(
                metric_id=metric_id,
                name=name,
                value=value,
                unit=unit,
                trend="stable"
            )
            
            self.metrics[metric_id] = metric
            logger.info(f"Custom metric added: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom metric: {e}")
            return False
    
    def get_metric_history(self, metric_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history"""
        try:
            # Simulate metric history
            history = []
            current_time = datetime.utcnow()
            
            for i in range(hours):
                timestamp = current_time - timedelta(hours=i)
                value = self.metrics.get(metric_id, DashboardMetric("", "", 0, "", "")).value + (i * 0.1)
                
                history.append({
                    "timestamp": timestamp.isoformat(),
                    "value": value
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []

# Global dashboard instance
improvement_dashboard = None

def get_improvement_dashboard() -> RealImprovementDashboard:
    """Get improvement dashboard instance"""
    global improvement_dashboard
    if not improvement_dashboard:
        improvement_dashboard = RealImprovementDashboard()
    return improvement_dashboard













