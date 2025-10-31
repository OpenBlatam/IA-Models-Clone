"""
Monitoring Service
=================

Advanced monitoring and observability for the professional documents system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import json
import psutil
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric type."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """Alert level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data point."""
    metric_id: str
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str]
    timestamp: datetime
    unit: str = ""


@dataclass
class Alert:
    """Alert data."""
    alert_id: str
    name: str
    description: str
    level: AlertLevel
    metric_name: str
    threshold: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class HealthCheck:
    """Health check result."""
    service_name: str
    status: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = None
    error_message: Optional[str] = None


class SystemMetricsCollector:
    """System metrics collector."""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
    
    async def collect_system_metrics(self) -> List[Metric]:
        """Collect system metrics."""
        
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.cpu.usage",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost"},
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.memory.usage",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost"},
                timestamp=timestamp,
                unit="percent"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.memory.available",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost"},
                timestamp=timestamp,
                unit="bytes"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.disk.usage",
                value=disk.percent,
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost", "mount": "/"},
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.network.bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricType.COUNTER,
                tags={"host": "localhost"},
                timestamp=timestamp,
                unit="bytes"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.network.bytes_recv",
                value=network.bytes_recv,
                metric_type=MetricType.COUNTER,
                tags={"host": "localhost"},
                timestamp=timestamp,
                unit="bytes"
            ))
            
            # Process metrics
            process = psutil.Process()
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.process.cpu_percent",
                value=process.cpu_percent(),
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost", "pid": str(process.pid)},
                timestamp=timestamp,
                unit="percent"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="system.process.memory_usage",
                value=process.memory_info().rss,
                metric_type=MetricType.GAUGE,
                tags={"host": "localhost", "pid": str(process.pid)},
                timestamp=timestamp,
                unit="bytes"
            ))
            
            # Store metrics
            for metric in metrics:
                self.metrics_history[metric.name].append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return []
    
    async def collect_application_metrics(self) -> List[Metric]:
        """Collect application-specific metrics."""
        
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Document metrics (mock data - in production, get from actual services)
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="documents.total_count",
                value=150,  # Mock value
                metric_type=MetricType.GAUGE,
                tags={"service": "documents"},
                timestamp=timestamp,
                unit="count"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="documents.active_sessions",
                value=25,  # Mock value
                metric_type=MetricType.GAUGE,
                tags={"service": "collaboration"},
                timestamp=timestamp,
                unit="count"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="documents.versions_created",
                value=5,  # Mock value
                metric_type=MetricType.COUNTER,
                tags={"service": "version_control"},
                timestamp=timestamp,
                unit="count"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="documents.analyses_completed",
                value=12,  # Mock value
                metric_type=MetricType.COUNTER,
                tags={"service": "ai_insights"},
                timestamp=timestamp,
                unit="count"
            ))
            
            # API metrics
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="api.requests_per_second",
                value=45.5,  # Mock value
                metric_type=MetricType.GAUGE,
                tags={"service": "api"},
                timestamp=timestamp,
                unit="requests/second"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="api.response_time",
                value=125.3,  # Mock value
                metric_type=MetricType.HISTOGRAM,
                tags={"service": "api"},
                timestamp=timestamp,
                unit="milliseconds"
            ))
            
            metrics.append(Metric(
                metric_id=str(uuid4()),
                name="api.error_rate",
                value=0.02,  # Mock value
                metric_type=MetricType.GAUGE,
                tags={"service": "api"},
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Store metrics
            for metric in metrics:
                self.metrics_history[metric.name].append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
            return []
    
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        condition: str = "greater_than",
        level: AlertLevel = AlertLevel.WARNING
    ):
        """Add alert rule."""
        
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,
            "level": level,
            "enabled": True
        }
    
    async def check_alerts(self) -> List[Alert]:
        """Check for alert conditions."""
        
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
            
            metric_name = rule["metric_name"]
            if metric_name not in self.metrics_history:
                continue
            
            # Get latest metric value
            metrics = list(self.metrics_history[metric_name])
            if not metrics:
                continue
            
            latest_metric = metrics[-1]
            current_value = latest_metric.value
            threshold = rule["threshold"]
            condition = rule["condition"]
            
            # Check condition
            should_alert = False
            if condition == "greater_than" and current_value > threshold:
                should_alert = True
            elif condition == "less_than" and current_value < threshold:
                should_alert = True
            elif condition == "equals" and current_value == threshold:
                should_alert = True
            
            if should_alert:
                # Check if alert already exists
                alert_key = f"{rule_name}_{metric_name}"
                if alert_key not in self.active_alerts:
                    alert = Alert(
                        alert_id=str(uuid4()),
                        name=rule_name,
                        description=f"{metric_name} {condition} {threshold} (current: {current_value})",
                        level=rule["level"],
                        metric_name=metric_name,
                        threshold=threshold,
                        current_value=current_value,
                        triggered_at=datetime.now()
                    )
                    
                    self.active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    
                    logger.warning(f"Alert triggered: {rule_name} - {alert.description}")
        
        return new_alerts
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        
        for key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved_at = datetime.now()
                alert.is_active = False
                del self.active_alerts[key]
                logger.info(f"Alert resolved: {alert.name}")
                return True
        
        return False
    
    def get_metric_history(
        self,
        metric_name: str,
        time_range: timedelta = timedelta(hours=1)
    ) -> List[Metric]:
        """Get metric history for a time range."""
        
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - time_range
        return [
            metric for metric in self.metrics_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get metric summary statistics."""
        
        if metric_name not in self.metrics_history:
            return {}
        
        metrics = list(self.metrics_history[metric_name])
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "metric_name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "first_seen": metrics[0].timestamp.isoformat(),
            "last_seen": metrics[-1].timestamp.isoformat()
        }


class HealthChecker:
    """Health checker for services."""
    
    def __init__(self):
        self.health_checks: Dict[str, callable] = {}
        self.health_history: Dict[str, List[HealthCheck]] = defaultdict(list)
    
    def register_health_check(self, service_name: str, check_function: callable):
        """Register health check function."""
        
        self.health_checks[service_name] = check_function
    
    async def run_health_checks(self) -> List[HealthCheck]:
        """Run all health checks."""
        
        results = []
        
        for service_name, check_function in self.health_checks.items():
            start_time = time.time()
            
            try:
                # Run health check
                health_data = await check_function()
                
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                health_check = HealthCheck(
                    service_name=service_name,
                    status="healthy",
                    response_time=response_time,
                    timestamp=datetime.now(),
                    details=health_data
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                
                health_check = HealthCheck(
                    service_name=service_name,
                    status="unhealthy",
                    response_time=response_time,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
            
            results.append(health_check)
            self.health_history[service_name].append(health_check)
            
            # Keep only last 100 health checks per service
            if len(self.health_history[service_name]) > 100:
                self.health_history[service_name] = self.health_history[service_name][-100:]
        
        return results
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for a service."""
        
        if service_name not in self.health_history:
            return {"status": "unknown", "message": "Service not monitored"}
        
        health_checks = self.health_history[service_name]
        if not health_checks:
            return {"status": "unknown", "message": "No health checks available"}
        
        latest_check = health_checks[-1]
        
        # Calculate uptime
        healthy_checks = [h for h in health_checks if h.status == "healthy"]
        uptime_percent = (len(healthy_checks) / len(health_checks)) * 100
        
        return {
            "service_name": service_name,
            "status": latest_check.status,
            "response_time": latest_check.response_time,
            "last_check": latest_check.timestamp.isoformat(),
            "uptime_percent": uptime_percent,
            "total_checks": len(health_checks),
            "healthy_checks": len(healthy_checks),
            "error_message": latest_check.error_message,
            "details": latest_check.details
        }


class MonitoringService:
    """Main monitoring service."""
    
    def __init__(self):
        self.metrics_collector = SystemMetricsCollector()
        self.health_checker = HealthChecker()
        self.monitoring_active = False
        self.collection_interval = 30  # seconds
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        
        self.metrics_collector.add_alert_rule(
            name="High CPU Usage",
            metric_name="system.cpu.usage",
            threshold=80.0,
            condition="greater_than",
            level=AlertLevel.WARNING
        )
        
        self.metrics_collector.add_alert_rule(
            name="High Memory Usage",
            metric_name="system.memory.usage",
            threshold=85.0,
            condition="greater_than",
            level=AlertLevel.WARNING
        )
        
        self.metrics_collector.add_alert_rule(
            name="High Disk Usage",
            metric_name="system.disk.usage",
            threshold=90.0,
            condition="greater_than",
            level=AlertLevel.CRITICAL
        )
        
        self.metrics_collector.add_alert_rule(
            name="High API Error Rate",
            metric_name="api.error_rate",
            threshold=5.0,
            condition="greater_than",
            level=AlertLevel.ERROR
        )
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        async def documents_service_health():
            # Mock health check for documents service
            return {
                "documents_count": 150,
                "active_sessions": 25,
                "last_activity": datetime.now().isoformat()
            }
        
        async def collaboration_service_health():
            # Mock health check for collaboration service
            return {
                "active_connections": 15,
                "rooms_count": 8,
                "last_message": datetime.now().isoformat()
            }
        
        async def ai_service_health():
            # Mock health check for AI service
            return {
                "models_loaded": 3,
                "queue_size": 2,
                "last_analysis": datetime.now().isoformat()
            }
        
        self.health_checker.register_health_check("documents", documents_service_health)
        self.health_checker.register_health_check("collaboration", collaboration_service_health)
        self.health_checker.register_health_check("ai_insights", ai_service_health)
    
    async def start_monitoring(self):
        """Start monitoring services."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting monitoring services")
        
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop monitoring services."""
        
        self.monitoring_active = False
        logger.info("Stopping monitoring services")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect metrics
                await self.metrics_collector.collect_system_metrics()
                await self.metrics_collector.collect_application_metrics()
                
                # Check alerts
                new_alerts = await self.metrics_collector.check_alerts()
                if new_alerts:
                    await self._handle_alerts(new_alerts)
                
                # Run health checks
                health_results = await self.health_checker.run_health_checks()
                await self._handle_health_results(health_results)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.collection_interval)
    
    async def _handle_alerts(self, alerts: List[Alert]):
        """Handle new alerts."""
        
        for alert in alerts:
            # In production, this would send notifications, log to external systems, etc.
            logger.warning(f"ALERT: {alert.name} - {alert.description}")
            
            # Could integrate with notification services here
            # await notification_service.send_alert(alert)
    
    async def _handle_health_results(self, health_results: List[HealthCheck]):
        """Handle health check results."""
        
        for health_check in health_results:
            if health_check.status != "healthy":
                logger.warning(f"HEALTH CHECK FAILED: {health_check.service_name} - {health_check.error_message}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        
        # Get current metrics
        system_metrics = await self.metrics_collector.collect_system_metrics()
        app_metrics = await self.metrics_collector.collect_application_metrics()
        
        # Get health status
        health_results = await self.health_checker.run_health_checks()
        
        # Get active alerts
        active_alerts = list(self.metrics_collector.active_alerts.values())
        
        # Get metric summaries
        metric_summaries = {}
        for metric_name in ["system.cpu.usage", "system.memory.usage", "api.requests_per_second"]:
            metric_summaries[metric_name] = self.metrics_collector.get_metric_summary(metric_name)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in system_metrics
            ],
            "application_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in app_metrics
            ],
            "health_checks": [
                {
                    "service_name": h.service_name,
                    "status": h.status,
                    "response_time": h.response_time,
                    "timestamp": h.timestamp.isoformat(),
                    "error_message": h.error_message
                }
                for h in health_results
            ],
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "name": a.name,
                    "level": a.level.value,
                    "description": a.description,
                    "triggered_at": a.triggered_at.isoformat()
                }
                for a in active_alerts
            ],
            "metric_summaries": metric_summaries
        }
    
    async def get_metric_data(
        self,
        metric_name: str,
        time_range: str = "1h"
    ) -> Dict[str, Any]:
        """Get metric data for a specific metric."""
        
        # Parse time range
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        delta = time_ranges.get(time_range, timedelta(hours=1))
        metrics = self.metrics_collector.get_metric_history(metric_name, delta)
        
        return {
            "metric_name": metric_name,
            "time_range": time_range,
            "data_points": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value
                }
                for m in metrics
            ],
            "summary": self.metrics_collector.get_metric_summary(metric_name)
        }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        
        services = ["documents", "collaboration", "ai_insights", "version_control", "security"]
        
        status = {}
        for service in services:
            status[service] = self.health_checker.get_service_health(service)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": status,
            "overall_status": "healthy" if all(
                s["status"] == "healthy" for s in status.values()
            ) else "degraded"
        }



























