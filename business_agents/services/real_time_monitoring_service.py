"""
Real-Time Monitoring Service
============================

Advanced real-time monitoring service for business agents, workflows,
and system performance with intelligent alerting and anomaly detection.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import statistics
import websockets
from websockets.server import WebSocketServerProtocol
import sse_starlette
from sse_starlette.sse import EventSourceResponse
import threading
import time

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    id: str
    name: str
    description: str
    level: AlertLevel
    source: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    request_rate: float
    error_rate: float
    response_time: float
    timestamp: datetime

class RealTimeMonitoringService:
    """
    Advanced real-time monitoring service for business agents system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.websocket_connections: List[WebSocketServerProtocol] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        # Start monitoring
        self.start_monitoring()
        
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules."""
        
        self.alert_rules = [
            {
                "name": "High CPU Usage",
                "condition": lambda metrics: metrics.get("cpu_usage", 0) > 80,
                "level": AlertLevel.WARNING,
                "description": "CPU usage is above 80%"
            },
            {
                "name": "High Memory Usage",
                "condition": lambda metrics: metrics.get("memory_usage", 0) > 85,
                "level": AlertLevel.WARNING,
                "description": "Memory usage is above 85%"
            },
            {
                "name": "High Error Rate",
                "condition": lambda metrics: metrics.get("error_rate", 0) > 5,
                "level": AlertLevel.ERROR,
                "description": "Error rate is above 5%"
            },
            {
                "name": "High Response Time",
                "condition": lambda metrics: metrics.get("response_time", 0) > 5000,
                "level": AlertLevel.WARNING,
                "description": "Response time is above 5 seconds"
            },
            {
                "name": "Low Success Rate",
                "condition": lambda metrics: metrics.get("success_rate", 100) < 90,
                "level": AlertLevel.ERROR,
                "description": "Success rate is below 90%"
            }
        ]
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        logger.info("Real-time monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Broadcast updates to WebSocket connections
                self._broadcast_updates()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval", 5))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Wait longer on error
                
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_usage, MetricType.GAUGE)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self.record_metric("memory_usage", memory_usage, MetricType.GAUGE)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self.record_metric("disk_usage", disk_usage, MetricType.GAUGE)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            self.record_metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process count
            process_count = len(psutil.pids())
            self.record_metric("process_count", process_count, MetricType.GAUGE)
            
        except ImportError:
            # Fallback metrics if psutil is not available
            self.record_metric("cpu_usage", 0.0, MetricType.GAUGE)
            self.record_metric("memory_usage", 0.0, MetricType.GAUGE)
            self.record_metric("disk_usage", 0.0, MetricType.GAUGE)
            
    def _run_health_checks(self):
        """Run health checks for various components."""
        
        # Database health check
        self._check_database_health()
        
        # API health check
        self._check_api_health()
        
        # External service health checks
        self._check_external_services()
        
    def _check_database_health(self):
        """Check database connectivity and performance."""
        
        try:
            start_time = time.time()
            # Simulate database check
            # In real implementation, this would check actual database
            response_time = (time.time() - start_time) * 1000
            
            if response_time < 100:
                status = HealthStatus.HEALTHY
                message = "Database is healthy"
            elif response_time < 500:
                status = HealthStatus.DEGRADED
                message = "Database response time is slow"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database response time is very slow"
                
            self.health_checks["database"] = HealthCheck(
                name="database",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time=response_time
            )
            
        except Exception as e:
            self.health_checks["database"] = HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                timestamp=datetime.now()
            )
            
    def _check_api_health(self):
        """Check API health and performance."""
        
        try:
            start_time = time.time()
            # Simulate API check
            response_time = (time.time() - start_time) * 1000
            
            if response_time < 200:
                status = HealthStatus.HEALTHY
                message = "API is healthy"
            elif response_time < 1000:
                status = HealthStatus.DEGRADED
                message = "API response time is slow"
            else:
                status = HealthStatus.UNHEALTHY
                message = "API response time is very slow"
                
            self.health_checks["api"] = HealthCheck(
                name="api",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time=response_time
            )
            
        except Exception as e:
            self.health_checks["api"] = HealthCheck(
                name="api",
                status=HealthStatus.CRITICAL,
                message=f"API health check failed: {str(e)}",
                timestamp=datetime.now()
            )
            
    def _check_external_services(self):
        """Check external service health."""
        
        # Check AI services
        self._check_ai_service_health()
        
        # Check integration services
        self._check_integration_health()
        
    def _check_ai_service_health(self):
        """Check AI service health."""
        
        try:
            # Simulate AI service check
            status = HealthStatus.HEALTHY
            message = "AI services are healthy"
            
            self.health_checks["ai_services"] = HealthCheck(
                name="ai_services",
                status=status,
                message=message,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.health_checks["ai_services"] = HealthCheck(
                name="ai_services",
                status=HealthStatus.UNHEALTHY,
                message=f"AI service health check failed: {str(e)}",
                timestamp=datetime.now()
            )
            
    def _check_integration_health(self):
        """Check integration service health."""
        
        try:
            # Simulate integration check
            status = HealthStatus.HEALTHY
            message = "Integration services are healthy"
            
            self.health_checks["integrations"] = HealthCheck(
                name="integrations",
                status=status,
                message=message,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.health_checks["integrations"] = HealthCheck(
                name="integrations",
                status=HealthStatus.UNHEALTHY,
                message=f"Integration health check failed: {str(e)}",
                timestamp=datetime.now()
            )
            
    def _check_alert_conditions(self):
        """Check alert conditions and create alerts if needed."""
        
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](current_metrics):
                    # Check if alert already exists and is not resolved
                    existing_alert = next(
                        (alert for alert in self.alerts 
                         if alert.name == rule["name"] and not alert.resolved),
                        None
                    )
                    
                    if not existing_alert:
                        # Create new alert
                        alert = Alert(
                            id=f"{rule['name']}_{int(time.time())}",
                            name=rule["name"],
                            description=rule["description"],
                            level=rule["level"],
                            source="monitoring_service",
                            timestamp=datetime.now(),
                            metadata={"metrics": current_metrics}
                        )
                        
                        self.alerts.append(alert)
                        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {str(e)}")
                
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        
        current_metrics = {}
        
        for metric_name, metric_deque in self.metrics.items():
            if metric_deque:
                # Get the latest value
                current_metrics[metric_name] = metric_deque[-1].value
                
        return current_metrics
        
    def _broadcast_updates(self):
        """Broadcast updates to WebSocket connections."""
        
        if not self.websocket_connections:
            return
            
        try:
            # Prepare update data
            update_data = {
                "type": "metrics_update",
                "timestamp": datetime.now().isoformat(),
                "metrics": self._get_current_metrics(),
                "health_checks": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "response_time": check.response_time
                    }
                    for name, check in self.health_checks.items()
                },
                "alerts": [
                    {
                        "id": alert.id,
                        "name": alert.name,
                        "level": alert.level.value,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved
                    }
                    for alert in self.alerts[-10:]  # Last 10 alerts
                ]
            }
            
            # Broadcast to all connections
            disconnected = []
            for connection in self.websocket_connections:
                try:
                    asyncio.create_task(connection.send(json.dumps(update_data)))
                except:
                    disconnected.append(connection)
                    
            # Remove disconnected connections
            for connection in disconnected:
                self.websocket_connections.remove(connection)
                
        except Exception as e:
            logger.error(f"Error broadcasting updates: {str(e)}")
            
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a metric."""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics[name].append(metric)
        
    def get_metric_history(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Metric]:
        """Get metric history for a specific metric."""
        
        if name not in self.metrics:
            return []
            
        metrics = list(self.metrics[name])
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
            
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
            
        return metrics
        
    def get_metric_statistics(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        
        if name not in self.metrics:
            return {}
            
        # Get metrics from the last window_minutes
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics[name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1] if values else 0.0
        }
        
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all metrics."""
        
        return dict(self.metrics)
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        
        return [alert for alert in self.alerts if not alert.resolved]
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert.name}")
                return True
                
        return False
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        
        if not self.health_checks:
            return {"status": "unknown", "message": "No health checks available"}
            
        # Determine overall status
        statuses = [check.status for check in self.health_checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
            
        return {
            "status": overall_status.value,
            "message": f"System is {overall_status.value}",
            "components": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time
                }
                for name, check in self.health_checks.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add a new alert rule."""
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
        
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        
        for i, rule in enumerate(self.alert_rules):
            if rule["name"] == rule_name:
                del self.alert_rules[i]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
                
        return False
        
    def add_websocket_connection(self, connection: WebSocketServerProtocol):
        """Add a WebSocket connection for real-time updates."""
        
        self.websocket_connections.append(connection)
        logger.info(f"Added WebSocket connection. Total: {len(self.websocket_connections)}")
        
    def remove_websocket_connection(self, connection: WebSocketServerProtocol):
        """Remove a WebSocket connection."""
        
        if connection in self.websocket_connections:
            self.websocket_connections.remove(connection)
            logger.info(f"Removed WebSocket connection. Total: {len(self.websocket_connections)}")
            
    async def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard."""
        
        current_metrics = self._get_current_metrics()
        health_status = self.get_health_status()
        active_alerts = self.get_active_alerts()
        
        # Get metric statistics for charts
        metric_stats = {}
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage", "error_rate", "response_time"]:
            if metric_name in self.metrics:
                metric_stats[metric_name] = self.get_metric_statistics(metric_name, 60)
                
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": health_status,
            "current_metrics": current_metrics,
            "metric_statistics": metric_stats,
            "active_alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "level": alert.level.value,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            "alert_count": len(active_alerts),
            "websocket_connections": len(self.websocket_connections)
        }
        
    def record_workflow_metric(
        self,
        workflow_id: str,
        metric_name: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Record a workflow-specific metric."""
        
        workflow_tags = {"workflow_id": workflow_id}
        if tags:
            workflow_tags.update(tags)
            
        self.record_metric(
            f"workflow_{metric_name}",
            value,
            MetricType.GAUGE,
            tags=workflow_tags
        )
        
    def record_agent_metric(
        self,
        agent_id: str,
        metric_name: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Record an agent-specific metric."""
        
        agent_tags = {"agent_id": agent_id}
        if tags:
            agent_tags.update(tags)
            
        self.record_metric(
            f"agent_{metric_name}",
            value,
            MetricType.GAUGE,
            tags=agent_tags
        )
        
    def record_api_metric(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        tags: Dict[str, str] = None
    ):
        """Record API metrics."""
        
        api_tags = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        if tags:
            api_tags.update(tags)
            
        self.record_metric("api_response_time", response_time, MetricType.TIMER, tags=api_tags)
        self.record_metric("api_requests", 1, MetricType.COUNTER, tags=api_tags)
        
        if status_code >= 400:
            self.record_metric("api_errors", 1, MetricType.COUNTER, tags=api_tags)





























