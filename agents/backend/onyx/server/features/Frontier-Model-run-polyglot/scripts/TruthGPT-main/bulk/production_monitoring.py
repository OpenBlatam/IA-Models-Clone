#!/usr/bin/env python3
"""
Production Monitoring - Production-ready monitoring system
Handles metrics collection, health checks, alerting, and monitoring integration
"""

import time
import threading
import psutil
import json
import requests
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """Health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = None
    timestamp: float = None
    description: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.labels is None:
            self.labels = {}

@dataclass
class Alert:
    """Alert data structure."""
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class HealthCheck:
    """Health check data structure."""
    name: str
    status: HealthStatus
    message: str
    response_time: float = 0.0
    timestamp: float = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}

class MetricsCollector:
    """Metrics collection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = {}
        self.metric_history = []
        self.collection_interval = self.config.get('collection_interval', 10)
        self.max_history_size = self.config.get('max_history_size', 1000)
        self.running = False
        self.collection_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start metrics collection."""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collect_metrics)
            self.collection_thread.start()
            self.logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("Metrics collection stopped")
    
    def _collect_metrics(self):
        """Collect metrics in background."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            self.add_metric(Metric(
                name="system_cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"host": psutil.Process().name()},
                description="CPU usage percentage"
            ))
            
            self.add_metric(Metric(
                name="system_cpu_count",
                value=cpu_count,
                metric_type=MetricType.GAUGE,
                description="Number of CPU cores"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.add_metric(Metric(
                name="system_memory_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                description="Memory usage percentage"
            ))
            
            self.add_metric(Metric(
                name="system_memory_available",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                description="Available memory in bytes"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.add_metric(Metric(
                name="system_disk_percent",
                value=disk.percent,
                metric_type=MetricType.GAUGE,
                description="Disk usage percentage"
            ))
            
            self.add_metric(Metric(
                name="system_disk_free",
                value=disk.free,
                metric_type=MetricType.GAUGE,
                description="Free disk space in bytes"
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.add_metric(Metric(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricType.COUNTER,
                description="Network bytes sent"
            ))
            
            self.add_metric(Metric(
                name="system_network_bytes_recv",
                value=network.bytes_recv,
                metric_type=MetricType.COUNTER,
                description="Network bytes received"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Process metrics
            process = psutil.Process()
            self.add_metric(Metric(
                name="application_memory_usage",
                value=process.memory_info().rss,
                metric_type=MetricType.GAUGE,
                description="Application memory usage in bytes"
            ))
            
            self.add_metric(Metric(
                name="application_cpu_percent",
                value=process.cpu_percent(),
                metric_type=MetricType.GAUGE,
                description="Application CPU usage percentage"
            ))
            
            self.add_metric(Metric(
                name="application_thread_count",
                value=process.num_threads(),
                metric_type=MetricType.GAUGE,
                description="Number of threads"
            ))
            
            self.add_metric(Metric(
                name="application_open_files",
                value=len(process.open_files()),
                metric_type=MetricType.GAUGE,
                description="Number of open files"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
    
    def add_metric(self, metric: Metric):
        """Add a metric."""
        self.metrics[metric.name] = metric
        self.metric_history.append(metric)
        
        # Trim history if too large
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.max_history_size:]
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[Metric]:
        """Get metrics by type."""
        return [m for m in self.metrics.values() if m.metric_type == metric_type]
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Metric]:
        """Get metric history."""
        return [m for m in self.metric_history if m.name == name][-limit:]

class AlertManager:
    """Alert management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alerts = []
        self.alert_rules = []
        self.alert_handlers = []
        self.logger = logging.getLogger(__name__)
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float, 
                      level: AlertLevel, message: str):
        """Add an alert rule."""
        rule = {
            'name': name,
            'metric_name': metric_name,
            'threshold': threshold,
            'level': level,
            'message': message
        }
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {name}")
    
    def check_alerts(self, metrics: Dict[str, Metric]):
        """Check alerts against current metrics."""
        for rule in self.alert_rules:
            metric = metrics.get(rule['metric_name'])
            if metric and self._should_trigger_alert(metric, rule):
                alert = Alert(
                    name=rule['name'],
                    level=rule['level'],
                    message=rule['message'],
                    metric_name=rule['metric_name'],
                    threshold=rule['threshold'],
                    current_value=metric.value
                )
                self.alerts.append(alert)
                self._handle_alert(alert)
    
    def _should_trigger_alert(self, metric: Metric, rule: Dict[str, Any]) -> bool:
        """Check if alert should be triggered."""
        if rule['level'] == AlertLevel.ERROR:
            return metric.value > rule['threshold']
        elif rule['level'] == AlertLevel.WARNING:
            return metric.value > rule['threshold'] * 0.8
        else:
            return metric.value > rule['threshold']
    
    def _handle_alert(self, alert: Alert):
        """Handle an alert."""
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_name: str):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                self.logger.info(f"Alert resolved: {alert_name}")

class HealthChecker:
    """Health check system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks = []
        self.check_interval = self.config.get('check_interval', 30)
        self.running = False
        self.check_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start health checks."""
        if not self.running:
            self.running = True
            self.check_thread = threading.Thread(target=self._run_health_checks)
            self.check_thread.start()
            self.logger.info("Health checks started")
    
    def stop(self):
        """Stop health checks."""
        self.running = False
        if self.check_thread:
            self.check_thread.join()
        self.logger.info("Health checks stopped")
    
    def _run_health_checks(self):
        """Run health checks in background."""
        while self.running:
            try:
                self._check_system_health()
                self._check_application_health()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health checks: {e}")
    
    def _check_system_health(self):
        """Check system health."""
        try:
            # CPU health
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage too high: {cpu_percent}%"
            elif cpu_percent > 80:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage high: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent}%"
            
            self.health_checks.append(HealthCheck(
                name="system_cpu",
                status=status,
                message=message,
                response_time=0.0
            ))
            
            # Memory health
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory.percent}%"
            elif memory.percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage high: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent}%"
            
            self.health_checks.append(HealthCheck(
                name="system_memory",
                status=status,
                message=message,
                response_time=0.0
            ))
            
            # Disk health
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk.percent}%"
            elif disk.percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage high: {disk.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk.percent}%"
            
            self.health_checks.append(HealthCheck(
                name="system_disk",
                status=status,
                message=message,
                response_time=0.0
            ))
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    def _check_application_health(self):
        """Check application health."""
        try:
            # Process health
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage > 1000:  # 1GB
                status = HealthStatus.UNHEALTHY
                message = f"Application memory usage high: {memory_usage:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Application memory usage normal: {memory_usage:.1f}MB"
            
            self.health_checks.append(HealthCheck(
                name="application_memory",
                status=status,
                message=message,
                response_time=0.0
            ))
            
        except Exception as e:
            self.logger.error(f"Error checking application health: {e}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall health status."""
        if not self.health_checks:
            return HealthStatus.HEALTHY
        
        latest_checks = {}
        for check in self.health_checks[-10:]:  # Last 10 checks
            latest_checks[check.name] = check.status
        
        if any(status == HealthStatus.CRITICAL for status in latest_checks.values()):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.UNHEALTHY for status in latest_checks.values()):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in latest_checks.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_latest_health_checks(self) -> List[HealthCheck]:
        """Get latest health checks."""
        return self.health_checks[-10:] if self.health_checks else []

class ProductionMonitor:
    """Production monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        self.alert_manager = AlertManager(config.get('alerts', {}))
        self.health_checker = HealthChecker(config.get('health', {}))
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def start(self):
        """Start monitoring."""
        if not self.running:
            self.running = True
            self.metrics_collector.start()
            self.health_checker.start()
            self._setup_default_alerts()
            self.logger.info("Production monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        self.metrics_collector.stop()
        self.health_checker.stop()
        self.logger.info("Production monitoring stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            "high_cpu",
            "system_cpu_percent",
            80.0,
            AlertLevel.WARNING,
            "CPU usage is high"
        )
        
        self.alert_manager.add_alert_rule(
            "critical_cpu",
            "system_cpu_percent",
            95.0,
            AlertLevel.CRITICAL,
            "CPU usage is critical"
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory",
            "system_memory_percent",
            85.0,
            AlertLevel.WARNING,
            "Memory usage is high"
        )
        
        self.alert_manager.add_alert_rule(
            "critical_memory",
            "system_memory_percent",
            95.0,
            AlertLevel.CRITICAL,
            "Memory usage is critical"
        )
        
        self.alert_manager.add_alert_rule(
            "high_disk",
            "system_disk_percent",
            90.0,
            AlertLevel.WARNING,
            "Disk usage is high"
        )
    
    def check_alerts(self):
        """Check alerts against current metrics."""
        self.alert_manager.check_alerts(self.metrics_collector.metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'total_metrics': len(self.metrics_collector.metrics),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'health_status': self.health_checker.get_overall_health().value,
            'latest_metrics': {
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'type': metric.metric_type.value
                }
                for name, metric in self.metrics_collector.metrics.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            'overall_status': self.health_checker.get_overall_health().value,
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': check.timestamp
                }
                for check in self.health_checker.get_latest_health_checks()
            ]
        }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary."""
        active_alerts = self.alert_manager.get_active_alerts()
        return {
            'total_alerts': len(active_alerts),
            'alerts_by_level': {
                level.value: len([a for a in active_alerts if a.level == level])
                for level in AlertLevel
            },
            'alerts': [
                {
                    'name': alert.name,
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric': alert.metric_name,
                    'threshold': alert.threshold,
                    'current_value': alert.current_value,
                    'timestamp': alert.timestamp
                }
                for alert in active_alerts
            ]
        }

def create_production_monitor(config: Optional[Dict[str, Any]] = None) -> ProductionMonitor:
    """Create production monitor."""
    return ProductionMonitor(config)

if __name__ == "__main__":
    # Example usage
    config = {
        'metrics': {
            'collection_interval': 5,
            'max_history_size': 100
        },
        'health': {
            'check_interval': 10
        },
        'alerts': {}
    }
    
    monitor = create_production_monitor(config)
    monitor.start()
    
    try:
        # Run for 60 seconds
        time.sleep(60)
        
        # Print status
        print("Metrics Summary:")
        print(json.dumps(monitor.get_metrics_summary(), indent=2))
        
        print("\nHealth Status:")
        print(json.dumps(monitor.get_health_status(), indent=2))
        
        print("\nAlerts Summary:")
        print(json.dumps(monitor.get_alerts_summary(), indent=2))
        
    finally:
        monitor.stop()
        print("✅ Production monitoring example completed")

