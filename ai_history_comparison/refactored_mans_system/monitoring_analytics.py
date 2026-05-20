"""
Advanced Monitoring and Analytics for MANS System

This module provides comprehensive monitoring and analytics capabilities:
- Real-time system monitoring
- Performance analytics
- Business intelligence
- Predictive analytics
- Anomaly detection
- Custom dashboards
- Alerting system
- Data visualization
- Machine learning insights
- Operational intelligence
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import threading
import weakref
import psutil
import gc

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DashboardType(Enum):
    """Dashboard types"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    CUSTOM = "custom"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dashboard:
    """Dashboard data structure"""
    id: str
    name: str
    dashboard_type: DashboardType
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    refresh_interval: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric"""
        with self._lock:
            self.metrics[metric.name].append(metric)
            
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric.name] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric.name] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histograms[metric.name].append(metric.value)
            elif metric.metric_type == MetricType.SUMMARY:
                self.summaries[metric.name] = {
                    "count": len(self.histograms[metric.name]),
                    "sum": sum(self.histograms[metric.name]) if self.histograms[metric.name] else 0,
                    "avg": statistics.mean(self.histograms[metric.name]) if self.histograms[metric.name] else 0,
                    "min": min(self.histograms[metric.name]) if self.histograms[metric.name] else 0,
                    "max": max(self.histograms[metric.name]) if self.histograms[metric.name] else 0
                }
    
    def get_metric(self, name: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Metric]:
        """Get metrics by name and time range"""
        with self._lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            
            if time_range:
                start_time, end_time = time_range
                metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]
            
            return metrics
    
    def get_metric_summary(self, name: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get metric summary statistics"""
        metrics = self.get_metric(name, time_range)
        
        if not metrics:
            return {"count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": np.percentile(values, 95) if values else 0,
            "p99": np.percentile(values, 99) if values else 0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {name: len(values) for name, values in self.histograms.items()},
                "summaries": dict(self.summaries),
                "total_metrics": sum(len(metrics) for metrics in self.metrics.values())
            }

class AnomalyDetector:
    """Advanced anomaly detection system"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, float] = {
            "cpu_usage": 0.8,
            "memory_usage": 0.85,
            "response_time": 1.0,
            "error_rate": 0.05,
            "throughput": 0.5
        }
        self.anomaly_history: List[Dict[str, Any]] = []
    
    def update_baseline(self, metric_name: str, values: List[float]) -> None:
        """Update baseline for anomaly detection"""
        if not values:
            return
        
        self.baseline_metrics[metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "count": len(values),
            "updated_at": datetime.utcnow()
        }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect anomaly in metric value"""
        if metric_name not in self.baseline_metrics:
            return {"is_anomaly": False, "reason": "no_baseline"}
        
        baseline = self.baseline_metrics[metric_name]
        mean = baseline["mean"]
        std = baseline["std"]
        
        if std == 0:
            return {"is_anomaly": False, "reason": "no_variance"}
        
        # Z-score based anomaly detection
        z_score = abs((value - mean) / std)
        threshold = self.anomaly_thresholds.get(metric_name, 3.0)
        
        is_anomaly = z_score > threshold
        
        if is_anomaly:
            anomaly = {
                "metric_name": metric_name,
                "value": value,
                "baseline_mean": mean,
                "baseline_std": std,
                "z_score": z_score,
                "threshold": threshold,
                "timestamp": datetime.utcnow(),
                "severity": "high" if z_score > threshold * 2 else "medium"
            }
            self.anomaly_history.append(anomaly)
            
            return {
                "is_anomaly": True,
                "anomaly": anomaly
            }
        
        return {"is_anomaly": False, "z_score": z_score}
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_anomalies = [a for a in self.anomaly_history if a["timestamp"] > cutoff_time]
        
        return {
            "period_hours": hours,
            "total_anomalies": len(recent_anomalies),
            "high_severity": len([a for a in recent_anomalies if a["severity"] == "high"]),
            "medium_severity": len([a for a in recent_anomalies if a["severity"] == "medium"]),
            "by_metric": self._count_anomalies_by_metric(recent_anomalies),
            "recent_anomalies": recent_anomalies[-10:]
        }
    
    def _count_anomalies_by_metric(self, anomalies: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count anomalies by metric name"""
        counts = defaultdict(int)
        for anomaly in anomalies:
            counts[anomaly["metric_name"]] += 1
        return dict(counts)

class AlertManager:
    """Advanced alerting system"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_channels: Dict[str, Callable] = {}
        self.alert_history: List[Alert] = []
    
    def add_alert_rule(self, name: str, condition: str, severity: AlertSeverity, 
                      description: str, enabled: bool = True) -> None:
        """Add alert rule"""
        self.alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "description": description,
            "enabled": enabled,
            "created_at": datetime.utcnow()
        }
    
    def add_alert_channel(self, name: str, handler: Callable) -> None:
        """Add alert channel"""
        self.alert_channels[name] = handler
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against alert rules"""
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
            
            # Simple condition evaluation (in real implementation, use proper expression parser)
            if self._evaluate_condition(rule["condition"], metrics):
                alert = Alert(
                    id=f"{rule_name}_{int(time.time())}",
                    name=rule_name,
                    description=rule["description"],
                    severity=rule["severity"],
                    labels={"rule": rule_name},
                    metadata={"condition": rule["condition"], "metrics": metrics}
                )
                
                new_alerts.append(alert)
                self.alerts.append(alert)
                self.alert_history.append(alert)
                
                # Send alert through channels
                for channel_name, handler in self.alert_channels.items():
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error sending alert through {channel_name}: {e}")
        
        return new_alerts
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate alert condition (simplified)"""
        # This is a simplified implementation
        # In production, use a proper expression parser like pyparsing
        
        try:
            # Replace metric names with values
            for metric_name, value in metrics.items():
                condition = condition.replace(metric_name, str(value))
            
            # Evaluate the condition
            return eval(condition)
        except:
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff_time]
        
        return {
            "period_hours": hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.get_active_alerts()),
            "resolved_alerts": len([a for a in recent_alerts if a.resolved]),
            "by_severity": self._count_alerts_by_severity(recent_alerts),
            "by_rule": self._count_alerts_by_rule(recent_alerts),
            "recent_alerts": [self._alert_to_dict(a) for a in recent_alerts[-10:]]
        }
    
    def _count_alerts_by_severity(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by severity"""
        counts = defaultdict(int)
        for alert in alerts:
            counts[alert.severity.value] += 1
        return dict(counts)
    
    def _count_alerts_by_rule(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by rule"""
        counts = defaultdict(int)
        for alert in alerts:
            rule = alert.labels.get("rule", "unknown")
            counts[rule] += 1
        return dict(counts)
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": alert.id,
            "name": alert.name,
            "description": alert.description,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
            "resolved": alert.resolved,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "labels": alert.labels,
            "metadata": alert.metadata
        }

class DashboardManager:
    """Advanced dashboard management system"""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.widget_types = {
            "line_chart": self._create_line_chart_widget,
            "bar_chart": self._create_bar_chart_widget,
            "gauge": self._create_gauge_widget,
            "table": self._create_table_widget,
            "metric_card": self._create_metric_card_widget,
            "heatmap": self._create_heatmap_widget
        }
    
    def create_dashboard(self, dashboard_id: str, name: str, 
                        dashboard_type: DashboardType) -> Dashboard:
        """Create new dashboard"""
        dashboard = Dashboard(
            id=dashboard_id,
            name=name,
            dashboard_type=dashboard_type
        )
        self.dashboards[dashboard_id] = dashboard
        return dashboard
    
    def add_widget(self, dashboard_id: str, widget_type: str, 
                   config: Dict[str, Any]) -> bool:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        if widget_type not in self.widget_types:
            return False
        
        widget = self.widget_types[widget_type](config)
        self.dashboards[dashboard_id].widgets.append(widget)
        self.dashboards[dashboard_id].updated_at = datetime.utcnow()
        
        return True
    
    def _create_line_chart_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart widget"""
        return {
            "type": "line_chart",
            "title": config.get("title", "Line Chart"),
            "metrics": config.get("metrics", []),
            "time_range": config.get("time_range", "1h"),
            "config": config
        }
    
    def _create_bar_chart_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart widget"""
        return {
            "type": "bar_chart",
            "title": config.get("title", "Bar Chart"),
            "metrics": config.get("metrics", []),
            "config": config
        }
    
    def _create_gauge_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create gauge widget"""
        return {
            "type": "gauge",
            "title": config.get("title", "Gauge"),
            "metric": config.get("metric", ""),
            "min_value": config.get("min_value", 0),
            "max_value": config.get("max_value", 100),
            "config": config
        }
    
    def _create_table_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create table widget"""
        return {
            "type": "table",
            "title": config.get("title", "Table"),
            "columns": config.get("columns", []),
            "data_source": config.get("data_source", ""),
            "config": config
        }
    
    def _create_metric_card_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create metric card widget"""
        return {
            "type": "metric_card",
            "title": config.get("title", "Metric"),
            "metric": config.get("metric", ""),
            "format": config.get("format", "number"),
            "config": config
        }
    
    def _create_heatmap_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap widget"""
        return {
            "type": "heatmap",
            "title": config.get("title", "Heatmap"),
            "data_source": config.get("data_source", ""),
            "config": config
        }
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data for rendering"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        return {
            "dashboard": {
                "id": dashboard.id,
                "name": dashboard.name,
                "type": dashboard.dashboard_type.value,
                "refresh_interval": dashboard.refresh_interval,
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat()
            },
            "widgets": dashboard.widgets
        }
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards"""
        return [
            {
                "id": dashboard.id,
                "name": dashboard.name,
                "type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets),
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat()
            }
            for dashboard in self.dashboards.values()
        ]

class PredictiveAnalytics:
    """Predictive analytics system"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def train_model(self, model_name: str, data: List[float], 
                   model_type: str = "linear_regression") -> bool:
        """Train predictive model"""
        try:
            if model_type == "linear_regression":
                # Simple linear regression implementation
                x = np.arange(len(data))
                y = np.array(data)
                
                # Calculate slope and intercept
                n = len(x)
                slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                intercept = (np.sum(y) - slope * np.sum(x)) / n
                
                self.models[model_name] = {
                    "type": model_type,
                    "slope": slope,
                    "intercept": intercept,
                    "trained_at": datetime.utcnow(),
                    "data_points": len(data)
                }
                
                return True
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return False
    
    def predict(self, model_name: str, steps_ahead: int = 10) -> List[float]:
        """Make predictions using trained model"""
        if model_name not in self.models:
            return []
        
        model = self.models[model_name]
        
        if model["type"] == "linear_regression":
            slope = model["slope"]
            intercept = model["intercept"]
            last_x = model["data_points"] - 1
            
            predictions = []
            for i in range(1, steps_ahead + 1):
                x = last_x + i
                y = slope * x + intercept
                predictions.append(y)
            
            return predictions
        
        return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        if model_name not in self.models:
            return {"error": "Model not found"}
        
        model = self.models[model_name]
        return {
            "name": model_name,
            "type": model["type"],
            "trained_at": model["trained_at"].isoformat(),
            "data_points": model["data_points"],
            "parameters": {k: v for k, v in model.items() if k not in ["trained_at"]}
        }

class MonitoringAnalytics:
    """Main monitoring and analytics manager"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.dashboard_manager = DashboardManager()
        self.predictive_analytics = PredictiveAnalytics()
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize monitoring and analytics"""
        # Set up default alert rules
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "cpu_usage > 0.8",
            AlertSeverity.WARNING,
            "CPU usage is above 80%"
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "memory_usage > 0.85",
            AlertSeverity.WARNING,
            "Memory usage is above 85%"
        )
        
        self.alert_manager.add_alert_rule(
            "high_response_time",
            "response_time > 1.0",
            AlertSeverity.ERROR,
            "Response time is above 1 second"
        )
        
        # Set up default alert channel
        self.alert_manager.add_alert_channel("logger", self._log_alert)
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_system())
        
        logger.info("Monitoring and analytics initialized")
    
    async def shutdown(self) -> None:
        """Shutdown monitoring and analytics"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        
        logger.info("Monitoring and analytics shut down")
    
    async def _monitor_system(self) -> None:
        """Monitor system metrics"""
        while True:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent / 100.0
                
                # Record metrics
                self.metrics_collector.record_metric(Metric("cpu_usage", cpu_usage))
                self.metrics_collector.record_metric(Metric("memory_usage", memory_usage))
                
                # Check for anomalies
                cpu_anomaly = self.anomaly_detector.detect_anomaly("cpu_usage", cpu_usage)
                memory_anomaly = self.anomaly_detector.detect_anomaly("memory_usage", memory_usage)
                
                # Check alerts
                current_metrics = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "response_time": 0.1,  # Placeholder
                    "error_rate": 0.01,    # Placeholder
                    "throughput": 100.0    # Placeholder
                }
                
                self.alert_manager.check_alerts(current_metrics)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert (default alert channel)"""
        logger.warning(f"ALERT: {alert.name} - {alert.description} [{alert.severity.value}]")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            "metrics": self.metrics_collector.get_all_metrics(),
            "anomalies": self.anomaly_detector.get_anomaly_summary(),
            "alerts": self.alert_manager.get_alert_summary(),
            "dashboards": self.dashboard_manager.list_dashboards(),
            "models": list(self.predictive_analytics.models.keys())
        }


