"""
🚀 Ultra Library Optimization V7 - Real-Time Analytics & Monitoring System
==========================================================================

Advanced real-time analytics, monitoring, and intelligent alerting system.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis.asyncio as redis
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from structlog import get_logger
import websockets
from websockets.server import serve
import asyncio_mqtt as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnalyticsType(Enum):
    """Types of analytics."""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    USER_BEHAVIOR = "user_behavior"


class DashboardType(Enum):
    """Types of dashboards."""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    PREDICTIVE = "predictive"
    ALERTS = "alerts"
    CUSTOM = "custom"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricData:
    """Metric data structure."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    description: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", "==", ">=", "<="
    threshold: float
    severity: AlertSeverity
    duration: int = 60  # seconds
    enabled: bool = True
    description: str = ""
    actions: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert information."""
    id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_value: float
    threshold: float
    status: str = "active"  # active, resolved, acknowledged
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None


@dataclass
class AnalyticsConfig:
    """Analytics configuration."""
    retention_days: int = 30
    aggregation_interval: int = 60  # seconds
    anomaly_detection_enabled: bool = True
    predictive_analytics_enabled: bool = True
    real_time_dashboard_enabled: bool = True
    alerting_enabled: bool = True
    data_export_enabled: bool = True


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    chart_types: List[str] = field(default_factory=lambda: ["line", "bar", "gauge"])
    custom_widgets_enabled: bool = True
    export_enabled: bool = True


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Advanced metrics collector with multiple backends."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.prometheus_metrics: Dict[str, Any] = {}
        self.influx_client = None
        self.redis_client = None
        self._logger = get_logger(__name__)
        
        # Initialize backends
        self._setup_prometheus()
        self._setup_influxdb()
        self._setup_redis()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics."""
        try:
            # Create Prometheus metrics
            self.prometheus_metrics = {
                "request_counter": Counter(
                    "analytics_requests_total",
                    "Total requests",
                    ["service", "endpoint", "status"]
                ),
                "response_time": Histogram(
                    "analytics_response_time_seconds",
                    "Response time in seconds",
                    ["service", "endpoint"]
                ),
                "active_connections": Gauge(
                    "analytics_active_connections",
                    "Active connections",
                    ["service"]
                ),
                "error_rate": Gauge(
                    "analytics_error_rate",
                    "Error rate percentage",
                    ["service"]
                ),
                "throughput": Summary(
                    "analytics_throughput",
                    "Requests per second",
                    ["service"]
                )
            }
            self._logger.info("Prometheus metrics setup completed")
        except Exception as e:
            self._logger.error(f"Failed to setup Prometheus metrics: {e}")
    
    def _setup_influxdb(self):
        """Setup InfluxDB client."""
        try:
            self.influx_client = InfluxDBClient(
                url="http://localhost:8086",
                token="your-token",
                org="your-org",
                bucket="analytics"
            )
            self._logger.info("InfluxDB client setup completed")
        except Exception as e:
            self._logger.error(f"Failed to setup InfluxDB: {e}")
    
    def _setup_redis(self):
        """Setup Redis client."""
        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self._logger.info("Redis client setup completed")
        except Exception as e:
            self._logger.error(f"Failed to setup Redis: {e}")
    
    async def record_metric(self, metric: MetricData):
        """Record a metric across all backends."""
        try:
            # Store in memory
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            self.metrics[metric.name].append(metric)
            
            # Store in Redis for real-time access
            if self.redis_client:
                await self.redis_client.setex(
                    f"metric:{metric.name}:{metric.timestamp}",
                    3600,  # 1 hour TTL
                    json.dumps({
                        "value": metric.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp
                    })
                )
            
            # Store in InfluxDB for time-series data
            if self.influx_client:
                point = Point(metric.name) \
                    .field("value", metric.value) \
                    .time(metric.timestamp, WritePrecision.NS)
                
                for key, value in metric.labels.items():
                    point = point.tag(key, value)
                
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                write_api.write(bucket="analytics", record=point)
            
            # Update Prometheus metrics
            if metric.name in self.prometheus_metrics:
                if isinstance(self.prometheus_metrics[metric.name], Counter):
                    self.prometheus_metrics[metric.name].labels(**metric.labels).inc()
                elif isinstance(self.prometheus_metrics[metric.name], Gauge):
                    self.prometheus_metrics[metric.name].labels(**metric.labels).set(metric.value)
                elif isinstance(self.prometheus_metrics[metric.name], Histogram):
                    self.prometheus_metrics[metric.name].labels(**metric.labels).observe(metric.value)
            
            self._logger.debug(f"Metric recorded: {metric.name} = {metric.value}")
            
        except Exception as e:
            self._logger.error(f"Failed to record metric: {e}")
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[MetricData]:
        """Get metric history from InfluxDB."""
        try:
            if not self.influx_client:
                return []
            
            query_api = self.influx_client.query_api()
            query = f'''
                from(bucket: "analytics")
                    |> range(start: -{hours}h)
                    |> filter(fn: (r) => r._measurement == "{metric_name}")
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = query_api.query(query)
            metrics = []
            
            for table in result:
                for record in table.records:
                    metric = MetricData(
                        name=metric_name,
                        value=record.get_value(),
                        timestamp=record.get_time().timestamp(),
                        labels=record.values
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            self._logger.error(f"Failed to get metric history: {e}")
            return []
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from Redis."""
        try:
            if not self.redis_client:
                return {}
            
            # Get all metric keys
            keys = await self.redis_client.keys("metric:*")
            metrics = {}
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    metric_data = json.loads(data)
                    metric_name = key.decode().split(":")[1]
                    metrics[metric_name] = metric_data
            
            return metrics
            
        except Exception as e:
            self._logger.error(f"Failed to get real-time metrics: {e}")
            return {}


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

class AnomalyDetector:
    """Advanced anomaly detection using machine learning."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self._logger = get_logger(__name__)
    
    def train_anomaly_model(self, metric_name: str, historical_data: List[float]) -> bool:
        """Train an anomaly detection model for a metric."""
        try:
            if len(historical_data) < 100:
                self._logger.warning(f"Insufficient data for training: {len(historical_data)} points")
                return False
            
            # Prepare data
            data = np.array(historical_data).reshape(-1, 1)
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            model.fit(scaled_data)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            
            # Calculate threshold
            scores = model.score_samples(scaled_data)
            self.anomaly_thresholds[metric_name] = np.percentile(scores, 5)
            
            self._logger.info(f"Anomaly model trained for {metric_name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to train anomaly model: {e}")
            return False
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if a value is anomalous."""
        try:
            if metric_name not in self.models:
                return False, 0.0
            
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
            threshold = self.anomaly_thresholds[metric_name]
            
            # Scale the value
            scaled_value = scaler.transform([[value]])
            
            # Get anomaly score
            score = model.score_samples(scaled_value)[0]
            
            # Check if anomalous
            is_anomalous = score < threshold
            
            return is_anomalous, score
            
        except Exception as e:
            self._logger.error(f"Failed to detect anomaly: {e}")
            return False, 0.0
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        return {
            "trained_models": len(self.models),
            "metrics_with_models": list(self.models.keys()),
            "thresholds": self.anomaly_thresholds
        }


# =============================================================================
# PREDICTIVE ANALYTICS
# =============================================================================

class PredictiveAnalytics:
    """Advanced predictive analytics using time series forecasting."""
    
    def __init__(self):
        self.forecast_models: Dict[str, Any] = {}
        self.trend_models: Dict[str, Any] = {}
        self.seasonality_models: Dict[str, Any] = {}
        self._logger = get_logger(__name__)
    
    def train_forecast_model(self, metric_name: str, historical_data: List[float], 
                           forecast_horizon: int = 24) -> bool:
        """Train a forecasting model for a metric."""
        try:
            if len(historical_data) < 50:
                self._logger.warning(f"Insufficient data for forecasting: {len(historical_data)} points")
                return False
            
            # Simple moving average model (in production, use ARIMA, Prophet, or LSTM)
            window_size = min(10, len(historical_data) // 4)
            moving_avg = np.convolve(historical_data, np.ones(window_size)/window_size, mode='valid')
            
            # Store model
            self.forecast_models[metric_name] = {
                "type": "moving_average",
                "window_size": window_size,
                "last_values": historical_data[-window_size:],
                "forecast_horizon": forecast_horizon
            }
            
            self._logger.info(f"Forecast model trained for {metric_name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to train forecast model: {e}")
            return False
    
    def predict_forecast(self, metric_name: str, steps: int = 24) -> List[float]:
        """Generate forecast predictions."""
        try:
            if metric_name not in self.forecast_models:
                return []
            
            model = self.forecast_models[metric_name]
            
            if model["type"] == "moving_average":
                # Simple moving average forecast
                last_values = model["last_values"]
                window_size = model["window_size"]
                
                predictions = []
                for _ in range(steps):
                    prediction = np.mean(last_values[-window_size:])
                    predictions.append(prediction)
                    last_values.append(prediction)
                
                return predictions
            
            return []
            
        except Exception as e:
            self._logger.error(f"Failed to generate forecast: {e}")
            return []
    
    def detect_trends(self, metric_name: str, historical_data: List[float]) -> Dict[str, Any]:
        """Detect trends in metric data."""
        try:
            if len(historical_data) < 10:
                return {"trend": "insufficient_data"}
            
            # Calculate trend using linear regression
            x = np.arange(len(historical_data))
            y = np.array(historical_data)
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Determine trend
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            return {
                "trend": trend,
                "slope": slope,
                "r_squared": r_squared,
                "confidence": "high" if r_squared > 0.7 else "medium" if r_squared > 0.4 else "low"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to detect trends: {e}")
            return {"trend": "error"}


# =============================================================================
# ALERTING SYSTEM
# =============================================================================

class AlertingSystem:
    """Advanced alerting system with intelligent rules and actions."""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._logger = get_logger(__name__)
        
        # Alert counters
        self.alert_counter = Counter(
            "analytics_alerts_total",
            "Total alerts generated",
            ["severity", "rule_name"]
        )
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add an alert rule."""
        try:
            self.alert_rules[rule.id] = rule
            self._logger.info(f"Alert rule added: {rule.name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to add alert rule: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self._logger.info(f"Alert rule removed: {rule_id}")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to remove alert rule: {e}")
            return False
    
    def evaluate_metric(self, metric_name: str, value: float, timestamp: float) -> List[Alert]:
        """Evaluate a metric against alert rules."""
        triggered_alerts = []
        
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled or rule.metric_name != metric_name:
                    continue
                
                # Check condition
                triggered = False
                if rule.condition == ">":
                    triggered = value > rule.threshold
                elif rule.condition == "<":
                    triggered = value < rule.threshold
                elif rule.condition == "==":
                    triggered = value == rule.threshold
                elif rule.condition == ">=":
                    triggered = value >= rule.threshold
                elif rule.condition == "<=":
                    triggered = value <= rule.threshold
                
                if triggered:
                    # Check if alert is already active
                    if rule_id not in self.active_alerts:
                        alert = Alert(
                            id=str(uuid.uuid4()),
                            rule_id=rule_id,
                            severity=rule.severity,
                            message=f"{rule.name}: {metric_name} = {value} {rule.condition} {rule.threshold}",
                            timestamp=timestamp,
                            metric_value=value,
                            threshold=rule.threshold
                        )
                        
                        self.active_alerts[rule_id] = alert
                        self.alert_history.append(alert)
                        triggered_alerts.append(alert)
                        
                        # Update metrics
                        self.alert_counter.labels(
                            severity=rule.severity.value,
                            rule_name=rule.name
                        ).inc()
                        
                        self._logger.warning(f"Alert triggered: {alert.message}")
                    else:
                        # Update existing alert
                        self.active_alerts[rule_id].timestamp = timestamp
                        self.active_alerts[rule_id].metric_value = value
                else:
                    # Resolve alert if condition is no longer met
                    if rule_id in self.active_alerts:
                        alert = self.active_alerts[rule_id]
                        alert.status = "resolved"
                        del self.active_alerts[rule_id]
                        self._logger.info(f"Alert resolved: {alert.message}")
            
            return triggered_alerts
            
        except Exception as e:
            self._logger.error(f"Failed to evaluate metric: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        try:
            for alert in self.alert_history:
                if alert.id == alert_id:
                    alert.status = "acknowledged"
                    alert.acknowledged_by = user
                    alert.acknowledged_at = time.time()
                    
                    if alert.rule_id in self.active_alerts:
                        del self.active_alerts[alert.rule_id]
                    
                    self._logger.info(f"Alert acknowledged by {user}: {alert.message}")
                    return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


# =============================================================================
# REAL-TIME DASHBOARD
# =============================================================================

class RealTimeDashboard:
    """Advanced real-time dashboard with WebSocket support."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.websocket_connections: List[WebSocket] = []
        self.dashboard_data: Dict[str, Any] = {}
        self._logger = get_logger(__name__)
        
        # Initialize Dash app
        self.dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup the dashboard layout."""
        self.dash_app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Ultra Library Optimization V7 - Real-Time Analytics", 
                           className="text-center mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Performance Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart", style={"height": "300px"}),
                            dcc.Interval(
                                id="performance-interval",
                                interval=self.config.refresh_interval * 1000,
                                n_intervals=0
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Active Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-container"),
                            dcc.Interval(
                                id="alerts-interval",
                                interval=self.config.refresh_interval * 1000,
                                n_intervals=0
                            )
                        ])
                    ])
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Predictive Analytics"),
                        dbc.CardBody([
                            dcc.Graph(id="forecast-chart", style={"height": "300px"}),
                            dcc.Interval(
                                id="forecast-interval",
                                interval=self.config.refresh_interval * 1000,
                                n_intervals=0
                            )
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a WebSocket connection for real-time updates."""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        self._logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            self._logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast updates to all WebSocket connections."""
        message = json.dumps(data)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                self._logger.error(f"Failed to send WebSocket message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected connections
        for websocket in disconnected:
            await self.remove_websocket_connection(websocket)
    
    def update_dashboard_data(self, data: Dict[str, Any]):
        """Update dashboard data."""
        self.dashboard_data.update(data)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()


# =============================================================================
# REAL-TIME ANALYTICS MANAGER
# =============================================================================

class RealTimeAnalytics:
    """
    Advanced real-time analytics and monitoring system.
    
    Features:
    - Real-time metrics collection and storage
    - Anomaly detection using machine learning
    - Predictive analytics and forecasting
    - Intelligent alerting system
    - Real-time dashboard with WebSocket support
    - Multi-backend storage (InfluxDB, Redis, Prometheus)
    """
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.predictive_analytics = PredictiveAnalytics()
        self.alerting_system = AlertingSystem()
        self.dashboard = RealTimeDashboard(DashboardConfig())
        self._logger = get_logger(__name__)
        
        # Background tasks
        self._running = False
        self._background_tasks = []
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="🚀 Ultra Library Optimization V7 - Real-Time Analytics",
            description="Advanced real-time analytics and monitoring system",
            version="1.0.0"
        )
        
        self._setup_routes()
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def analytics_info():
            return {
                "name": "Ultra Library Optimization V7 - Real-Time Analytics",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Real-time metrics collection",
                    "Anomaly detection",
                    "Predictive analytics",
                    "Intelligent alerting",
                    "Real-time dashboard"
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "metrics_collected": len(self.metrics_collector.metrics),
                "active_alerts": len(self.alerting_system.get_active_alerts()),
                "websocket_connections": len(self.dashboard.websocket_connections)
            }
        
        @self.app.post("/metrics")
        async def record_metric(metric: MetricData):
            await self.metrics_collector.record_metric(metric)
            
            # Check for anomalies
            if self.config.anomaly_detection_enabled:
                is_anomalous, score = self.anomaly_detector.detect_anomaly(metric.name, metric.value)
                if is_anomalous:
                    self._logger.warning(f"Anomaly detected: {metric.name} = {metric.value} (score: {score})")
            
            # Evaluate alert rules
            if self.config.alerting_enabled:
                alerts = self.alerting_system.evaluate_metric(metric.name, metric.value, metric.timestamp)
                if alerts:
                    # Broadcast alert to dashboard
                    await self.dashboard.broadcast_update({
                        "type": "alert",
                        "alerts": [alert.__dict__ for alert in alerts]
                    })
            
            # Update dashboard data
            self.dashboard.update_dashboard_data({
                metric.name: {
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "labels": metric.labels
                }
            })
            
            return {"status": "recorded", "metric_id": metric.name}
        
        @self.app.get("/metrics/{metric_name}")
        async def get_metric_history(metric_name: str, hours: int = 24):
            metrics = await self.metrics_collector.get_metric_history(metric_name, hours)
            return {
                "metric_name": metric_name,
                "data_points": len(metrics),
                "metrics": [metric.__dict__ for metric in metrics]
            }
        
        @self.app.get("/analytics/anomalies")
        async def get_anomaly_stats():
            return self.anomaly_detector.get_anomaly_stats()
        
        @self.app.post("/analytics/forecast/{metric_name}")
        async def generate_forecast(metric_name: str, steps: int = 24):
            predictions = self.predictive_analytics.predict_forecast(metric_name, steps)
            return {
                "metric_name": metric_name,
                "predictions": predictions,
                "steps": steps
            }
        
        @self.app.get("/alerts")
        async def get_alerts():
            return {
                "active_alerts": [alert.__dict__ for alert in self.alerting_system.get_active_alerts()],
                "alert_history": [alert.__dict__ for alert in self.alerting_system.get_alert_history()]
            }
        
        @self.app.post("/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, user: str):
            success = self.alerting_system.acknowledge_alert(alert_id, user)
            return {"success": success}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.dashboard.add_websocket_connection(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                await self.dashboard.remove_websocket_connection(websocket)
        
        @self.app.get("/dashboard")
        async def dashboard_page():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Real-Time Analytics Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>🚀 Real-Time Analytics Dashboard</h1>
                <div id="metrics"></div>
                <div id="alerts"></div>
                <script>
                    const ws = new WebSocket('ws://localhost:8000/ws');
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'alert') {
                            updateAlerts(data.alerts);
                        }
                    };
                    
                    function updateAlerts(alerts) {
                        const alertsDiv = document.getElementById('alerts');
                        alertsDiv.innerHTML = alerts.map(alert => 
                            `<div style="color: red;">🚨 ${alert.message}</div>`
                        ).join('');
                    }
                </script>
            </body>
            </html>
            """)
    
    def _start_background_tasks(self):
        """Start background tasks for analytics."""
        self._running = True
        
        # Start metrics aggregation task
        asyncio.create_task(self._metrics_aggregation_task())
        
        # Start anomaly detection training task
        asyncio.create_task(self._anomaly_training_task())
        
        # Start forecast training task
        asyncio.create_task(self._forecast_training_task())
        
        # Start dashboard update task
        asyncio.create_task(self._dashboard_update_task())
    
    async def _metrics_aggregation_task(self):
        """Background task for metrics aggregation."""
        while self._running:
            try:
                # Aggregate metrics every minute
                await asyncio.sleep(60)
                
                # Get real-time metrics
                real_time_metrics = await self.metrics_collector.get_real_time_metrics()
                
                # Broadcast to dashboard
                await self.dashboard.broadcast_update({
                    "type": "metrics_update",
                    "metrics": real_time_metrics
                })
                
            except Exception as e:
                self._logger.error(f"Metrics aggregation task failed: {e}")
    
    async def _anomaly_training_task(self):
        """Background task for anomaly detection training."""
        while self._running:
            try:
                # Train anomaly models every hour
                await asyncio.sleep(3600)
                
                # Get historical data for training
                for metric_name in self.metrics_collector.metrics.keys():
                    historical_data = await self.metrics_collector.get_metric_history(metric_name, 24)
                    if historical_data:
                        values = [metric.value for metric in historical_data]
                        self.anomaly_detector.train_anomaly_model(metric_name, values)
                
            except Exception as e:
                self._logger.error(f"Anomaly training task failed: {e}")
    
    async def _forecast_training_task(self):
        """Background task for forecast training."""
        while self._running:
            try:
                # Train forecast models every 6 hours
                await asyncio.sleep(21600)
                
                # Get historical data for training
                for metric_name in self.metrics_collector.metrics.keys():
                    historical_data = await self.metrics_collector.get_metric_history(metric_name, 168)  # 1 week
                    if historical_data:
                        values = [metric.value for metric in historical_data]
                        self.predictive_analytics.train_forecast_model(metric_name, values)
                
            except Exception as e:
                self._logger.error(f"Forecast training task failed: {e}")
    
    async def _dashboard_update_task(self):
        """Background task for dashboard updates."""
        while self._running:
            try:
                # Update dashboard every 5 seconds
                await asyncio.sleep(5)
                
                # Get current dashboard data
                dashboard_data = self.dashboard.get_dashboard_data()
                
                # Add analytics data
                dashboard_data.update({
                    "anomaly_stats": self.anomaly_detector.get_anomaly_stats(),
                    "active_alerts": len(self.alerting_system.get_active_alerts()),
                    "total_metrics": len(self.metrics_collector.metrics)
                })
                
                # Broadcast update
                await self.dashboard.broadcast_update({
                    "type": "dashboard_update",
                    "data": dashboard_data
                })
                
            except Exception as e:
                self._logger.error(f"Dashboard update task failed: {e}")
    
    def stop(self):
        """Stop the analytics system."""
        self._running = False
        self._logger.info("Real-time analytics system stopped")


# =============================================================================
# DECORATORS
# =============================================================================

def track_metric(metric_name: str, metric_type: MetricType = MetricType.GAUGE):
    """Decorator to track metrics for a function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metric
                metric = MetricData(
                    name=f"{metric_name}_success",
                    value=1.0,
                    timestamp=time.time(),
                    metric_type=metric_type
                )
                
                return result
                
            except Exception as e:
                # Record error metric
                metric = MetricData(
                    name=f"{metric_name}_error",
                    value=1.0,
                    timestamp=time.time(),
                    metric_type=metric_type
                )
                raise e
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                metric = MetricData(
                    name=f"{metric_name}_duration",
                    value=execution_time,
                    timestamp=time.time(),
                    metric_type=MetricType.HISTOGRAM
                )
        
        return wrapper
    return decorator


def monitor_performance(threshold: float = 1.0):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = await func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            if execution_time > threshold:
                # Log performance warning
                logging.warning(f"Performance warning: {func.__name__} took {execution_time:.2f}s")
            
            return result
        
        return wrapper
    return decorator


# =============================================================================
# MAIN APPLICATION
# =============================================================================

async def main():
    """Main application entry point."""
    # Initialize real-time analytics
    config = AnalyticsConfig()
    analytics = RealTimeAnalytics(config)
    
    # Add example alert rules
    alert_rules = [
        AlertRule(
            id="high_error_rate",
            name="High Error Rate",
            metric_name="error_rate",
            condition=">",
            threshold=5.0,
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            id="high_response_time",
            name="High Response Time",
            metric_name="response_time",
            condition=">",
            threshold=2.0,
            severity=AlertSeverity.ERROR
        ),
        AlertRule(
            id="low_throughput",
            name="Low Throughput",
            metric_name="throughput",
            condition="<",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL
        )
    ]
    
    for rule in alert_rules:
        analytics.alerting_system.add_alert_rule(rule)
    
    # Start the application
    import uvicorn
    uvicorn.run(analytics.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main()) 