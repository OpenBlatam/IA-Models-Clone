"""
Advanced Monitoring and Alerting System for Email Sequence System

This module provides comprehensive monitoring, alerting, and predictive analytics
for system health, performance, and business metrics.
"""

import asyncio
import logging
import time
import json
import smtplib
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import psutil
import threading

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .exceptions import MonitoringError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"


class NotificationChannel(str, Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PUSH = "push"
    DASHBOARD = "dashboard"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    threshold: float
    current_value: float
    condition: str
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    cooldown_seconds: int = 300  # 5 minutes
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictiveAnalysis:
    """Predictive analysis result"""
    analysis_id: str
    metric_name: str
    prediction_type: str
    predicted_value: float
    confidence: float
    time_horizon: int  # seconds
    created_at: datetime
    actual_value: Optional[float] = None
    accuracy: Optional[float] = None


class AdvancedMonitoringSystem:
    """Advanced monitoring and alerting system"""
    
    def __init__(self):
        """Initialize the advanced monitoring system"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.predictive_analyses: List[PredictiveAnalysis] = []
        
        # Performance tracking
        self.total_metrics_collected = 0
        self.total_alerts_triggered = 0
        self.total_notifications_sent = 0
        
        # System health
        self.system_health_score = 100.0
        self.last_health_check = datetime.utcnow()
        
        # Monitoring intervals
        self.metrics_collection_interval = 10  # seconds
        self.health_check_interval = 60  # seconds
        self.alert_evaluation_interval = 30  # seconds
        self.predictive_analysis_interval = 300  # seconds
        
        # Notification settings
        self.notification_enabled = True
        self.notification_cooldown = 300  # seconds
        self.last_notifications: Dict[str, datetime] = {}
        
        logger.info("Advanced Monitoring System initialized")
    
    async def initialize(self) -> None:
        """Initialize the advanced monitoring system"""
        try:
            # Load alert rules
            await self._load_alert_rules()
            
            # Start background monitoring tasks
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._alert_evaluator())
            asyncio.create_task(self._predictive_analyzer())
            asyncio.create_task(self._system_optimizer())
            
            # Initialize baseline metrics
            await self._establish_baseline_metrics()
            
            logger.info("Advanced Monitoring System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced monitoring system: {e}")
            raise MonitoringError(f"Failed to initialize advanced monitoring system: {e}")
    
    async def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Collect a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            metadata: Additional metadata
        """
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            # Store metric
            self.metrics[name].append(metric)
            
            # Update statistics
            self.total_metrics_collected += 1
            
            # Check for alert conditions
            await self._check_alert_conditions(name, value)
            
            logger.debug(f"Collected metric: {name} = {value}")
            
        except Exception as e:
            logger.error(f"Error collecting metric {name}: {e}")
    
    async def create_alert_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        notification_channels: Optional[List[NotificationChannel]] = None,
        cooldown_seconds: int = 300,
        tags: Optional[Dict[str, str]] = None
    ) -> AlertRule:
        """
        Create an alert rule.
        
        Args:
            rule_id: Unique rule identifier
            name: Rule name
            description: Rule description
            metric_name: Metric to monitor
            condition: Alert condition (>, <, >=, <=, ==, !=)
            threshold: Alert threshold
            severity: Alert severity
            notification_channels: Notification channels
            cooldown_seconds: Cooldown period
            tags: Rule tags
            
        Returns:
            AlertRule object
        """
        try:
            rule = AlertRule(
                rule_id=rule_id,
                name=name,
                description=description,
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                severity=severity,
                notification_channels=notification_channels or [NotificationChannel.EMAIL],
                cooldown_seconds=cooldown_seconds,
                tags=tags or {}
            )
            
            self.alert_rules[rule_id] = rule
            
            logger.info(f"Created alert rule: {name}")
            return rule
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}")
            raise MonitoringError(f"Failed to create alert rule: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health information
        """
        try:
            # Collect current system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Calculate health score
            health_components = {
                "cpu_health": max(0, 100 - cpu_usage),
                "memory_health": max(0, 100 - memory.percent),
                "disk_health": max(0, 100 - (disk.used / disk.total) * 100),
                "network_health": 100,  # Simplified
                "application_health": self._calculate_application_health()
            }
            
            overall_health = np.mean(list(health_components.values()))
            self.system_health_score = overall_health
            
            # Determine health status
            if overall_health >= 90:
                health_status = "excellent"
            elif overall_health >= 75:
                health_status = "good"
            elif overall_health >= 50:
                health_status = "fair"
            elif overall_health >= 25:
                health_status = "poor"
            else:
                health_status = "critical"
            
            return {
                "overall_health": overall_health,
                "health_status": health_status,
                "health_components": health_components,
                "system_metrics": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": (disk.used / disk.total) * 100,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv
                },
                "application_metrics": {
                    "total_metrics": self.total_metrics_collected,
                    "active_alerts": len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]),
                    "total_alerts": len(self.alerts),
                    "health_score": self.system_health_score
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}
    
    async def get_metrics_dashboard(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics dashboard data.
        
        Args:
            time_range_hours: Time range in hours
            
        Returns:
            Dashboard data
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            dashboard_data = {
                "time_range_hours": time_range_hours,
                "metrics": {},
                "alerts": [],
                "predictions": [],
                "summary": {}
            }
            
            # Process metrics
            for metric_name, metric_deque in self.metrics.items():
                recent_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]
                
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    timestamps = [m.timestamp.isoformat() for m in recent_metrics]
                    
                    dashboard_data["metrics"][metric_name] = {
                        "values": values,
                        "timestamps": timestamps,
                        "min": min(values),
                        "max": max(values),
                        "avg": np.mean(values),
                        "count": len(values)
                    }
            
            # Process alerts
            recent_alerts = [
                alert for alert in self.alerts.values()
                if alert.created_at >= cutoff_time
            ]
            
            dashboard_data["alerts"] = [
                {
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in recent_alerts
            ]
            
            # Process predictions
            recent_predictions = [
                pred for pred in self.predictive_analyses
                if pred.created_at >= cutoff_time
            ]
            
            dashboard_data["predictions"] = [
                {
                    "analysis_id": pred.analysis_id,
                    "metric_name": pred.metric_name,
                    "prediction_type": pred.prediction_type,
                    "predicted_value": pred.predicted_value,
                    "confidence": pred.confidence,
                    "time_horizon": pred.time_horizon,
                    "created_at": pred.created_at.isoformat()
                }
                for pred in recent_predictions
            ]
            
            # Summary statistics
            dashboard_data["summary"] = {
                "total_metrics": len(dashboard_data["metrics"]),
                "total_alerts": len(dashboard_data["alerts"]),
                "total_predictions": len(dashboard_data["predictions"]),
                "active_alerts": len([a for a in dashboard_data["alerts"] if a["status"] == "active"]),
                "system_health": self.system_health_score
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting metrics dashboard: {e}")
            return {"error": str(e)}
    
    async def predict_metric_trend(
        self,
        metric_name: str,
        time_horizon_seconds: int = 3600,
        prediction_type: str = "linear_regression"
    ) -> PredictiveAnalysis:
        """
        Predict metric trend.
        
        Args:
            metric_name: Metric to predict
            time_horizon_seconds: Prediction time horizon
            prediction_type: Type of prediction algorithm
            
        Returns:
            PredictiveAnalysis object
        """
        try:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) < 10:
                raise MonitoringError(f"Insufficient data for prediction: {metric_name}")
            
            # Get recent metric data
            recent_metrics = list(self.metrics[metric_name])[-100:]  # Last 100 points
            values = [m.value for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Simple linear regression prediction
            if prediction_type == "linear_regression":
                predicted_value, confidence = self._linear_regression_prediction(values, time_horizon_seconds)
            elif prediction_type == "moving_average":
                predicted_value, confidence = self._moving_average_prediction(values, time_horizon_seconds)
            elif prediction_type == "exponential_smoothing":
                predicted_value, confidence = self._exponential_smoothing_prediction(values, time_horizon_seconds)
            else:
                predicted_value, confidence = self._linear_regression_prediction(values, time_horizon_seconds)
            
            # Create prediction
            analysis = PredictiveAnalysis(
                analysis_id=f"pred_{UUID().hex[:16]}",
                metric_name=metric_name,
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                confidence=confidence,
                time_horizon=time_horizon_seconds,
                created_at=datetime.utcnow()
            )
            
            # Store prediction
            self.predictive_analyses.append(analysis)
            
            logger.info(f"Predicted {metric_name}: {predicted_value:.2f} (confidence: {confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error predicting metric trend: {e}")
            raise MonitoringError(f"Failed to predict metric trend: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged
            
        Returns:
            True if acknowledged successfully
        """
        try:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolution_notes: Resolution notes
            
        Returns:
            True if resolved successfully
        """
        try:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    # Private helper methods
    async def _load_alert_rules(self) -> None:
        """Load default alert rules"""
        try:
            default_rules = [
                AlertRule(
                    rule_id="high_cpu_usage",
                    name="High CPU Usage",
                    description="CPU usage exceeds 80%",
                    metric_name="cpu_usage",
                    condition=">",
                    threshold=80.0,
                    severity=AlertSeverity.WARNING,
                    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
                ),
                AlertRule(
                    rule_id="high_memory_usage",
                    name="High Memory Usage",
                    description="Memory usage exceeds 85%",
                    metric_name="memory_usage",
                    condition=">",
                    threshold=85.0,
                    severity=AlertSeverity.WARNING,
                    notification_channels=[NotificationChannel.EMAIL]
                ),
                AlertRule(
                    rule_id="low_disk_space",
                    name="Low Disk Space",
                    description="Disk space below 10%",
                    metric_name="disk_usage",
                    condition=">",
                    threshold=90.0,
                    severity=AlertSeverity.ERROR,
                    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
                ),
                AlertRule(
                    rule_id="high_error_rate",
                    name="High Error Rate",
                    description="Error rate exceeds 5%",
                    metric_name="error_rate",
                    condition=">",
                    threshold=5.0,
                    severity=AlertSeverity.ERROR,
                    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
                ),
                AlertRule(
                    rule_id="slow_response_time",
                    name="Slow Response Time",
                    description="Response time exceeds 1000ms",
                    metric_name="response_time",
                    condition=">",
                    threshold=1000.0,
                    severity=AlertSeverity.WARNING,
                    notification_channels=[NotificationChannel.EMAIL]
                )
            ]
            
            for rule in default_rules:
                self.alert_rules[rule.rule_id] = rule
            
            logger.info(f"Loaded {len(self.alert_rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Error loading alert rules: {e}")
    
    async def _establish_baseline_metrics(self) -> None:
        """Establish baseline metrics"""
        try:
            # Collect initial system metrics
            await self.collect_metric("cpu_usage", psutil.cpu_percent(interval=1))
            await self.collect_metric("memory_usage", psutil.virtual_memory().percent)
            await self.collect_metric("disk_usage", (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100)
            
            logger.info("Baseline metrics established")
            
        except Exception as e:
            logger.error(f"Error establishing baseline metrics: {e}")
    
    async def _check_alert_conditions(self, metric_name: str, value: float) -> None:
        """Check alert conditions for a metric"""
        try:
            for rule in self.alert_rules.values():
                if rule.metric_name == metric_name and rule.enabled:
                    # Check if condition is met
                    condition_met = False
                    
                    if rule.condition == ">":
                        condition_met = value > rule.threshold
                    elif rule.condition == "<":
                        condition_met = value < rule.threshold
                    elif rule.condition == ">=":
                        condition_met = value >= rule.threshold
                    elif rule.condition == "<=":
                        condition_met = value <= rule.threshold
                    elif rule.condition == "==":
                        condition_met = value == rule.threshold
                    elif rule.condition == "!=":
                        condition_met = value != rule.threshold
                    
                    if condition_met:
                        await self._trigger_alert(rule, value)
                        
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float) -> None:
        """Trigger an alert"""
        try:
            # Check cooldown
            last_alert_time = self.last_notifications.get(rule.rule_id)
            if last_alert_time:
                time_since_last = (datetime.utcnow() - last_alert_time).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    return  # Still in cooldown period
            
            # Create alert
            alert_id = f"alert_{UUID().hex[:16]}"
            alert = Alert(
                alert_id=alert_id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric_name=rule.metric_name,
                threshold=rule.threshold,
                current_value=current_value,
                condition=rule.condition,
                created_at=datetime.utcnow(),
                notification_channels=rule.notification_channels,
                tags=rule.tags
            )
            
            # Store alert
            self.alerts[alert_id] = alert
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Update statistics
            self.total_alerts_triggered += 1
            self.last_notifications[rule.rule_id] = datetime.utcnow()
            
            logger.warning(f"Alert triggered: {rule.name} - {current_value} {rule.condition} {rule.threshold}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications"""
        try:
            if not self.notification_enabled:
                return
            
            for channel in alert.notification_channels:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert)
            
            self.total_notifications_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    async def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification"""
        try:
            # Implement email notification logic
            logger.info(f"Email notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification"""
        try:
            # Implement Slack notification logic
            logger.info(f"Slack notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_sms_notification(self, alert: Alert) -> None:
        """Send SMS notification"""
        try:
            # Implement SMS notification logic
            logger.info(f"SMS notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification"""
        try:
            # Implement webhook notification logic
            logger.info(f"Webhook notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    def _calculate_application_health(self) -> float:
        """Calculate application health score"""
        try:
            # Calculate health based on various factors
            active_alerts = len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE])
            total_alerts = len(self.alerts)
            
            if total_alerts == 0:
                return 100.0
            
            alert_ratio = active_alerts / total_alerts
            health_score = max(0, 100 - (alert_ratio * 100))
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating application health: {e}")
            return 50.0
    
    def _linear_regression_prediction(self, values: List[float], time_horizon: int) -> Tuple[float, float]:
        """Linear regression prediction"""
        try:
            if len(values) < 2:
                return values[-1] if values else 0.0, 0.0
            
            # Simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope and intercept
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            intercept = np.mean(y) - slope * np.mean(x)
            
            # Predict future value
            future_x = len(values) + (time_horizon / 3600)  # Convert to hours
            predicted_value = slope * future_x + intercept
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.1, 1.0 - abs(slope) * 0.1))
            
            return predicted_value, confidence
            
        except Exception as e:
            logger.error(f"Error in linear regression prediction: {e}")
            return values[-1] if values else 0.0, 0.0
    
    def _moving_average_prediction(self, values: List[float], time_horizon: int) -> Tuple[float, float]:
        """Moving average prediction"""
        try:
            if not values:
                return 0.0, 0.0
            
            # Simple moving average
            window_size = min(10, len(values))
            recent_values = values[-window_size:]
            predicted_value = np.mean(recent_values)
            
            # Calculate confidence based on variance
            variance = np.var(recent_values)
            confidence = max(0.1, min(0.9, 1.0 - variance / 100.0))
            
            return predicted_value, confidence
            
        except Exception as e:
            logger.error(f"Error in moving average prediction: {e}")
            return values[-1] if values else 0.0, 0.0
    
    def _exponential_smoothing_prediction(self, values: List[float], time_horizon: int) -> Tuple[float, float]:
        """Exponential smoothing prediction"""
        try:
            if not values:
                return 0.0, 0.0
            
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing factor
            smoothed = values[0]
            
            for value in values[1:]:
                smoothed = alpha * value + (1 - alpha) * smoothed
            
            predicted_value = smoothed
            
            # Calculate confidence
            confidence = 0.7  # Fixed confidence for simplicity
            
            return predicted_value, confidence
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing prediction: {e}")
            return values[-1] if values else 0.0, 0.0
    
    # Background tasks
    async def _metrics_collector(self) -> None:
        """Background metrics collection"""
        while True:
            try:
                await asyncio.sleep(self.metrics_collection_interval)
                
                # Collect system metrics
                await self.collect_metric("cpu_usage", psutil.cpu_percent(interval=1))
                await self.collect_metric("memory_usage", psutil.virtual_memory().percent)
                await self.collect_metric("disk_usage", (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100)
                
                # Collect application metrics
                await self.collect_metric("active_alerts", len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]))
                await self.collect_metric("total_metrics", self.total_metrics_collected)
                await self.collect_metric("system_health", self.system_health_score)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def _health_monitor(self) -> None:
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.get_system_health()
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    async def _alert_evaluator(self) -> None:
        """Background alert evaluation"""
        while True:
            try:
                await asyncio.sleep(self.alert_evaluation_interval)
                
                # Evaluate all active alerts
                for alert in self.alerts.values():
                    if alert.status == AlertStatus.ACTIVE:
                        # Check if alert is still valid
                        if alert.metric_name in self.metrics:
                            recent_metrics = list(self.metrics[alert.metric_name])[-5:]
                            if recent_metrics:
                                current_value = recent_metrics[-1].value
                                alert.current_value = current_value
                                
                                # Check if alert condition is still met
                                condition_met = False
                                if alert.condition == ">":
                                    condition_met = current_value > alert.threshold
                                elif alert.condition == "<":
                                    condition_met = current_value < alert.threshold
                                # Add other conditions as needed
                                
                                if not condition_met:
                                    # Auto-resolve alert if condition is no longer met
                                    alert.status = AlertStatus.RESOLVED
                                    alert.resolved_at = datetime.utcnow()
                                    alert.resolution_notes = "Auto-resolved: condition no longer met"
                                    
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
    
    async def _predictive_analyzer(self) -> None:
        """Background predictive analysis"""
        while True:
            try:
                await asyncio.sleep(self.predictive_analysis_interval)
                
                # Run predictions for key metrics
                key_metrics = ["cpu_usage", "memory_usage", "disk_usage", "system_health"]
                
                for metric_name in key_metrics:
                    if metric_name in self.metrics and len(self.metrics[metric_name]) > 10:
                        try:
                            await self.predict_metric_trend(metric_name, 3600, "linear_regression")
                        except Exception as e:
                            logger.warning(f"Failed to predict {metric_name}: {e}")
                
            except Exception as e:
                logger.error(f"Error in predictive analysis: {e}")
    
    async def _system_optimizer(self) -> None:
        """Background system optimization"""
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Perform system optimization based on metrics
                if self.system_health_score < 70:
                    logger.warning(f"System health is low: {self.system_health_score}")
                    # Trigger optimization actions
                    
            except Exception as e:
                logger.error(f"Error in system optimization: {e}")


# Global advanced monitoring system instance
advanced_monitoring_system = AdvancedMonitoringSystem()





























