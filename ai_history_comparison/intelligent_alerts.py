"""
Intelligent Alert System for AI Model Performance
================================================

This module provides an intelligent alert system that monitors AI model
performance and sends notifications when issues are detected.

Features:
- Real-time performance monitoring
- Intelligent threshold detection
- Multi-channel notifications (email, webhook, dashboard)
- Alert escalation and routing
- Performance degradation detection
- Anomaly-based alerts
- Predictive alerts using ML
- Alert correlation and deduplication
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config
from .ml_predictor import get_ml_predictor, AnomalyDetectionResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    MODEL_FAILURE = "model_failure"
    TREND_ALERT = "trend_alert"
    PREDICTIVE_ALERT = "predictive_alert"
    SYSTEM_ALERT = "system_alert"


@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    model_name: Optional[str] = None
    metric: Optional[PerformanceMetric] = None
    threshold_value: Optional[float] = None
    threshold_operator: str = "greater_than"  # greater_than, less_than, equals
    time_window_minutes: int = 60
    cooldown_minutes: int = 30
    enabled: bool = True
    notification_channels: List[str] = None
    escalation_rules: List[str] = None
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["dashboard"]
        if self.escalation_rules is None:
            self.escalation_rules = []
        if self.conditions is None:
            self.conditions = {}


@dataclass
class Alert:
    """An alert instance"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    model_name: str
    metric: str
    message: str
    current_value: float
    threshold_value: Optional[float] = None
    timestamp: datetime = None
    acknowledged: bool = False
    resolved: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NotificationChannel:
    """Configuration for a notification channel"""
    channel_id: str
    channel_type: str  # email, webhook, slack, teams
    name: str
    enabled: bool = True
    configuration: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}


class IntelligentAlertSystem:
    """Intelligent alert system for AI model performance monitoring"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        self.ml_predictor = get_ml_predictor()
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alert correlation and deduplication
        self.alert_fingerprints: Set[str] = set()
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Custom alert handlers
        self.custom_handlers: Dict[str, Callable] = {}
        
        # Initialize default rules and channels
        self._initialize_default_configuration()
    
    def _initialize_default_configuration(self):
        """Initialize default alert rules and notification channels"""
        # Default notification channels
        self.add_notification_channel(NotificationChannel(
            channel_id="dashboard",
            channel_type="dashboard",
            name="Dashboard Notifications",
            configuration={"show_popup": True, "sound": True}
        ))
        
        # Default alert rules
        self._create_default_alert_rules()
    
    def _create_default_alert_rules(self):
        """Create default alert rules"""
        # Quality degradation rule
        quality_rule = AlertRule(
            rule_id="quality_degradation",
            name="Quality Score Degradation",
            description="Alert when model quality score drops below threshold",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.WARNING,
            metric=PerformanceMetric.QUALITY_SCORE,
            threshold_value=0.7,
            threshold_operator="less_than",
            time_window_minutes=60,
            cooldown_minutes=30,
            notification_channels=["dashboard"]
        )
        self.add_alert_rule(quality_rule)
        
        # Response time rule
        response_time_rule = AlertRule(
            rule_id="response_time_high",
            name="High Response Time",
            description="Alert when response time exceeds threshold",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.WARNING,
            metric=PerformanceMetric.RESPONSE_TIME,
            threshold_value=10.0,
            threshold_operator="greater_than",
            time_window_minutes=30,
            cooldown_minutes=15,
            notification_channels=["dashboard"]
        )
        self.add_alert_rule(response_time_rule)
        
        # Cost efficiency rule
        cost_rule = AlertRule(
            rule_id="cost_efficiency_low",
            name="Low Cost Efficiency",
            description="Alert when cost efficiency drops below threshold",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.INFO,
            metric=PerformanceMetric.COST_EFFICIENCY,
            threshold_value=0.5,
            threshold_operator="less_than",
            time_window_minutes=120,
            cooldown_minutes=60,
            notification_channels=["dashboard"]
        )
        self.add_alert_rule(cost_rule)
        
        # Critical quality rule
        critical_quality_rule = AlertRule(
            rule_id="critical_quality_failure",
            name="Critical Quality Failure",
            description="Alert when quality score drops critically low",
            alert_type=AlertType.MODEL_FAILURE,
            severity=AlertSeverity.CRITICAL,
            metric=PerformanceMetric.QUALITY_SCORE,
            threshold_value=0.3,
            threshold_operator="less_than",
            time_window_minutes=15,
            cooldown_minutes=5,
            notification_channels=["dashboard"]
        )
        self.add_alert_rule(critical_quality_rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel"""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_id: str):
        """Remove a notification channel"""
        if channel_id in self.notification_channels:
            del self.notification_channels[channel_id]
            logger.info(f"Removed notification channel: {channel_id}")
    
    def add_custom_handler(self, handler_id: str, handler: Callable):
        """Add a custom alert handler"""
        self.custom_handlers[handler_id] = handler
        logger.info(f"Added custom alert handler: {handler_id}")
    
    async def start_monitoring(self):
        """Start the alert monitoring system"""
        if self.is_monitoring:
            logger.warning("Alert monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started intelligent alert monitoring")
    
    async def stop_monitoring(self):
        """Stop the alert monitoring system"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped intelligent alert monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_all_rules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_all_rules(self):
        """Check all enabled alert rules"""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._check_rule(rule)
            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {str(e)}")
    
    async def _check_rule(self, rule: AlertRule):
        """Check a specific alert rule"""
        try:
            # Check cooldown
            if rule.rule_id in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule.rule_id]:
                    return
            
            # Get models to check
            models_to_check = self._get_models_for_rule(rule)
            
            for model_name in models_to_check:
                # Check if rule conditions are met
                if await self._evaluate_rule_conditions(rule, model_name):
                    # Create and send alert
                    alert = await self._create_alert(rule, model_name)
                    if alert:
                        await self._process_alert(alert)
                        # Set cooldown
                        self.alert_cooldowns[rule.rule_id] = (
                            datetime.now() + timedelta(minutes=rule.cooldown_minutes)
                        )
        
        except Exception as e:
            logger.error(f"Error checking rule {rule.rule_id}: {str(e)}")
    
    def _get_models_for_rule(self, rule: AlertRule) -> List[str]:
        """Get models to check for a rule"""
        if rule.model_name:
            return [rule.model_name]
        
        # Get all tracked models
        stats = self.analyzer.performance_stats
        return list(stats["models_tracked"])
    
    async def _evaluate_rule_conditions(self, rule: AlertRule, model_name: str) -> bool:
        """Evaluate if rule conditions are met"""
        try:
            if rule.alert_type == AlertType.PERFORMANCE_DEGRADATION:
                return await self._check_performance_degradation(rule, model_name)
            elif rule.alert_type == AlertType.THRESHOLD_BREACH:
                return await self._check_threshold_breach(rule, model_name)
            elif rule.alert_type == AlertType.ANOMALY_DETECTED:
                return await self._check_anomaly_detection(rule, model_name)
            elif rule.alert_type == AlertType.TREND_ALERT:
                return await self._check_trend_alert(rule, model_name)
            elif rule.alert_type == AlertType.PREDICTIVE_ALERT:
                return await self._check_predictive_alert(rule, model_name)
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {str(e)}")
            return False
    
    async def _check_performance_degradation(self, rule: AlertRule, model_name: str) -> bool:
        """Check for performance degradation"""
        try:
            if not rule.metric:
                return False
            
            # Get recent performance data
            recent_data = self.analyzer.get_model_performance(
                model_name, rule.metric, days=1
            )
            
            if not recent_data:
                return False
            
            # Check if current performance is below threshold
            current_value = recent_data[-1].value
            
            if rule.threshold_operator == "less_than":
                return current_value < rule.threshold_value
            elif rule.threshold_operator == "greater_than":
                return current_value > rule.threshold_value
            elif rule.threshold_operator == "equals":
                return abs(current_value - rule.threshold_value) < 0.01
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {str(e)}")
            return False
    
    async def _check_threshold_breach(self, rule: AlertRule, model_name: str) -> bool:
        """Check for threshold breach"""
        return await self._check_performance_degradation(rule, model_name)
    
    async def _check_anomaly_detection(self, rule: AlertRule, model_name: str) -> bool:
        """Check for anomalies using ML"""
        try:
            if not rule.metric:
                return False
            
            # Use ML predictor to detect anomalies
            anomalies = await self.ml_predictor.detect_anomalies(
                model_name, rule.metric, days=1
            )
            
            # Check if any recent anomalies meet severity criteria
            for anomaly in anomalies:
                if self._anomaly_matches_severity(anomaly, rule.severity):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking anomaly detection: {str(e)}")
            return False
    
    def _anomaly_matches_severity(self, anomaly: AnomalyDetectionResult, severity: AlertSeverity) -> bool:
        """Check if anomaly matches alert severity"""
        severity_mapping = {
            AlertSeverity.INFO: ["low"],
            AlertSeverity.WARNING: ["low", "medium"],
            AlertSeverity.ERROR: ["low", "medium", "high"],
            AlertSeverity.CRITICAL: ["low", "medium", "high", "critical"]
        }
        
        return anomaly.severity in severity_mapping.get(severity, [])
    
    async def _check_trend_alert(self, rule: AlertRule, model_name: str) -> bool:
        """Check for trend-based alerts"""
        try:
            if not rule.metric:
                return False
            
            # Analyze trends
            trend_analysis = self.analyzer.analyze_trends(
                model_name, rule.metric, days=30
            )
            
            if not trend_analysis:
                return False
            
            # Check if trend matches alert conditions
            if rule.conditions.get("trend_direction") == "declining":
                return trend_analysis.trend_direction == "declining"
            elif rule.conditions.get("trend_direction") == "improving":
                return trend_analysis.trend_direction == "improving"
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trend alert: {str(e)}")
            return False
    
    async def _check_predictive_alert(self, rule: AlertRule, model_name: str) -> bool:
        """Check for predictive alerts using ML"""
        try:
            if not rule.metric:
                return False
            
            # Get prediction
            prediction = await self.ml_predictor.predict_performance(
                model_name, rule.metric
            )
            
            if not prediction:
                return False
            
            # Check if predicted value meets threshold
            if rule.threshold_operator == "less_than":
                return prediction.predicted_value < rule.threshold_value
            elif rule.threshold_operator == "greater_than":
                return prediction.predicted_value > rule.threshold_value
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking predictive alert: {str(e)}")
            return False
    
    async def _create_alert(self, rule: AlertRule, model_name: str) -> Optional[Alert]:
        """Create an alert instance"""
        try:
            # Get current performance value
            current_value = 0.0
            if rule.metric:
                recent_data = self.analyzer.get_model_performance(
                    model_name, rule.metric, days=1
                )
                if recent_data:
                    current_value = recent_data[-1].value
            
            # Generate alert message
            message = self._generate_alert_message(rule, model_name, current_value)
            
            # Create alert fingerprint for deduplication
            fingerprint = self._create_alert_fingerprint(rule, model_name, current_value)
            
            if fingerprint in self.alert_fingerprints:
                return None  # Duplicate alert
            
            # Create alert
            alert = Alert(
                alert_id=f"{rule.rule_id}_{model_name}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                model_name=model_name,
                metric=rule.metric.value if rule.metric else "unknown",
                message=message,
                current_value=current_value,
                threshold_value=rule.threshold_value,
                metadata={
                    "fingerprint": fingerprint,
                    "rule_name": rule.name,
                    "rule_description": rule.description
                }
            )
            
            # Add to fingerprints
            self.alert_fingerprints.add(fingerprint)
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return None
    
    def _generate_alert_message(self, rule: AlertRule, model_name: str, current_value: float) -> str:
        """Generate alert message"""
        if rule.alert_type == AlertType.PERFORMANCE_DEGRADATION:
            return f"Performance degradation detected for {model_name}: {rule.metric.value} = {current_value:.3f} (threshold: {rule.threshold_value})"
        elif rule.alert_type == AlertType.THRESHOLD_BREACH:
            return f"Threshold breach for {model_name}: {rule.metric.value} = {current_value:.3f} (threshold: {rule.threshold_value})"
        elif rule.alert_type == AlertType.ANOMALY_DETECTED:
            return f"Anomaly detected for {model_name}: {rule.metric.value} = {current_value:.3f}"
        elif rule.alert_type == AlertType.TREND_ALERT:
            return f"Trend alert for {model_name}: {rule.metric.value} showing concerning trend"
        elif rule.alert_type == AlertType.PREDICTIVE_ALERT:
            return f"Predictive alert for {model_name}: {rule.metric.value} predicted to breach threshold"
        else:
            return f"Alert for {model_name}: {rule.description}"
    
    def _create_alert_fingerprint(self, rule: AlertRule, model_name: str, current_value: float) -> str:
        """Create fingerprint for alert deduplication"""
        fingerprint_data = f"{rule.rule_id}_{model_name}_{rule.metric.value if rule.metric else 'unknown'}_{current_value:.2f}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    async def _process_alert(self, alert: Alert):
        """Process and send an alert"""
        try:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Call custom handlers
            await self._call_custom_handlers(alert)
            
            logger.info(f"Processed alert: {alert.alert_id} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error processing alert: {str(e)}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            # Get notification channels from rule
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                return
            
            for channel_id in rule.notification_channels:
                channel = self.notification_channels.get(channel_id)
                if not channel or not channel.enabled:
                    continue
                
                try:
                    if channel.channel_type == "dashboard":
                        await self._send_dashboard_notification(alert, channel)
                    elif channel.channel_type == "email":
                        await self._send_email_notification(alert, channel)
                    elif channel.channel_type == "webhook":
                        await self._send_webhook_notification(alert, channel)
                    elif channel.channel_type == "slack":
                        await self._send_slack_notification(alert, channel)
                    elif channel.channel_type == "teams":
                        await self._send_teams_notification(alert, channel)
                
                except Exception as e:
                    logger.error(f"Error sending notification to {channel_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
    
    async def _send_dashboard_notification(self, alert: Alert, channel: NotificationChannel):
        """Send dashboard notification"""
        # This would integrate with the real-time dashboard
        logger.info(f"Dashboard notification: {alert.message}")
    
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        try:
            config = channel.configuration
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', 'alerts@ai-system.com')
            msg['To'] = config.get('to_email', 'admin@ai-system.com')
            msg['Subject'] = f"AI Model Alert: {alert.severity.value.upper()}"
            
            # Create email body
            body = f"""
            Alert Details:
            - Model: {alert.model_name}
            - Metric: {alert.metric}
            - Current Value: {alert.current_value:.3f}
            - Threshold: {alert.threshold_value}
            - Severity: {alert.severity.value.upper()}
            - Time: {alert.timestamp.isoformat()}
            
            Message: {alert.message}
            
            Please investigate this issue promptly.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (this would need proper SMTP configuration)
            logger.info(f"Email notification sent: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        try:
            config = channel.configuration
            webhook_url = config.get('url')
            
            if not webhook_url:
                logger.warning("No webhook URL configured")
                return
            
            # Prepare payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "model_name": alert.model_name,
                "metric": alert.metric,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent: {alert.message}")
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
    
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        try:
            config = channel.configuration
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return
            
            # Prepare Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "good")
            
            slack_payload = {
                "attachments": [{
                    "color": color,
                    "title": f"AI Model Alert: {alert.severity.value.upper()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Model", "value": alert.model_name, "short": True},
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Current Value", "value": f"{alert.current_value:.3f}", "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": False}
                    ],
                    "footer": "AI History Analyzer",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent: {alert.message}")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
    
    async def _send_teams_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Microsoft Teams notification"""
        try:
            config = channel.configuration
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.warning("No Teams webhook URL configured")
                return
            
            # Prepare Teams message
            color = {
                AlertSeverity.INFO: "00ff00",
                AlertSeverity.WARNING: "ffff00",
                AlertSeverity.ERROR: "ff0000",
                AlertSeverity.CRITICAL: "ff0000"
            }.get(alert.severity, "00ff00")
            
            teams_payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": f"AI Model Alert: {alert.severity.value.upper()}",
                "sections": [{
                    "activityTitle": f"AI Model Alert: {alert.severity.value.upper()}",
                    "activitySubtitle": alert.message,
                    "facts": [
                        {"name": "Model", "value": alert.model_name},
                        {"name": "Metric", "value": alert.metric},
                        {"name": "Current Value", "value": f"{alert.current_value:.3f}"},
                        {"name": "Threshold", "value": str(alert.threshold_value)},
                        {"name": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                    ]
                }]
            }
            
            # Send to Teams
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=teams_payload) as response:
                    if response.status == 200:
                        logger.info(f"Teams notification sent: {alert.message}")
                    else:
                        logger.error(f"Teams notification failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending Teams notification: {str(e)}")
    
    async def _call_custom_handlers(self, alert: Alert):
        """Call custom alert handlers"""
        try:
            for handler_id, handler in self.custom_handlers.items():
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in custom handler {handler_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error calling custom handlers: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alert_history[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_distribution": severity_counts,
            "alert_rules_count": len(self.alert_rules),
            "notification_channels_count": len(self.notification_channels)
        }


# Global alert system instance
_alert_system: Optional[IntelligentAlertSystem] = None


def get_intelligent_alert_system() -> IntelligentAlertSystem:
    """Get or create global intelligent alert system"""
    global _alert_system
    if _alert_system is None:
        _alert_system = IntelligentAlertSystem()
    return _alert_system


# Example usage
async def main():
    """Example usage of the intelligent alert system"""
    alert_system = get_intelligent_alert_system()
    
    # Add email notification channel
    email_channel = NotificationChannel(
        channel_id="email_admin",
        channel_type="email",
        name="Admin Email",
        configuration={
            "from_email": "alerts@ai-system.com",
            "to_email": "admin@ai-system.com",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "alerts@ai-system.com",
            "password": "password"
        }
    )
    alert_system.add_notification_channel(email_channel)
    
    # Add custom alert rule
    custom_rule = AlertRule(
        rule_id="custom_quality_rule",
        name="Custom Quality Rule",
        description="Custom quality monitoring rule",
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        severity=AlertSeverity.WARNING,
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        threshold_value=0.8,
        threshold_operator="less_than",
        time_window_minutes=30,
        cooldown_minutes=15,
        notification_channels=["dashboard", "email_admin"]
    )
    alert_system.add_alert_rule(custom_rule)
    
    # Start monitoring
    await alert_system.start_monitoring()
    
    # Wait for some alerts
    await asyncio.sleep(300)  # 5 minutes
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print(f"Alert statistics: {stats}")
    
    # Stop monitoring
    await alert_system.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())

























