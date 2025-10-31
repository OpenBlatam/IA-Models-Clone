"""
Ultimate BUL System - Advanced Alerting & Notification System
Comprehensive alerting with intelligent routing, escalation, and notification management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import smtplib
import aiohttp
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

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

class NotificationChannel(str, Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TEAMS = "teams"

class AlertRule(str, Enum):
    """Alert rules"""
    THRESHOLD = "threshold"
    RATE_CHANGE = "rate_change"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    ABSENCE = "absence"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationRule:
    """Notification rule configuration"""
    id: str
    name: str
    alert_severity: List[AlertSeverity]
    alert_sources: List[str]
    channels: List[NotificationChannel]
    recipients: List[str]
    escalation_delay: int = 0  # minutes
    escalation_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_recipients: List[str] = field(default_factory=list)
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    rule_type: AlertRule
    metric: str
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    duration: int  # seconds
    severity: AlertSeverity
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)

class AdvancedAlertingSystem:
    """Advanced alerting and notification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts = []
        self.alert_rules = self._initialize_alert_rules()
        self.notification_rules = self._initialize_notification_rules()
        self.alert_history = []
        self.suppressed_alerts = set()
        
        # Notification channels
        self.notification_channels = self._initialize_notification_channels()
        
        # Alert processing
        self.processing_active = False
        self.alert_queue = asyncio.Queue()
        
        # Start alert processing
        self.start_alert_processing()
    
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize alert rules"""
        return [
            # System Health Rules
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                rule_type=AlertRule.THRESHOLD,
                metric="cpu_usage_percent",
                threshold=80.0,
                comparison=">",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                tags={"component": "system", "type": "performance"}
            ),
            AlertRule(
                id="critical_cpu_usage",
                name="Critical CPU Usage",
                rule_type=AlertRule.THRESHOLD,
                metric="cpu_usage_percent",
                threshold=95.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.CRITICAL,
                tags={"component": "system", "type": "performance"}
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                rule_type=AlertRule.THRESHOLD,
                metric="memory_usage_percent",
                threshold=85.0,
                comparison=">",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                tags={"component": "system", "type": "performance"}
            ),
            AlertRule(
                id="critical_memory_usage",
                name="Critical Memory Usage",
                rule_type=AlertRule.THRESHOLD,
                metric="memory_usage_percent",
                threshold=95.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.CRITICAL,
                tags={"component": "system", "type": "performance"}
            ),
            
            # API Performance Rules
            AlertRule(
                id="high_response_time",
                name="High Response Time",
                rule_type=AlertRule.THRESHOLD,
                metric="response_time_seconds",
                threshold=5.0,
                comparison=">",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                tags={"component": "api", "type": "performance"}
            ),
            AlertRule(
                id="critical_response_time",
                name="Critical Response Time",
                rule_type=AlertRule.THRESHOLD,
                metric="response_time_seconds",
                threshold=10.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.CRITICAL,
                tags={"component": "api", "type": "performance"}
            ),
            AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                rule_type=AlertRule.THRESHOLD,
                metric="error_rate_percent",
                threshold=5.0,
                comparison=">",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                tags={"component": "api", "type": "reliability"}
            ),
            AlertRule(
                id="critical_error_rate",
                name="Critical Error Rate",
                rule_type=AlertRule.THRESHOLD,
                metric="error_rate_percent",
                threshold=10.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.CRITICAL,
                tags={"component": "api", "type": "reliability"}
            ),
            
            # Database Rules
            AlertRule(
                id="database_connection_failure",
                name="Database Connection Failure",
                rule_type=AlertRule.ABSENCE,
                metric="database_connections_active",
                threshold=1.0,
                comparison="<",
                duration=30,  # 30 seconds
                severity=AlertSeverity.CRITICAL,
                tags={"component": "database", "type": "connectivity"}
            ),
            AlertRule(
                id="high_database_connections",
                name="High Database Connections",
                rule_type=AlertRule.THRESHOLD,
                metric="database_connections_active",
                threshold=80.0,
                comparison=">",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                tags={"component": "database", "type": "performance"}
            ),
            
            # Security Rules
            AlertRule(
                id="brute_force_attack",
                name="Brute Force Attack",
                rule_type=AlertRule.THRESHOLD,
                metric="failed_login_attempts",
                threshold=5.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.HIGH,
                tags={"component": "security", "type": "attack"}
            ),
            AlertRule(
                id="suspicious_api_usage",
                name="Suspicious API Usage",
                rule_type=AlertRule.THRESHOLD,
                metric="api_requests_per_minute",
                threshold=1000.0,
                comparison=">",
                duration=60,  # 1 minute
                severity=AlertSeverity.WARNING,
                tags={"component": "security", "type": "abuse"}
            )
        ]
    
    def _initialize_notification_rules(self) -> List[NotificationRule]:
        """Initialize notification rules"""
        return [
            # Emergency notifications
            NotificationRule(
                id="emergency_notifications",
                name="Emergency Notifications",
                alert_severity=[AlertSeverity.EMERGENCY, AlertSeverity.CRITICAL],
                alert_sources=["system", "api", "database", "security"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                recipients=["admin@bul.local", "oncall@bul.local"],
                escalation_delay=15,  # 15 minutes
                escalation_channels=[NotificationChannel.SMS, NotificationChannel.PAGERDUTY],
                escalation_recipients=["manager@bul.local", "cto@bul.local"]
            ),
            
            # Warning notifications
            NotificationRule(
                id="warning_notifications",
                name="Warning Notifications",
                alert_severity=[AlertSeverity.WARNING, AlertSeverity.ERROR],
                alert_sources=["system", "api", "database"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                recipients=["devops@bul.local", "team@bul.local"],
                escalation_delay=60,  # 1 hour
                escalation_channels=[NotificationChannel.EMAIL],
                escalation_recipients=["manager@bul.local"]
            ),
            
            # Info notifications
            NotificationRule(
                id="info_notifications",
                name="Info Notifications",
                alert_severity=[AlertSeverity.INFO],
                alert_sources=["system", "api", "database", "security"],
                channels=[NotificationChannel.SLACK],
                recipients=["team@bul.local"]
            )
        ]
    
    def _initialize_notification_channels(self) -> Dict[NotificationChannel, Dict[str, Any]]:
        """Initialize notification channels"""
        return {
            NotificationChannel.EMAIL: {
                "smtp_server": self.config.get("smtp_server", "smtp.gmail.com"),
                "smtp_port": self.config.get("smtp_port", 587),
                "username": self.config.get("email_username"),
                "password": self.config.get("email_password"),
                "from_address": self.config.get("from_address", "alerts@bul.local")
            },
            NotificationChannel.SLACK: {
                "webhook_url": self.config.get("slack_webhook_url"),
                "channel": self.config.get("slack_channel", "#alerts"),
                "username": self.config.get("slack_username", "BUL Alerts")
            },
            NotificationChannel.WEBHOOK: {
                "url": self.config.get("webhook_url"),
                "headers": self.config.get("webhook_headers", {}),
                "timeout": self.config.get("webhook_timeout", 30)
            },
            NotificationChannel.PAGERDUTY: {
                "integration_key": self.config.get("pagerduty_integration_key"),
                "api_url": "https://events.pagerduty.com/v2/enqueue"
            },
            NotificationChannel.SMS: {
                "provider": self.config.get("sms_provider", "twilio"),
                "account_sid": self.config.get("sms_account_sid"),
                "auth_token": self.config.get("sms_auth_token"),
                "from_number": self.config.get("sms_from_number")
            }
        }
    
    async def start_alert_processing(self):
        """Start alert processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        logger.info("Starting alert processing")
        
        # Start processing tasks
        asyncio.create_task(self._process_alert_queue())
        asyncio.create_task(self._evaluate_alert_rules())
        asyncio.create_task(self._handle_escalations())
        asyncio.create_task(self._cleanup_old_alerts())
    
    async def stop_alert_processing(self):
        """Stop alert processing"""
        self.processing_active = False
        logger.info("Stopping alert processing")
    
    async def _process_alert_queue(self):
        """Process alerts from queue"""
        while self.processing_active:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Process alert
                await self._process_alert(alert)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics"""
        while self.processing_active:
            try:
                # This would typically fetch metrics from Prometheus or other sources
                # For now, we'll simulate metric evaluation
                await self._evaluate_system_metrics()
                await self._evaluate_api_metrics()
                await self._evaluate_database_metrics()
                await self._evaluate_security_metrics()
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error evaluating alert rules: {e}")
                await asyncio.sleep(60)
    
    async def _handle_escalations(self):
        """Handle alert escalations"""
        while self.processing_active:
            try:
                # Check for alerts that need escalation
                current_time = datetime.utcnow()
                
                for alert in self.alerts:
                    if alert.status == AlertStatus.ACTIVE:
                        # Check if alert needs escalation
                        for rule in self.notification_rules:
                            if (alert.severity in rule.alert_severity and 
                                alert.source in rule.alert_sources and
                                rule.escalation_delay > 0):
                                
                                escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_delay)
                                
                                if current_time >= escalation_time:
                                    await self._escalate_alert(alert, rule)
                
                await asyncio.sleep(60)  # Check escalations every minute
                
            except Exception as e:
                logger.error(f"Error handling escalations: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_alerts(self):
        """Cleanup old alerts"""
        while self.processing_active:
            try:
                # Remove alerts older than 7 days
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                old_alerts = [a for a in self.alerts if a.timestamp < cutoff_time]
                for alert in old_alerts:
                    self.alerts.remove(alert)
                    self.alert_history.append(alert)
                
                # Keep only last 1000 alerts in history
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old alerts: {e}")
                await asyncio.sleep(3600)
    
    async def _evaluate_system_metrics(self):
        """Evaluate system metrics against alert rules"""
        # This would typically fetch from Prometheus
        # For now, simulate some metrics
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Check CPU usage rules
        for rule in self.alert_rules:
            if rule.metric == "cpu_usage_percent" and rule.enabled:
                if self._evaluate_rule(rule, cpu_percent):
                    await self._create_alert(rule, cpu_percent, "system")
        
        # Check memory usage rules
        for rule in self.alert_rules:
            if rule.metric == "memory_usage_percent" and rule.enabled:
                if self._evaluate_rule(rule, memory.percent):
                    await self._create_alert(rule, memory.percent, "system")
    
    async def _evaluate_api_metrics(self):
        """Evaluate API metrics against alert rules"""
        # This would typically fetch from Prometheus
        # For now, simulate some metrics
        pass
    
    async def _evaluate_database_metrics(self):
        """Evaluate database metrics against alert rules"""
        # This would typically fetch from Prometheus
        # For now, simulate some metrics
        pass
    
    async def _evaluate_security_metrics(self):
        """Evaluate security metrics against alert rules"""
        # This would typically fetch from security monitoring
        # For now, simulate some metrics
        pass
    
    def _evaluate_rule(self, rule: AlertRule, value: float) -> bool:
        """Evaluate a rule against a value"""
        if rule.comparison == ">":
            return value > rule.threshold
        elif rule.comparison == "<":
            return value < rule.threshold
        elif rule.comparison == ">=":
            return value >= rule.threshold
        elif rule.comparison == "<=":
            return value <= rule.threshold
        elif rule.comparison == "==":
            return value == rule.threshold
        elif rule.comparison == "!=":
            return value != rule.threshold
        return False
    
    async def _create_alert(self, rule: AlertRule, value: float, source: str):
        """Create a new alert"""
        alert_id = f"{rule.id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            title=rule.name,
            description=f"{rule.metric} is {value} (threshold: {rule.threshold})",
            severity=rule.severity,
            source=source,
            metric=rule.metric,
            value=value,
            threshold=rule.threshold,
            timestamp=datetime.utcnow(),
            tags=rule.tags
        )
        
        # Add to queue for processing
        await self.alert_queue.put(alert)
    
    async def _process_alert(self, alert: Alert):
        """Process a new alert"""
        # Check if alert is suppressed
        if alert.id in self.suppressed_alerts:
            return
        
        # Add to active alerts
        self.alerts.append(alert)
        
        # Find matching notification rules
        matching_rules = []
        for rule in self.notification_rules:
            if (rule.enabled and 
                alert.severity in rule.alert_severity and 
                alert.source in rule.alert_sources):
                matching_rules.append(rule)
        
        # Send notifications
        for rule in matching_rules:
            await self._send_notifications(alert, rule)
        
        logger.info(f"Processed alert: {alert.title} ({alert.severity.value})")
    
    async def _send_notifications(self, alert: Alert, rule: NotificationRule):
        """Send notifications for an alert"""
        for channel in rule.channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert, rule.recipients)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert)
                elif channel == NotificationChannel.PAGERDUTY:
                    await self._send_pagerduty_notification(alert)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(alert, rule.recipients)
                
            except Exception as e:
                logger.error(f"Error sending {channel.value} notification: {e}")
    
    async def _send_email_notification(self, alert: Alert, recipients: List[str]):
        """Send email notification"""
        try:
            channel_config = self.notification_channels[NotificationChannel.EMAIL]
            
            msg = MimeMultipart()
            msg['From'] = channel_config["from_address"]
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value.upper()}
            Source: {alert.source}
            Metric: {alert.metric}
            Value: {alert.value}
            Threshold: {alert.threshold}
            Time: {alert.timestamp.isoformat()}
            
            Description: {alert.description}
            
            Tags: {json.dumps(alert.tags, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(channel_config["smtp_server"], channel_config["smtp_port"])
            server.starttls()
            server.login(channel_config["username"], channel_config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            channel_config = self.notification_channels[NotificationChannel.SLACK]
            webhook_url = channel_config["webhook_url"]
            
            if not webhook_url:
                return
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": channel_config["channel"],
                "username": channel_config["username"],
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "good"),
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Metric", "value": alert.metric, "short": True},
                            {"title": "Value", "value": str(alert.value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                        ],
                        "footer": "BUL Alerting System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            channel_config = self.notification_channels[NotificationChannel.WEBHOOK]
            webhook_url = channel_config["url"]
            
            if not webhook_url:
                return
            
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "source": alert.source,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "tags": alert.tags,
                "metadata": alert.metadata
            }
            
            headers = channel_config.get("headers", {})
            timeout = channel_config.get("timeout", 30)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.id}")
                    else:
                        logger.error(f"Failed to send webhook notification: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_pagerduty_notification(self, alert: Alert):
        """Send PagerDuty notification"""
        try:
            channel_config = self.notification_channels[NotificationChannel.PAGERDUTY]
            integration_key = channel_config["integration_key"]
            api_url = channel_config["api_url"]
            
            if not integration_key:
                return
            
            # Determine severity mapping
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical",
                AlertSeverity.EMERGENCY: "critical"
            }
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.id,
                "payload": {
                    "summary": alert.title,
                    "severity": severity_map.get(alert.severity, "info"),
                    "source": alert.source,
                    "custom_details": {
                        "description": alert.description,
                        "metric": alert.metric,
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "tags": alert.tags
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty notification sent for alert {alert.id}")
                    else:
                        logger.error(f"Failed to send PagerDuty notification: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
    
    async def _send_sms_notification(self, alert: Alert, recipients: List[str]):
        """Send SMS notification"""
        try:
            channel_config = self.notification_channels[NotificationChannel.SMS]
            
            # This would typically use a service like Twilio
            # For now, just log the notification
            message = f"[{alert.severity.value.upper()}] {alert.title}: {alert.description}"
            
            for recipient in recipients:
                logger.info(f"SMS notification to {recipient}: {message}")
                
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    async def _escalate_alert(self, alert: Alert, rule: NotificationRule):
        """Escalate an alert"""
        logger.info(f"Escalating alert {alert.id} using rule {rule.name}")
        
        # Send escalation notifications
        for channel in rule.escalation_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert, rule.escalation_recipients)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == NotificationChannel.PAGERDUTY:
                    await self._send_pagerduty_notification(alert)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(alert, rule.escalation_recipients)
                
            except Exception as e:
                logger.error(f"Error sending escalation {channel.value} notification: {e}")
    
    # Public methods
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = [a for a in self.alerts if a.status == AlertStatus.ACTIVE]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        alert = next((a for a in self.alerts if a.id == alert_id), None)
        
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = user
            alert.acknowledged_at = datetime.utcnow()
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert"""
        alert = next((a for a in self.alerts if a.id == alert_id), None)
        
        if alert and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            logger.info(f"Alert {alert_id} resolved by {user}")
            return True
        
        return False
    
    def suppress_alert(self, alert_id: str, duration_hours: int = 24) -> bool:
        """Suppress an alert"""
        alert = next((a for a in self.alerts if a.id == alert_id), None)
        
        if alert:
            self.suppressed_alerts.add(alert_id)
            
            # Remove from suppressed alerts after duration
            asyncio.create_task(self._unsuppress_alert(alert_id, duration_hours))
            
            logger.info(f"Alert {alert_id} suppressed for {duration_hours} hours")
            return True
        
        return False
    
    async def _unsuppress_alert(self, alert_id: str, duration_hours: int):
        """Unsuppress an alert after duration"""
        await asyncio.sleep(duration_hours * 3600)
        self.suppressed_alerts.discard(alert_id)
        logger.info(f"Alert {alert_id} unsuppressed")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alerts) + len(self.alert_history)
        active_alerts = len([a for a in self.alerts if a.status == AlertStatus.ACTIVE])
        acknowledged_alerts = len([a for a in self.alerts if a.status == AlertStatus.ACKNOWLEDGED])
        resolved_alerts = len([a for a in self.alerts if a.status == AlertStatus.RESOLVED])
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            count = len([a for a in self.alerts if a.severity == severity])
            severity_counts[severity.value] = count
        
        # Count by source
        source_counts = {}
        for alert in self.alerts:
            source = alert.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "resolved_alerts": resolved_alerts,
            "severity_counts": severity_counts,
            "source_counts": source_counts,
            "suppressed_alerts": len(self.suppressed_alerts)
        }
    
    def export_alert_data(self) -> Dict[str, Any]:
        """Export alert data for analysis"""
        return {
            "active_alerts": [a.__dict__ for a in self.alerts],
            "alert_history": [a.__dict__ for a in self.alert_history[-1000:]],
            "alert_rules": [r.__dict__ for r in self.alert_rules],
            "notification_rules": [r.__dict__ for r in self.notification_rules],
            "statistics": self.get_alert_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global alerting system instance
alerting_system = None

def get_alerting_system() -> AdvancedAlertingSystem:
    """Get the global alerting system instance"""
    global alerting_system
    if alerting_system is None:
        config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_username": os.getenv("EMAIL_USERNAME"),
            "email_password": os.getenv("EMAIL_PASSWORD"),
            "from_address": "alerts@bul.local",
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
            "slack_channel": "#alerts",
            "pagerduty_integration_key": os.getenv("PAGERDUTY_INTEGRATION_KEY")
        }
        alerting_system = AdvancedAlertingSystem(config)
    return alerting_system

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_username": "alerts@bul.local",
            "email_password": "password",
            "from_address": "alerts@bul.local",
            "slack_webhook_url": "https://hooks.slack.com/services/...",
            "slack_channel": "#alerts"
        }
        
        alerting = AdvancedAlertingSystem(config)
        
        # Wait for some alerts to be processed
        await asyncio.sleep(60)
        
        # Get alert statistics
        stats = alerting.get_alert_statistics()
        print("Alert Statistics:")
        print(json.dumps(stats, indent=2))
        
        await alerting.stop_alert_processing()
    
    asyncio.run(main())













