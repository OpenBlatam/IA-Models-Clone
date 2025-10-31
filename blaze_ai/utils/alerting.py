"""
Intelligent Alerting System for Blaze AI

This module provides advanced alerting capabilities with configurable rules,
thresholds, and multiple notification channels for production monitoring.
"""

from __future__ import annotations

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..core.interfaces import CoreConfig
from .logging import get_logger
from .metrics import AdvancedMetricsCollector

# =============================================================================
# Alert Types and Severities
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"

class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"

# =============================================================================
# Alert Rules and Definitions
# =============================================================================

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Python expression or metric query
    threshold: Union[float, int, str]
    comparison: str  # >, <, >=, <=, ==, !=, contains, regex
    duration: float = 0.0  # How long condition must be true before alerting
    cooldown: float = 300.0  # Cooldown period between alerts
    labels: List[str] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """Validate alert rule."""
        if self.duration < 0:
            raise ValueError("Duration must be non-negative")
        if self.cooldown < 0:
            raise ValueError("Cooldown must be non-negative")
        if self.comparison not in ['>', '<', '>=', '<=', '==', '!=', 'contains', 'regex']:
            raise ValueError(f"Invalid comparison operator: {self.comparison}")

@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    severity: AlertSeverity
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    labels: List[str] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[Union[float, int, str]] = None
    threshold: Optional[Union[float, int, str]] = None

# =============================================================================
# Notification Providers
# =============================================================================

class NotificationProvider:
    """Base class for notification providers."""
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """Send notification for an alert."""
        raise NotImplementedError

class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """Send email notification."""
        try:
            if not self.to_emails:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            body = f"""
Alert Details:
- Rule: {alert.rule_name}
- Severity: {alert.severity.value}
- Message: {message}
- Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.created_at))}
- Status: {alert.status.value}

Alert Information:
- Labels: {', '.join(alert.labels) if alert.labels else 'None'}
- Annotations: {json.dumps(alert.annotations, indent=2) if alert.annotations else 'None'}
- Current Value: {alert.value}
- Threshold: {alert.threshold}

This is an automated alert from the Blaze AI monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            return True
            
        except Exception as e:
            get_logger("email_notification").error(f"Failed to send email notification: {e}")
            return False

class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Blaze AI Monitor')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """Send Slack notification."""
        try:
            if not self.webhook_url:
                return False
            
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            slack_message = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [{
                    "color": color_map.get(alert.severity, "#000000"),
                    "title": f"Alert: {alert.rule_name}",
                    "text": message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": alert.status.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Created",
                            "value": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.created_at)),
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": str(alert.value) if alert.value is not None else "N/A",
                            "short": True
                        }
                    ],
                    "footer": "Blaze AI Monitoring System"
                }]
            }
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=slack_message,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            get_logger("slack_notification").error(f"Failed to send Slack notification: {e}")
            return False

class WebhookNotificationProvider(NotificationProvider):
    """Webhook notification provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """Send webhook notification."""
        try:
            if not self.webhook_url:
                return False
            
            # Prepare webhook payload
            payload = {
                "alert": {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": message,
                    "status": alert.status.value,
                    "created_at": alert.created_at,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "value": alert.value,
                    "threshold": alert.threshold
                },
                "timestamp": time.time(),
                "source": "blaze_ai_monitoring"
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            get_logger("webhook_notification").error(f"Failed to send webhook notification: {e}")
            return False

# =============================================================================
# Intelligent Alerting Engine
# =============================================================================

class IntelligentAlertingEngine:
    """Intelligent alerting engine with rule evaluation and notification."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config
        self.logger = get_logger("alerting_engine")
        
        # Alert rules and instances
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification providers
        self.notification_providers: Dict[NotificationChannel, NotificationProvider] = {}
        
        # Metrics collector
        self.metrics_collector: Optional[AdvancedMetricsCollector] = None
        
        # Background processing
        self._evaluation_task: Optional[asyncio.Task] = None
        self._notification_task: Optional[asyncio.Task] = None
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background alert evaluation and notification tasks."""
        self._evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
        self._notification_task = asyncio.create_task(self._notification_loop())
    
    async def _alert_evaluation_loop(self):
        """Background loop for evaluating alert rules."""
        while True:
            try:
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                await self._evaluate_all_rules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _notification_loop(self):
        """Background loop for sending notifications."""
        while True:
            try:
                await asyncio.sleep(10)  # Check notifications every 10 seconds
                await self._process_notifications()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Notification processing error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                self.logger.error(f"Failed to evaluate rule {rule_name}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        try:
            # Get current metric value
            current_value = await self._get_metric_value(rule.condition)
            if current_value is None:
                return
            
            # Check if condition is met
            condition_met = self._check_condition(current_value, rule.threshold, rule.comparison)
            
            if condition_met:
                # Check if alert should be triggered
                if await self._should_trigger_alert(rule):
                    await self._trigger_alert(rule, current_value)
            else:
                # Check if alert should be resolved
                if rule.name in self.active_alerts:
                    await self._resolve_alert(rule.name)
                    
        except Exception as e:
            self.logger.error(f"Rule evaluation failed for {rule.name}: {e}")
    
    async def _get_metric_value(self, condition: str) -> Optional[Union[float, int, str]]:
        """Get current value for a metric condition."""
        try:
            # This is a simplified implementation
            # In a real system, you would parse the condition and query metrics
            if self.metrics_collector:
                # For now, return a mock value
                return 75.0  # Mock metric value
            return None
        except Exception as e:
            self.logger.error(f"Failed to get metric value: {e}")
            return None
    
    def _check_condition(self, value: Union[float, int, str], 
                        threshold: Union[float, int, str], 
                        comparison: str) -> bool:
        """Check if a condition is met."""
        try:
            if comparison == '>':
                return value > threshold
            elif comparison == '<':
                return value < threshold
            elif comparison == '>=':
                return value >= threshold
            elif comparison == '<=':
                return value <= threshold
            elif comparison == '==':
                return value == threshold
            elif comparison == '!=':
                return value != threshold
            elif comparison == 'contains':
                return str(threshold) in str(value)
            elif comparison == 'regex':
                import re
                return bool(re.search(str(threshold), str(value)))
            else:
                return False
        except Exception:
            return False
    
    async def _should_trigger_alert(self, rule: AlertRule) -> bool:
        """Check if an alert should be triggered."""
        try:
            # Check cooldown
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                if time.time() - alert.updated_at < rule.cooldown:
                    return False
            
            # Check duration (simplified)
            # In a real system, you would track how long the condition has been true
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check alert trigger: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, value: Union[float, int, str]):
        """Trigger a new alert."""
        try:
            # Create alert
            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Alert triggered: {rule.description}",
                labels=rule.labels.copy(),
                annotations=rule.annotations.copy(),
                value=value,
                threshold=rule.threshold
            )
            
            # Store alert
            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)
            
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter(
                    f"alerts_triggered_total",
                    1.0,
                    [f"severity:{rule.severity.value}", f"rule:{rule.name}"]
                )
            
            self.logger.info(f"Alert triggered: {rule.name} ({rule.severity.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert {rule.name}: {e}")
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        try:
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.updated_at = time.time()
                
                # Update metrics
                if self.metrics_collector:
                    self.metrics_collector.increment_counter(
                        f"alerts_resolved_total",
                        1.0,
                        [f"severity:{alert.severity.value}", f"rule:{rule_name}"]
                    )
                
                self.logger.info(f"Alert resolved: {rule_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {rule_name}: {e}")
    
    async def _process_notifications(self):
        """Process pending notifications."""
        try:
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    await self._send_notifications(alert)
                    
        except Exception as e:
            self.logger.error(f"Failed to process notifications: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        try:
            message = f"Alert: {alert.message} (Value: {alert.value}, Threshold: {alert.threshold})"
            
            for channel, provider in self.notification_providers.items():
                try:
                    success = await provider.send_notification(alert, message)
                    if success:
                        self.logger.debug(f"Notification sent via {channel.value}")
                    else:
                        self.logger.warning(f"Failed to send notification via {channel.value}")
                except Exception as e:
                    self.logger.error(f"Notification provider {channel.value} error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {e}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        try:
            self.alert_rules[rule.name] = rule
            self.logger.info(f"Alert rule added: {rule.name}")
        except Exception as e:
            self.logger.error(f"Failed to add alert rule {rule.name}: {e}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        try:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                self.logger.info(f"Alert rule removed: {rule_name}")
        except Exception as e:
            self.logger.error(f"Failed to remove alert rule {rule_name}: {e}")
    
    def add_notification_provider(self, channel: NotificationChannel, provider: NotificationProvider):
        """Add a notification provider."""
        try:
            self.notification_providers[channel] = provider
            self.logger.info(f"Notification provider added: {channel.value}")
        except Exception as e:
            self.logger.error(f"Failed to add notification provider {channel.value}: {e}")
    
    def acknowledge_alert(self, rule_name: str):
        """Acknowledge an active alert."""
        try:
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.updated_at = time.time()
                self.logger.info(f"Alert acknowledged: {rule_name}")
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {rule_name}: {e}")
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get a summary of all alerts."""
        summary = {
            "total_rules": len(self.alert_rules),
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "alerts_by_severity": {},
            "alerts_by_status": {}
        }
        
        # Count by severity
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            summary["alerts_by_severity"][severity] = summary["alerts_by_severity"].get(severity, 0) + 1
        
        # Count by status
        for alert in self.active_alerts.values():
            status = alert.status.value
            summary["alerts_by_status"][status] = summary["alerts_by_status"].get(status, 0) + 1
        
        return summary
    
    async def shutdown(self):
        """Shutdown the alerting engine."""
        self.logger.info("Shutting down alerting engine...")
        
        # Cancel background tasks
        for task in [self._evaluation_task, self._notification_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Alerting engine shutdown complete")

# =============================================================================
# Global Alerting Instance
# =============================================================================

_global_alerting_engine: Optional[IntelligentAlertingEngine] = None

def get_alerting_engine(config: Optional[CoreConfig] = None) -> IntelligentAlertingEngine:
    """Get the global alerting engine instance."""
    global _global_alerting_engine
    if _global_alerting_engine is None:
        _global_alerting_engine = IntelligentAlertingEngine(config)
    return _global_alerting_engine

async def shutdown_alerting_engine():
    """Shutdown the global alerting engine."""
    global _global_alerting_engine
    if _global_alerting_engine:
        await _global_alerting_engine.shutdown()
        _global_alerting_engine = None


