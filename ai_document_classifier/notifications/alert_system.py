"""
Alert and Notification System
=============================

Advanced alert system for monitoring system health, performance issues,
and important events in the AI Document Classifier.
"""

import logging
import smtplib
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
from pathlib import Path
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    CLASSIFICATION_ERROR = "classification_error"
    EXTERNAL_SERVICE = "external_service"
    SECURITY = "security"
    CAPACITY = "capacity"
    CUSTOM = "custom"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    description: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    alert_type: AlertType
    enabled: bool = True
    cooldown_minutes: int = 15
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    id: str
    name: str
    type: str  # email, webhook, slack, etc.
    enabled: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)

class AlertSystem:
    """
    Advanced alert and notification system
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize alert system
        
        Args:
            db_path: Path to alerts database
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "alerts.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Alert storage
        self.active_alerts = {}
        self.alert_history = []
        self.alert_rules = {}
        self.notification_channels = {}
        
        # Cooldown tracking
        self.last_alert_times = {}
        
        # Initialize database
        self._init_database()
        
        # Load default rules and channels
        self._load_default_rules()
        self._load_default_channels()
    
    def _init_database(self):
        """Initialize alerts database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_by TEXT,
                    acknowledged_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    condition TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    cooldown_minutes INTEGER DEFAULT 15,
                    notification_channels TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_channels (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    configuration TEXT,
                    filters TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
    
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="Alert when CPU usage exceeds 80%",
                condition="system_metrics.cpu_percent > 80",
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.SYSTEM_HEALTH,
                cooldown_minutes=30,
                notification_channels=["email", "webhook"]
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                description="Alert when memory usage exceeds 85%",
                condition="system_metrics.memory_percent > 85",
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.SYSTEM_HEALTH,
                cooldown_minutes=30,
                notification_channels=["email"]
            ),
            AlertRule(
                id="high_disk_usage",
                name="High Disk Usage",
                description="Alert when disk usage exceeds 90%",
                condition="system_metrics.disk_usage_percent > 90",
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.SYSTEM_HEALTH,
                cooldown_minutes=60,
                notification_channels=["email", "webhook", "slack"]
            ),
            AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                description="Alert when classification error rate exceeds 10%",
                condition="classification_metrics.failed_requests / max(classification_metrics.total_requests, 1) > 0.1",
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.CLASSIFICATION_ERROR,
                cooldown_minutes=15,
                notification_channels=["email", "webhook"]
            ),
            AlertRule(
                id="slow_processing",
                name="Slow Processing",
                description="Alert when average processing time exceeds 5 seconds",
                condition="classification_metrics.avg_processing_time > 5.0",
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.PERFORMANCE,
                cooldown_minutes=20,
                notification_channels=["email"]
            ),
            AlertRule(
                id="external_service_down",
                name="External Service Down",
                description="Alert when external service is unavailable",
                condition="external_service_status.available == False",
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.EXTERNAL_SERVICE,
                cooldown_minutes=10,
                notification_channels=["email", "webhook"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def _load_default_channels(self):
        """Load default notification channels"""
        default_channels = [
            NotificationChannel(
                id="email",
                name="Email Notifications",
                type="email",
                enabled=False,  # Disabled by default, requires configuration
                configuration={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "",
                    "to_emails": []
                },
                filters={
                    "min_severity": AlertSeverity.WARNING.value
                }
            ),
            NotificationChannel(
                id="webhook",
                name="Webhook Notifications",
                type="webhook",
                enabled=False,
                configuration={
                    "url": "",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "timeout": 30
                },
                filters={
                    "min_severity": AlertSeverity.ERROR.value
                }
            ),
            NotificationChannel(
                id="slack",
                name="Slack Notifications",
                type="slack",
                enabled=False,
                configuration={
                    "webhook_url": "",
                    "channel": "#alerts",
                    "username": "AI Document Classifier"
                },
                filters={
                    "min_severity": AlertSeverity.CRITICAL.value
                }
            )
        ]
        
        for channel in default_channels:
            self.notification_channels[channel.id] = channel
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.id] = rule
        self._save_alert_rule(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self._delete_alert_rule(rule_id)
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a new notification channel"""
        self.notification_channels[channel.id] = channel
        self._save_notification_channel(channel)
        logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_id: str):
        """Remove a notification channel"""
        if channel_id in self.notification_channels:
            del self.notification_channels[channel_id]
            self._delete_notification_channel(channel_id)
            logger.info(f"Removed notification channel: {channel_id}")
    
    def check_alerts(self, context: Dict[str, Any]):
        """
        Check all alert rules against current context
        
        Args:
            context: Current system context (metrics, status, etc.)
        """
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_id):
                continue
            
            try:
                # Evaluate condition
                if self._evaluate_condition(rule.condition, context):
                    # Create alert
                    alert = Alert(
                        id=f"{rule_id}_{int(datetime.now().timestamp())}",
                        type=rule.alert_type,
                        severity=rule.severity,
                        title=rule.name,
                        message=rule.description,
                        timestamp=datetime.now(),
                        source=rule_id,
                        metadata=rule.metadata
                    )
                    
                    # Store alert
                    self._store_alert(alert)
                    self.active_alerts[alert.id] = alert
                    
                    # Send notifications
                    self._send_notifications(alert, rule.notification_channels)
                    
                    # Update cooldown
                    self.last_alert_times[rule_id] = datetime.now()
                    
                    logger.warning(f"Alert triggered: {rule.name}")
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_id}: {e}")
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate alert condition
        
        Args:
            condition: Python expression to evaluate
            context: Context variables
            
        Returns:
            True if condition is met
        """
        try:
            # Create safe evaluation context
            safe_context = {
                '__builtins__': {
                    'max': max,
                    'min': min,
                    'len': len,
                    'sum': sum,
                    'abs': abs,
                    'round': round
                }
            }
            
            # Add context variables
            safe_context.update(context)
            
            # Evaluate condition
            result = eval(condition, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.last_alert_times:
            return False
        
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        last_alert = self.last_alert_times[rule_id]
        cooldown_end = last_alert + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() < cooldown_end
    
    def _send_notifications(self, alert: Alert, channel_ids: List[str]):
        """Send notifications through specified channels"""
        for channel_id in channel_ids:
            if channel_id not in self.notification_channels:
                continue
            
            channel = self.notification_channels[channel_id]
            if not channel.enabled:
                continue
            
            # Check severity filter
            if not self._should_send_notification(alert, channel):
                continue
            
            try:
                if channel.type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, channel)
                elif channel.type == "slack":
                    self._send_slack_notification(alert, channel)
                else:
                    logger.warning(f"Unknown notification channel type: {channel.type}")
            
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_id}: {e}")
    
    def _should_send_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Check if notification should be sent based on channel filters"""
        filters = channel.filters
        
        # Check minimum severity
        if "min_severity" in filters:
            min_severity = AlertSeverity(filters["min_severity"])
            severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]
            
            if severity_order.index(alert.severity) < severity_order.index(min_severity):
                return False
        
        # Check alert type filter
        if "alert_types" in filters:
            if alert.type.value not in filters["alert_types"]:
                return False
        
        return True
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.configuration
        
        if not all(config.get(key) for key in ["smtp_server", "username", "password", "from_email", "to_emails"]):
            logger.warning("Email channel not properly configured")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config["from_email"]
            msg['To'] = ", ".join(config["to_emails"])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create body
            body = f"""
Alert Details:
- Type: {alert.type.value}
- Severity: {alert.severity.value}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Source: {alert.source}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config.get("smtp_port", 587))
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        config = channel.configuration
        
        if not config.get("url"):
            logger.warning("Webhook channel not properly configured")
            return
        
        try:
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata
            }
            
            # Send webhook (synchronous for now)
            import requests
            response = requests.post(
                config["url"],
                json=payload,
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30)
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for alert: {alert.id}")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        config = channel.configuration
        
        if not config.get("webhook_url"):
            logger.warning("Slack channel not properly configured")
            return
        
        try:
            # Create Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "good")
            
            payload = {
                "channel": config.get("channel", "#alerts"),
                "username": config.get("username", "AI Document Classifier"),
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "AI Document Classifier",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            import requests
            response = requests.post(config["webhook_url"], json=payload)
            
            if response.status_code == 200:
                logger.info(f"Slack notification sent for alert: {alert.id}")
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            self._update_alert(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            self._update_alert(alert)
            
            logger.info(f"Alert acknowledged by {acknowledged_by}: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM alerts 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                alerts = []
                for row in cursor.fetchall():
                    alert = Alert(
                        id=row[0],
                        type=AlertType(row[1]),
                        severity=AlertSeverity(row[2]),
                        title=row[3],
                        message=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        source=row[6],
                        metadata=json.loads(row[7]) if row[7] else {},
                        resolved=bool(row[8]),
                        resolved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                        acknowledged=bool(row[10]),
                        acknowledged_by=row[11],
                        acknowledged_at=datetime.fromisoformat(row[12]) if row[12] else None
                    )
                    alerts.append(alert)
                
                return alerts
                
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts 
                    (id, type, severity, title, message, timestamp, source, metadata,
                     resolved, resolved_at, acknowledged, acknowledged_by, acknowledged_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id, alert.type.value, alert.severity.value, alert.title,
                    alert.message, alert.timestamp.isoformat(), alert.source,
                    json.dumps(alert.metadata), alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.acknowledged, alert.acknowledged_by,
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE alerts 
                    SET resolved = ?, resolved_at = ?, acknowledged = ?, 
                        acknowledged_by = ?, acknowledged_at = ?
                    WHERE id = ?
                """, (
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.acknowledged,
                    alert.acknowledged_by,
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    alert.id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
    
    def _save_alert_rule(self, rule: AlertRule):
        """Save alert rule to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alert_rules 
                    (id, name, description, condition, severity, alert_type, enabled,
                     cooldown_minutes, notification_channels, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.id, rule.name, rule.description, rule.condition,
                    rule.severity.value, rule.alert_type.value, rule.enabled,
                    rule.cooldown_minutes, json.dumps(rule.notification_channels),
                    json.dumps(rule.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving alert rule: {e}")
    
    def _delete_alert_rule(self, rule_id: str):
        """Delete alert rule from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting alert rule: {e}")
    
    def _save_notification_channel(self, channel: NotificationChannel):
        """Save notification channel to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO notification_channels 
                    (id, name, type, enabled, configuration, filters)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    channel.id, channel.name, channel.type, channel.enabled,
                    json.dumps(channel.configuration), json.dumps(channel.filters)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving notification channel: {e}")
    
    def _delete_notification_channel(self, channel_id: str):
        """Delete notification channel from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM notification_channels WHERE id = ?", (channel_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting notification channel: {e}")

# Global alert system instance
alert_system = AlertSystem()

# Example usage
if __name__ == "__main__":
    # Initialize alert system
    alerts = AlertSystem()
    
    # Test alert checking
    test_context = {
        "system_metrics": {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "disk_usage_percent": 95.0
        },
        "classification_metrics": {
            "total_requests": 100,
            "failed_requests": 15,
            "avg_processing_time": 6.0
        }
    }
    
    # Check alerts
    alerts.check_alerts(test_context)
    
    # Get active alerts
    active = alerts.get_active_alerts()
    print(f"Active alerts: {len(active)}")
    
    for alert in active:
        print(f"- {alert.title} ({alert.severity.value})")
    
    print("Alert system initialized successfully")



























