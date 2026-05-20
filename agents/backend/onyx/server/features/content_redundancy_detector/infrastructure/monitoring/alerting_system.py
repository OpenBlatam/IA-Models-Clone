"""
Advanced Alerting System - Real-time alerts and notifications
Production-ready alerting and notification system
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    message_template: str
    cooldown: float = 300.0  # 5 minutes
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    channel_type: str
    config: Dict[str, Any]
    enabled: bool = True

class AlertingSystem:
    """Advanced alerting and notification system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Cooldown tracking
        self.last_alert_time: Dict[str, float] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.resolve_callbacks: List[Callable[[Alert], None]] = []

    async def start(self):
        """Start alerting system"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_worker())

    async def stop(self):
        """Stop alerting system"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: AlertLevel,
        message_template: str,
        cooldown: float = 300.0,
        enabled: bool = True,
        tags: Dict[str, str] = None
    ):
        """Add an alert rule"""
        rule = AlertRule(
            name=name,
            condition=condition,
            level=level,
            message_template=message_template,
            cooldown=cooldown,
            enabled=enabled,
            tags=tags or {}
        )
        
        with self.lock:
            self.alert_rules[name] = rule

    def remove_alert_rule(self, name: str):
        """Remove an alert rule"""
        with self.lock:
            if name in self.alert_rules:
                del self.alert_rules[name]
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.rule_name == name
            ]
            
            for alert_id in alerts_to_resolve:
                self._resolve_alert(alert_id)

    def add_notification_channel(
        self,
        name: str,
        channel_type: str,
        config: Dict[str, Any],
        enabled: bool = True
    ):
        """Add a notification channel"""
        channel = NotificationChannel(
            name=name,
            channel_type=channel_type,
            config=config,
            enabled=enabled
        )
        
        with self.lock:
            self.notification_channels[name] = channel

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert events"""
        self.alert_callbacks.append(callback)

    def add_resolve_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert resolution"""
        self.resolve_callbacks.append(callback)

    async def check_alerts(self, data: Dict[str, Any]):
        """Check all alert rules against data"""
        with self.lock:
            current_time = time.time()
            
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule_name in self.last_alert_time:
                    if current_time - self.last_alert_time[rule_name] < rule.cooldown:
                        continue
                
                try:
                    # Check condition
                    if rule.condition(data):
                        # Create alert
                        alert = self._create_alert(rule, data)
                        
                        # Check if alert already exists
                        existing_alert = self._find_active_alert(rule_name)
                        if existing_alert:
                            # Update existing alert
                            existing_alert.timestamp = current_time
                            existing_alert.data.update(data)
                        else:
                            # Create new alert
                            self.active_alerts[alert.id] = alert
                            self.alert_history.append(alert)
                            self.last_alert_time[rule_name] = current_time
                            
                            # Notify callbacks
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert)
                                except Exception as e:
                                    print(f"Alert callback error: {e}")
                            
                            # Send notifications
                            await self._send_notifications(alert)
                    else:
                        # Condition not met, resolve alert if exists
                        existing_alert = self._find_active_alert(rule_name)
                        if existing_alert:
                            self._resolve_alert(existing_alert.id)
                
                except Exception as e:
                    print(f"Alert rule '{rule_name}' error: {e}")

    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create an alert from rule and data"""
        alert_id = f"{rule.name}_{int(time.time() * 1000)}"
        
        # Format message
        try:
            message = rule.message_template.format(**data)
        except KeyError as e:
            message = f"{rule.message_template} (missing data: {e})"
        
        return Alert(
            id=alert_id,
            rule_name=rule.name,
            level=rule.level,
            message=message,
            timestamp=time.time(),
            data=data.copy(),
            tags=rule.tags.copy()
        )

    def _find_active_alert(self, rule_name: str) -> Optional[Alert]:
        """Find active alert for rule"""
        for alert in self.active_alerts.values():
            if alert.rule_name == rule_name and not alert.resolved:
                return alert
        return None

    def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            
            # Notify callbacks
            for callback in self.resolve_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Resolve callback error: {e}")
            
            # Remove from active alerts
            del self.active_alerts[alert_id]

    async def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        for channel_name, channel in self.notification_channels.items():
            if not channel.enabled:
                continue
            
            try:
                if channel.channel_type == "email":
                    await self._send_email_notification(alert, channel)
                elif channel.channel_type == "webhook":
                    await self._send_webhook_notification(alert, channel)
                elif channel.channel_type == "slack":
                    await self._send_slack_notification(alert, channel)
                elif channel.channel_type == "console":
                    await self._send_console_notification(alert, channel)
            except Exception as e:
                print(f"Notification error for channel '{channel_name}': {e}")

    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config
        
        msg = MimeMultipart()
        msg['From'] = config.get('from_email')
        msg['To'] = config.get('to_email')
        msg['Subject'] = f"[{alert.level.value.upper()}] {alert.rule_name}"
        
        body = f"""
        Alert: {alert.rule_name}
        Level: {alert.level.value.upper()}
        Time: {time.ctime(alert.timestamp)}
        Message: {alert.message}
        
        Data: {json.dumps(alert.data, indent=2)}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email (simplified - in production use proper SMTP)
        print(f"Email notification: {msg['Subject']}")

    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        import aiohttp
        
        config = channel.config
        webhook_url = config.get('url')
        
        if not webhook_url:
            return
        
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "level": alert.level.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "data": alert.data,
            "tags": alert.tags
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    print(f"Webhook notification failed: {response.status}")

    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        import aiohttp
        
        config = channel.config
        webhook_url = config.get('webhook_url')
        
        if not webhook_url:
            return
        
        # Color coding based on alert level
        colors = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ffaa00",
            AlertLevel.ERROR: "#ff6b6b",
            AlertLevel.CRITICAL: "#ff0000"
        }
        
        payload = {
            "attachments": [{
                "color": colors.get(alert.level, "#36a64f"),
                "title": f"Alert: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Level", "value": alert.level.value.upper(), "short": True},
                    {"title": "Time", "value": time.ctime(alert.timestamp), "short": True}
                ],
                "footer": "Content Redundancy Detector",
                "ts": int(alert.timestamp)
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    print(f"Slack notification failed: {response.status}")

    async def _send_console_notification(self, alert: Alert, channel: NotificationChannel):
        """Send console notification"""
        print(f"\n🚨 ALERT [{alert.level.value.upper()}] {alert.rule_name}")
        print(f"   Message: {alert.message}")
        print(f"   Time: {time.ctime(alert.timestamp)}")
        print(f"   Data: {json.dumps(alert.data, indent=2)}")
        print("-" * 50)

    async def _monitoring_worker(self):
        """Background monitoring worker"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for stale alerts (older than 1 hour)
                current_time = time.time()
                stale_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if current_time - alert.timestamp > 3600
                ]
                
                for alert_id in stale_alerts:
                    self._resolve_alert(alert_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Alert monitoring error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self.lock:
            return list(self.alert_history)[-limit:]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self.lock:
            active_count = len(self.active_alerts)
            total_alerts = len(self.alert_history)
            
            # Count by level
            level_counts = defaultdict(int)
            for alert in self.alert_history:
                level_counts[alert.level.value] += 1
            
            # Count by rule
            rule_counts = defaultdict(int)
            for alert in self.alert_history:
                rule_counts[alert.rule_name] += 1
            
            return {
                "active_alerts": active_count,
                "total_alerts": total_alerts,
                "alerts_by_level": dict(level_counts),
                "alerts_by_rule": dict(rule_counts),
                "notification_channels": len(self.notification_channels),
                "alert_rules": len(self.alert_rules)
            }

    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        with self.lock:
            self._resolve_alert(alert_id)

    def resolve_all_alerts(self):
        """Resolve all active alerts"""
        with self.lock:
            alert_ids = list(self.active_alerts.keys())
            for alert_id in alert_ids:
                self._resolve_alert(alert_id)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of alerting system"""
        with self.lock:
            return {
                "status": "healthy" if self.running else "stopped",
                "active_alerts": len(self.active_alerts),
                "total_alerts": len(self.alert_history),
                "alert_rules": len(self.alert_rules),
                "notification_channels": len(self.notification_channels),
                "monitoring_running": self.monitoring_task is not None,
                "callbacks_registered": len(self.alert_callbacks)
            }





