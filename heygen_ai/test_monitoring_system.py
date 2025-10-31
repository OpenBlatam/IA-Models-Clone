#!/usr/bin/env python3
"""
Advanced Test Monitoring and Alerting System
===========================================

This module provides comprehensive monitoring, alerting, and notification
capabilities for test execution and quality metrics.
"""

import sys
import json
import time
import smtplib
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import schedule
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types"""
    TEST_FAILURE = "test_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COVERAGE_DROP = "coverage_drop"
    SECURITY_ISSUE = "security_issue"
    SYSTEM_ERROR = "system_error"
    QUALITY_GATE_FAILURE = "quality_gate_failure"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MonitoringRule:
    """Monitoring rule definition"""
    id: str
    name: str
    description: str
    condition: str
    alert_type: AlertType
    alert_level: AlertLevel
    enabled: bool = True
    threshold_value: Optional[float] = None
    time_window: int = 300  # seconds
    cooldown_period: int = 3600  # seconds
    last_triggered: Optional[datetime] = None

@dataclass
class NotificationChannel:
    """Notification channel definition"""
    id: str
    name: str
    type: str  # "email", "webhook", "slack", "teams"
    config: Dict[str, Any]
    enabled: bool = True

class TestMonitoringSystem:
    """Advanced test monitoring and alerting system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.alerts: Dict[str, Alert] = {}
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Initialize default rules and channels
        self._initialize_default_rules()
        self._initialize_default_channels()
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "check_interval": 60,  # seconds
                "retention_days": 30,
                "max_alerts": 1000
            },
            "alerting": {
                "enabled": True,
                "default_cooldown": 3600,  # seconds
                "escalation_enabled": True,
                "escalation_delay": 1800  # seconds
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "",
                    "to_addresses": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {},
                    "timeout": 30
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#testing",
                    "username": "TestMonitor"
                }
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _initialize_default_rules(self):
        """Initialize default monitoring rules"""
        default_rules = [
            MonitoringRule(
                id="test_failure_rate",
                name="High Test Failure Rate",
                description="Alert when test failure rate exceeds 10%",
                condition="failure_rate > 0.1",
                alert_type=AlertType.TEST_FAILURE,
                alert_level=AlertLevel.WARNING,
                threshold_value=0.1
            ),
            MonitoringRule(
                id="coverage_drop",
                name="Test Coverage Drop",
                description="Alert when test coverage drops below 80%",
                condition="coverage < 0.8",
                alert_type=AlertType.COVERAGE_DROP,
                alert_level=AlertLevel.WARNING,
                threshold_value=0.8
            ),
            MonitoringRule(
                id="performance_degradation",
                name="Performance Degradation",
                description="Alert when test execution time increases by 50%",
                condition="execution_time_increase > 0.5",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                alert_level=AlertLevel.WARNING,
                threshold_value=0.5
            ),
            MonitoringRule(
                id="security_score_low",
                name="Low Security Score",
                description="Alert when security score drops below 7.0",
                condition="security_score < 7.0",
                alert_type=AlertType.SECURITY_ISSUE,
                alert_level=AlertLevel.ERROR,
                threshold_value=7.0
            ),
            MonitoringRule(
                id="quality_gate_failure",
                name="Quality Gate Failure",
                description="Alert when quality gate fails",
                condition="quality_gate_status == 'failed'",
                alert_type=AlertType.QUALITY_GATE_FAILURE,
                alert_level=AlertLevel.CRITICAL
            )
        ]
        
        for rule in default_rules:
            self.monitoring_rules[rule.id] = rule
    
    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        default_channels = [
            NotificationChannel(
                id="email_default",
                name="Default Email",
                type="email",
                config={
                    "enabled": self.config["notifications"]["email"]["enabled"],
                    "smtp_server": self.config["notifications"]["email"]["smtp_server"],
                    "smtp_port": self.config["notifications"]["email"]["smtp_port"],
                    "username": self.config["notifications"]["email"]["username"],
                    "password": self.config["notifications"]["email"]["password"],
                    "from_address": self.config["notifications"]["email"]["from_address"],
                    "to_addresses": self.config["notifications"]["email"]["to_addresses"]
                }
            ),
            NotificationChannel(
                id="webhook_default",
                name="Default Webhook",
                type="webhook",
                config={
                    "enabled": self.config["notifications"]["webhook"]["enabled"],
                    "url": self.config["notifications"]["webhook"]["url"],
                    "headers": self.config["notifications"]["webhook"]["headers"],
                    "timeout": self.config["notifications"]["webhook"]["timeout"]
                }
            ),
            NotificationChannel(
                id="slack_default",
                name="Default Slack",
                type="slack",
                config={
                    "enabled": self.config["notifications"]["slack"]["enabled"],
                    "webhook_url": self.config["notifications"]["slack"]["webhook_url"],
                    "channel": self.config["notifications"]["slack"]["channel"],
                    "username": self.config["notifications"]["slack"]["username"]
                }
            )
        ]
        
        for channel in default_channels:
            self.notification_channels[channel.id] = channel
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a new monitoring rule"""
        self.monitoring_rules[rule.id] = rule
        print(f"✅ Added monitoring rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a new notification channel"""
        self.notification_channels[channel.id] = channel
        print(f"✅ Added notification channel: {channel.name}")
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """Record test metrics for monitoring"""
        timestamp = datetime.now()
        metrics_record = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        }
        
        self.metrics_history.append(metrics_record)
        
        # Clean up old metrics
        retention_days = self.config["monitoring"]["retention_days"]
        cutoff_date = timestamp - timedelta(days=retention_days)
        self.metrics_history = [
            record for record in self.metrics_history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]
        
        # Check monitoring rules
        self._check_monitoring_rules(metrics, timestamp)
    
    def _check_monitoring_rules(self, metrics: Dict[str, Any], timestamp: datetime):
        """Check all monitoring rules against current metrics"""
        for rule_id, rule in self.monitoring_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if rule.last_triggered:
                time_since_last = (timestamp - rule.last_triggered).total_seconds()
                if time_since_last < rule.cooldown_period:
                    continue
            
            # Evaluate rule condition
            if self._evaluate_rule_condition(rule, metrics):
                self._trigger_alert(rule, metrics, timestamp)
                rule.last_triggered = timestamp
    
    def _evaluate_rule_condition(self, rule: MonitoringRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate a monitoring rule condition"""
        try:
            # Simple condition evaluation
            condition = rule.condition
            
            # Replace metric names with actual values
            for metric_name, metric_value in metrics.items():
                condition = condition.replace(metric_name, str(metric_value))
            
            # Replace threshold value
            if rule.threshold_value is not None:
                condition = condition.replace("threshold", str(rule.threshold_value))
            
            # Evaluate the condition
            result = eval(condition)
            return bool(result)
            
        except Exception as e:
            print(f"⚠️  Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    def _trigger_alert(self, rule: MonitoringRule, metrics: Dict[str, Any], timestamp: datetime):
        """Trigger an alert based on a monitoring rule"""
        alert_id = f"{rule.id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        alert = Alert(
            id=alert_id,
            type=rule.alert_type,
            level=rule.alert_level,
            title=f"{rule.name} - {rule.alert_level.value.upper()}",
            message=f"Rule '{rule.name}' triggered: {rule.description}",
            timestamp=timestamp,
            source="monitoring_system",
            metadata={
                "rule_id": rule.id,
                "condition": rule.condition,
                "metrics": metrics,
                "threshold": rule.threshold_value
            }
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        self._send_notifications(alert)
        
        print(f"🚨 Alert triggered: {alert.title}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for channel_id, channel in self.notification_channels.items():
            if not channel.enabled or not channel.config.get("enabled", False):
                continue
            
            try:
                if channel.type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, channel)
                elif channel.type == "slack":
                    self._send_slack_notification(alert, channel)
            except Exception as e:
                print(f"❌ Error sending notification via {channel.name}: {e}")
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config
        
        msg = MIMEMultipart()
        msg['From'] = config["from_address"]
        msg['To'] = ", ".join(config["to_addresses"])
        msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
        
        body = f"""
        Alert Details:
        =============
        
        Type: {alert.type.value}
        Level: {alert.level.value.upper()}
        Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        
        Message: {alert.message}
        
        Metadata:
        {json.dumps(alert.metadata, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
        server.starttls()
        server.login(config["username"], config["password"])
        server.send_message(msg)
        server.quit()
        
        print(f"📧 Email notification sent via {channel.name}")
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        config = channel.config
        
        payload = {
            "alert_id": alert.id,
            "type": alert.type.value,
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata
        }
        
        response = requests.post(
            config["url"],
            json=payload,
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30)
        )
        
        response.raise_for_status()
        print(f"🔗 Webhook notification sent via {channel.name}")
    
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        config = channel.config
        
        # Determine color based on alert level
        color_map = {
            AlertLevel.INFO: "good",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "danger",
            AlertLevel.CRITICAL: "danger"
        }
        
        payload = {
            "channel": config["channel"],
            "username": config["username"],
            "attachments": [{
                "color": color_map.get(alert.level, "good"),
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Type", "value": alert.type.value, "short": True},
                    {"title": "Level", "value": alert.level.value.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                ],
                "footer": "Test Monitoring System",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        response = requests.post(config["webhook_url"], json=payload)
        response.raise_for_status()
        print(f"💬 Slack notification sent via {channel.name}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            print(f"✅ Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts.values() if alert.level == level]
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        active_alerts = self.get_active_alerts()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": {
                "active": self.monitoring_active,
                "total_rules": len(self.monitoring_rules),
                "enabled_rules": len([r for r in self.monitoring_rules.values() if r.enabled]),
                "total_channels": len(self.notification_channels),
                "enabled_channels": len([c for c in self.notification_channels.values() if c.enabled])
            },
            "alerts_summary": {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len([a for a in self.alerts.values() if a.resolved]),
                "by_level": {
                    level.value: len(self.get_alerts_by_level(level))
                    for level in AlertLevel
                },
                "by_type": {}
            },
            "recent_metrics": self.metrics_history[-10:] if self.metrics_history else [],
            "active_alerts": [asdict(alert) for alert in active_alerts]
        }
        
        # Count alerts by type
        for alert in self.alerts.values():
            alert_type = alert.type.value
            if alert_type not in report["alerts_summary"]["by_type"]:
                report["alerts_summary"]["by_type"][alert_type] = 0
            report["alerts_summary"]["by_type"][alert_type] += 1
        
        return report
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            print("⚠️  Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("🚀 Test monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🛑 Test monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        check_interval = self.config["monitoring"]["check_interval"]
        
        while self.monitoring_active:
            try:
                # Perform monitoring checks
                self._perform_health_checks()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _perform_health_checks(self):
        """Perform system health checks"""
        # Check if metrics are being recorded
        if not self.metrics_history:
            return
        
        # Check for stale metrics
        latest_metrics = self.metrics_history[-1]
        latest_time = datetime.fromisoformat(latest_metrics["timestamp"])
        time_since_last = (datetime.now() - latest_time).total_seconds()
        
        if time_since_last > 300:  # 5 minutes
            alert = Alert(
                id=f"stale_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.WARNING,
                title="Stale Metrics Detected",
                message=f"No metrics recorded for {time_since_last:.0f} seconds",
                timestamp=datetime.now(),
                source="monitoring_system",
                metadata={"time_since_last": time_since_last}
            )
            self.alerts[alert.id] = alert
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        max_alerts = self.config["monitoring"]["max_alerts"]
        
        if len(self.alerts) > max_alerts:
            # Sort alerts by timestamp and remove oldest resolved ones
            sorted_alerts = sorted(
                self.alerts.items(),
                key=lambda x: x[1].timestamp
            )
            
            resolved_alerts = [(id, alert) for id, alert in sorted_alerts if alert.resolved]
            
            # Remove oldest resolved alerts
            alerts_to_remove = len(self.alerts) - max_alerts
            for i in range(min(alerts_to_remove, len(resolved_alerts))):
                alert_id = resolved_alerts[i][0]
                del self.alerts[alert_id]


def main():
    """Main function for test monitoring system"""
    print("📊 Advanced Test Monitoring and Alerting System")
    print("=" * 60)
    
    # Initialize monitoring system
    monitor = TestMonitoringSystem()
    
    # Add some sample notification channels
    print("📧 Setting up notification channels...")
    
    # Example email channel (disabled by default)
    email_channel = NotificationChannel(
        id="test_email",
        name="Test Email Channel",
        type="email",
        config={
            "enabled": False,  # Set to True and configure for actual use
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@gmail.com",
            "password": "your-app-password",
            "from_address": "your-email@gmail.com",
            "to_addresses": ["recipient@example.com"]
        }
    )
    monitor.add_notification_channel(email_channel)
    
    # Example webhook channel (disabled by default)
    webhook_channel = NotificationChannel(
        id="test_webhook",
        name="Test Webhook Channel",
        type="webhook",
        config={
            "enabled": False,  # Set to True and configure for actual use
            "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30
        }
    )
    monitor.add_notification_channel(webhook_channel)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some test metrics
    print("📈 Simulating test metrics...")
    
    sample_metrics = [
        {
            "test_count": 100,
            "passed_tests": 95,
            "failed_tests": 5,
            "coverage": 0.85,
            "execution_time": 45.2,
            "security_score": 8.5,
            "quality_gate_status": "passed"
        },
        {
            "test_count": 100,
            "passed_tests": 88,
            "failed_tests": 12,
            "coverage": 0.75,
            "execution_time": 67.8,
            "security_score": 6.5,
            "quality_gate_status": "failed"
        }
    ]
    
    for i, metrics in enumerate(sample_metrics):
        print(f"📊 Recording metrics batch {i+1}...")
        monitor.record_metrics(metrics)
        time.sleep(2)  # Simulate time between metric recordings
    
    # Generate monitoring report
    print("\n📋 Generating monitoring report...")
    report = monitor.generate_monitoring_report()
    
    # Print summary
    print("\n📊 Monitoring System Summary:")
    print(f"  📈 Total rules: {report['monitoring_status']['total_rules']}")
    print(f"  ✅ Enabled rules: {report['monitoring_status']['enabled_rules']}")
    print(f"  📧 Total channels: {report['monitoring_status']['total_channels']}")
    print(f"  ✅ Enabled channels: {report['monitoring_status']['enabled_channels']}")
    
    print(f"\n🚨 Alerts Summary:")
    print(f"  📊 Total alerts: {report['alerts_summary']['total_alerts']}")
    print(f"  🔴 Active alerts: {report['alerts_summary']['active_alerts']}")
    print(f"  ✅ Resolved alerts: {report['alerts_summary']['resolved_alerts']}")
    
    # Print active alerts
    active_alerts = monitor.get_active_alerts()
    if active_alerts:
        print(f"\n🚨 Active Alerts:")
        for alert in active_alerts:
            print(f"  🔴 [{alert.level.value.upper()}] {alert.title}")
            print(f"     {alert.message}")
    
    # Save report
    with open("monitoring_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Monitoring report saved: monitoring_report.json")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\n🎉 Test monitoring system demonstration completed!")
    print("📄 Check 'monitoring_report.json' for detailed results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


