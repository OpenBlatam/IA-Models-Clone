"""
BUL Notification System
======================

Advanced notification system for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import sqlite3
from dataclasses import dataclass
from enum import Enum
import yaml

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Notification types."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    LOG = "log"
    FILE = "file"

class NotificationPriority(Enum):
    """Notification priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class NotificationTemplate:
    """Notification template."""
    id: str
    name: str
    notification_type: NotificationType
    subject: str
    body: str
    variables: List[str] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM

@dataclass
class NotificationRule:
    """Notification rule."""
    id: str
    name: str
    condition: str
    template_id: str
    recipients: List[str]
    enabled: bool = True
    cooldown_minutes: int = 0

class NotificationSystem:
    """Advanced notification system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.templates = {}
        self.rules = {}
        self.notification_history = []
        self.init_database()
        self.init_notification_channels()
        self.load_templates()
        self.load_rules()
    
    def init_database(self):
        """Initialize notification database."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                notification_type TEXT,
                recipient TEXT,
                subject TEXT,
                body TEXT,
                status TEXT,
                priority TEXT,
                template_id TEXT,
                error_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_templates (
                id TEXT PRIMARY KEY,
                name TEXT,
                notification_type TEXT,
                subject TEXT,
                body TEXT,
                variables TEXT,
                priority TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_rules (
                id TEXT PRIMARY KEY,
                name TEXT,
                condition TEXT,
                template_id TEXT,
                recipients TEXT,
                enabled BOOLEAN,
                cooldown_minutes INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_notification_channels(self):
        """Initialize notification channels."""
        print("📧 Initializing notification channels...")
        
        # Email configuration
        if hasattr(self.config, 'smtp_server') and self.config.smtp_server:
            self.email_config = {
                'smtp_server': self.config.smtp_server,
                'smtp_port': getattr(self.config, 'smtp_port', 587),
                'username': getattr(self.config, 'smtp_username', ''),
                'password': getattr(self.config, 'smtp_password', ''),
                'use_tls': getattr(self.config, 'smtp_use_tls', True)
            }
            print("✅ Email notification channel initialized")
        
        # Slack configuration
        if hasattr(self.config, 'slack_webhook_url') and self.config.slack_webhook_url:
            self.slack_config = {
                'webhook_url': self.config.slack_webhook_url,
                'channel': getattr(self.config, 'slack_channel', '#general'),
                'username': getattr(self.config, 'slack_username', 'BUL System')
            }
            print("✅ Slack notification channel initialized")
        
        # Teams configuration
        if hasattr(self.config, 'teams_webhook_url') and self.config.teams_webhook_url:
            self.teams_config = {
                'webhook_url': self.config.teams_webhook_url
            }
            print("✅ Teams notification channel initialized")
        
        # SMS configuration
        if hasattr(self.config, 'twilio_account_sid') and self.config.twilio_account_sid:
            self.sms_config = {
                'account_sid': self.config.twilio_account_sid,
                'auth_token': getattr(self.config, 'twilio_auth_token', ''),
                'from_number': getattr(self.config, 'twilio_from_number', '')
            }
            print("✅ SMS notification channel initialized")
    
    def load_templates(self):
        """Load notification templates."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM notification_templates")
        rows = cursor.fetchall()
        
        for row in rows:
            template = NotificationTemplate(
                id=row[0],
                name=row[1],
                notification_type=NotificationType(row[2]),
                subject=row[3],
                body=row[4],
                variables=json.loads(row[5]) if row[5] else [],
                priority=NotificationPriority(row[6])
            )
            self.templates[template.id] = template
        
        conn.close()
        
        # Create default templates if none exist
        if not self.templates:
            self.create_default_templates()
    
    def load_rules(self):
        """Load notification rules."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM notification_rules")
        rows = cursor.fetchall()
        
        for row in rows:
            rule = NotificationRule(
                id=row[0],
                name=row[1],
                condition=row[2],
                template_id=row[3],
                recipients=json.loads(row[4]),
                enabled=bool(row[5]),
                cooldown_minutes=row[6]
            )
            self.rules[rule.id] = rule
        
        conn.close()
    
    def create_default_templates(self):
        """Create default notification templates."""
        default_templates = [
            {
                'id': 'system_startup',
                'name': 'System Startup',
                'type': NotificationType.EMAIL,
                'subject': 'BUL System Started',
                'body': 'The BUL system has been started successfully at {timestamp}.',
                'variables': ['timestamp'],
                'priority': NotificationPriority.MEDIUM
            },
            {
                'id': 'system_error',
                'name': 'System Error',
                'type': NotificationType.EMAIL,
                'subject': 'BUL System Error',
                'body': 'An error occurred in the BUL system: {error_message}\n\nTime: {timestamp}\nComponent: {component}',
                'variables': ['error_message', 'timestamp', 'component'],
                'priority': NotificationPriority.HIGH
            },
            {
                'id': 'document_generated',
                'name': 'Document Generated',
                'type': NotificationType.EMAIL,
                'subject': 'Document Generated: {document_title}',
                'body': 'A new document has been generated:\n\nTitle: {document_title}\nType: {document_type}\nBusiness Area: {business_area}\nGenerated at: {timestamp}',
                'variables': ['document_title', 'document_type', 'business_area', 'timestamp'],
                'priority': NotificationPriority.LOW
            },
            {
                'id': 'backup_completed',
                'name': 'Backup Completed',
                'type': NotificationType.EMAIL,
                'subject': 'Backup Completed Successfully',
                'body': 'System backup completed successfully.\n\nBackup Name: {backup_name}\nSize: {backup_size}\nCompleted at: {timestamp}',
                'variables': ['backup_name', 'backup_size', 'timestamp'],
                'priority': NotificationPriority.LOW
            },
            {
                'id': 'security_alert',
                'name': 'Security Alert',
                'type': NotificationType.EMAIL,
                'subject': 'Security Alert: {alert_type}',
                'body': 'Security alert detected:\n\nType: {alert_type}\nDescription: {description}\nSeverity: {severity}\nDetected at: {timestamp}',
                'variables': ['alert_type', 'description', 'severity', 'timestamp'],
                'priority': NotificationPriority.CRITICAL
            }
        ]
        
        for template_data in default_templates:
            self.create_template(
                template_id=template_data['id'],
                name=template_data['name'],
                notification_type=template_data['type'],
                subject=template_data['subject'],
                body=template_data['body'],
                variables=template_data['variables'],
                priority=template_data['priority']
            )
    
    def create_template(self, template_id: str, name: str, notification_type: NotificationType,
                       subject: str, body: str, variables: List[str] = None,
                       priority: NotificationPriority = NotificationPriority.MEDIUM) -> NotificationTemplate:
        """Create a new notification template."""
        template = NotificationTemplate(
            id=template_id,
            name=name,
            notification_type=notification_type,
            subject=subject,
            body=body,
            variables=variables or [],
            priority=priority
        )
        
        self.templates[template_id] = template
        
        # Save to database
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO notification_templates 
            (id, name, notification_type, subject, body, variables, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (template_id, name, notification_type.value, subject, body, 
              json.dumps(variables or []), priority.value))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created notification template: {name}")
        return template
    
    def create_rule(self, rule_id: str, name: str, condition: str, template_id: str,
                   recipients: List[str], enabled: bool = True, 
                   cooldown_minutes: int = 0) -> NotificationRule:
        """Create a new notification rule."""
        rule = NotificationRule(
            id=rule_id,
            name=name,
            condition=condition,
            template_id=template_id,
            recipients=recipients,
            enabled=enabled,
            cooldown_minutes=cooldown_minutes
        )
        
        self.rules[rule_id] = rule
        
        # Save to database
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO notification_rules 
            (id, name, condition, template_id, recipients, enabled, cooldown_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (rule_id, name, condition, template_id, json.dumps(recipients), 
              enabled, cooldown_minutes))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created notification rule: {name}")
        return rule
    
    async def send_notification(self, notification_type: NotificationType, 
                              recipient: str, subject: str, body: str,
                              priority: NotificationPriority = NotificationPriority.MEDIUM,
                              template_id: str = None, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a notification."""
        print(f"📧 Sending {notification_type.value} notification to {recipient}")
        
        # Replace variables in subject and body
        if variables:
            for key, value in variables.items():
                subject = subject.replace(f"{{{key}}}", str(value))
                body = body.replace(f"{{{key}}}", str(value))
        
        try:
            if notification_type == NotificationType.EMAIL:
                result = await self._send_email(recipient, subject, body)
            elif notification_type == NotificationType.SLACK:
                result = await self._send_slack(recipient, subject, body)
            elif notification_type == NotificationType.TEAMS:
                result = await self._send_teams(recipient, subject, body)
            elif notification_type == NotificationType.SMS:
                result = await self._send_sms(recipient, subject, body)
            elif notification_type == NotificationType.WEBHOOK:
                result = await self._send_webhook(recipient, subject, body)
            elif notification_type == NotificationType.LOG:
                result = await self._send_log(recipient, subject, body)
            elif notification_type == NotificationType.FILE:
                result = await self._send_file(recipient, subject, body)
            else:
                raise ValueError(f"Unsupported notification type: {notification_type}")
            
            # Log notification
            self._log_notification(notification_type, recipient, subject, body, "sent", priority, template_id)
            
            return {
                'status': 'sent',
                'notification_type': notification_type.value,
                'recipient': recipient,
                'sent_at': datetime.now().isoformat(),
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            
            # Log failed notification
            self._log_notification(notification_type, recipient, subject, body, "failed", priority, template_id, str(e))
            
            return {
                'status': 'failed',
                'notification_type': notification_type.value,
                'recipient': recipient,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
    
    async def send_template_notification(self, template_id: str, recipient: str, 
                                       variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send notification using a template."""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        variables = variables or {}
        
        return await self.send_notification(
            notification_type=template.notification_type,
            recipient=recipient,
            subject=template.subject,
            body=template.body,
            priority=template.priority,
            template_id=template_id,
            variables=variables
        )
    
    async def evaluate_rules(self, context: Dict[str, Any]):
        """Evaluate notification rules and send notifications."""
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if rule.cooldown_minutes > 0:
                    last_notification = self._get_last_notification_time(rule_id)
                    if last_notification:
                        time_diff = datetime.now() - last_notification
                        if time_diff.total_seconds() < rule.cooldown_minutes * 60:
                            continue
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, context):
                    # Send notifications to all recipients
                    for recipient in rule.recipients:
                        await self.send_template_notification(
                            template_id=rule.template_id,
                            recipient=recipient,
                            variables=context
                        )
                    
                    print(f"✅ Rule triggered: {rule.name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def _send_email(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send email notification."""
        if not hasattr(self, 'email_config'):
            raise ValueError("Email configuration not found")
        
        msg = MIMEMultipart()
        msg['From'] = self.email_config['username']
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        context = ssl.create_default_context()
        
        with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
            if self.email_config['use_tls']:
                server.starttls(context=context)
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
        
        return {'method': 'smtp', 'server': self.email_config['smtp_server']}
    
    async def _send_slack(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send Slack notification."""
        if not hasattr(self, 'slack_config'):
            raise ValueError("Slack configuration not found")
        
        payload = {
            'channel': recipient,
            'username': self.slack_config['username'],
            'text': f"*{subject}*\n{body}",
            'icon_emoji': ':bul:'
        }
        
        response = requests.post(self.slack_config['webhook_url'], json=payload)
        response.raise_for_status()
        
        return {'method': 'webhook', 'status_code': response.status_code}
    
    async def _send_teams(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send Teams notification."""
        if not hasattr(self, 'teams_config'):
            raise ValueError("Teams configuration not found")
        
        payload = {
            'text': f"**{subject}**\n\n{body}"
        }
        
        response = requests.post(self.teams_config['webhook_url'], json=payload)
        response.raise_for_status()
        
        return {'method': 'webhook', 'status_code': response.status_code}
    
    async def _send_sms(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send SMS notification."""
        if not hasattr(self, 'sms_config'):
            raise ValueError("SMS configuration not found")
        
        # This would require Twilio integration
        # For now, just log the SMS
        logger.info(f"SMS to {recipient}: {subject} - {body}")
        
        return {'method': 'sms', 'provider': 'twilio'}
    
    async def _send_webhook(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send webhook notification."""
        payload = {
            'subject': subject,
            'body': body,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(recipient, json=payload)
        response.raise_for_status()
        
        return {'method': 'webhook', 'status_code': response.status_code}
    
    async def _send_log(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send log notification."""
        logger.info(f"NOTIFICATION [{subject}] {body}")
        
        return {'method': 'log'}
    
    async def _send_file(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Send file notification."""
        log_file = Path(recipient)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {subject}: {body}\n")
        
        return {'method': 'file', 'file_path': str(log_file)}
    
    def _log_notification(self, notification_type: NotificationType, recipient: str,
                         subject: str, body: str, status: str, priority: NotificationPriority,
                         template_id: str = None, error_message: str = None):
        """Log notification to database."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notifications 
            (notification_type, recipient, subject, body, status, priority, template_id, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (notification_type.value, recipient, subject, body, status, 
              priority.value, template_id, error_message))
        
        conn.commit()
        conn.close()
    
    def _get_last_notification_time(self, rule_id: str) -> Optional[datetime]:
        """Get last notification time for a rule."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp FROM notifications 
            WHERE template_id = (SELECT template_id FROM notification_rules WHERE id = ?)
            ORDER BY timestamp DESC LIMIT 1
        ''', (rule_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate notification condition."""
        # Simple condition evaluation
        try:
            # Replace variables in condition
            for key, value in context.items():
                condition = condition.replace(f"{{{key}}}", str(value))
            
            # Evaluate condition (simplified)
            return eval(condition)
        except Exception:
            return False
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notification history."""
        conn = sqlite3.connect("notifications.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM notifications 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'timestamp': row[1],
                'notification_type': row[2],
                'recipient': row[3],
                'subject': row[4],
                'body': row[5],
                'status': row[6],
                'priority': row[7],
                'template_id': row[8],
                'error_message': row[9]
            })
        
        return history
    
    def generate_notification_report(self) -> str:
        """Generate notification system report."""
        history = self.get_notification_history(100)
        
        # Calculate statistics
        total_notifications = len(history)
        sent_notifications = len([n for n in history if n['status'] == 'sent'])
        failed_notifications = len([n for n in history if n['status'] == 'failed'])
        
        # Group by type
        type_counts = {}
        for notification in history:
            notification_type = notification['notification_type']
            type_counts[notification_type] = type_counts.get(notification_type, 0) + 1
        
        # Group by priority
        priority_counts = {}
        for notification in history:
            priority = notification['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        report = f"""
BUL Notification System Report
=============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TEMPLATES
---------
Total Templates: {len(self.templates)}
"""
        
        for template_id, template in self.templates.items():
            report += f"""
{template.name} ({template_id}):
  Type: {template.notification_type.value}
  Priority: {template.priority.value}
  Variables: {', '.join(template.variables)}
"""
        
        report += f"""
RULES
-----
Total Rules: {len(self.rules)}
Enabled Rules: {len([r for r in self.rules.values() if r.enabled])}
"""
        
        for rule_id, rule in self.rules.items():
            report += f"""
{rule.name} ({rule_id}):
  Condition: {rule.condition}
  Template: {rule.template_id}
  Recipients: {len(rule.recipients)}
  Enabled: {rule.enabled}
  Cooldown: {rule.cooldown_minutes} minutes
"""
        
        report += f"""
STATISTICS
----------
Total Notifications: {total_notifications}
Sent: {sent_notifications}
Failed: {failed_notifications}
Success Rate: {(sent_notifications/total_notifications*100):.1f}% if total_notifications > 0 else 0

By Type:
{chr(10).join(f"  {ntype}: {count}" for ntype, count in type_counts.items())}

By Priority:
{chr(10).join(f"  {priority}: {count}" for priority, count in priority_counts.items())}

RECENT NOTIFICATIONS
-------------------
"""
        
        for notification in history[:10]:
            report += f"""
{notification['timestamp']}: {notification['notification_type']} to {notification['recipient']}
  Subject: {notification['subject']}
  Status: {notification['status']}
  Priority: {notification['priority']}
"""
        
        return report

def main():
    """Main notification system function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Notification System")
    parser.add_argument("--send", help="Send notification")
    parser.add_argument("--template", help="Send notification using template")
    parser.add_argument("--recipient", help="Notification recipient")
    parser.add_argument("--subject", help="Notification subject")
    parser.add_argument("--body", help="Notification body")
    parser.add_argument("--type", choices=['email', 'slack', 'teams', 'sms', 'webhook', 'log', 'file'],
                       default='email', help="Notification type")
    parser.add_argument("--priority", choices=['low', 'medium', 'high', 'critical'],
                       default='medium', help="Notification priority")
    parser.add_argument("--create-template", help="Create notification template")
    parser.add_argument("--create-rule", help="Create notification rule")
    parser.add_argument("--list-templates", action="store_true", help="List notification templates")
    parser.add_argument("--list-rules", action="store_true", help="List notification rules")
    parser.add_argument("--history", action="store_true", help="Show notification history")
    parser.add_argument("--report", action="store_true", help="Generate notification report")
    
    args = parser.parse_args()
    
    system = NotificationSystem()
    
    print("📧 BUL Notification System")
    print("=" * 40)
    
    if args.send:
        async def send_notification():
            try:
                result = await system.send_notification(
                    NotificationType(args.type),
                    args.recipient,
                    args.subject,
                    args.body,
                    NotificationPriority(args.priority)
                )
                print(f"✅ Notification sent: {result['status']}")
                if result['status'] == 'failed':
                    print(f"   Error: {result['error']}")
            except Exception as e:
                print(f"❌ Notification failed: {e}")
        
        asyncio.run(send_notification())
    
    elif args.template:
        async def send_template_notification():
            try:
                result = await system.send_template_notification(
                    args.template,
                    args.recipient,
                    {'timestamp': datetime.now().isoformat()}
                )
                print(f"✅ Template notification sent: {result['status']}")
                if result['status'] == 'failed':
                    print(f"   Error: {result['error']}")
            except Exception as e:
                print(f"❌ Template notification failed: {e}")
        
        asyncio.run(send_template_notification())
    
    elif args.create_template:
        template = system.create_template(
            template_id=args.create_template,
            name=f"Template {args.create_template}",
            notification_type=NotificationType(args.type),
            subject=args.subject or "Default Subject",
            body=args.body or "Default Body",
            priority=NotificationPriority(args.priority)
        )
        print(f"✅ Created template: {template.name}")
    
    elif args.create_rule:
        rule = system.create_rule(
            rule_id=args.create_rule,
            name=f"Rule {args.create_rule}",
            condition="True",
            template_id="system_startup",
            recipients=[args.recipient or "admin@example.com"]
        )
        print(f"✅ Created rule: {rule.name}")
    
    elif args.list_templates:
        templates = system.templates
        if templates:
            print(f"\n📋 Notification Templates ({len(templates)}):")
            print("-" * 50)
            for template_id, template in templates.items():
                print(f"{template.name} ({template_id}):")
                print(f"  Type: {template.notification_type.value}")
                print(f"  Priority: {template.priority.value}")
                print(f"  Variables: {', '.join(template.variables)}")
                print()
        else:
            print("No templates found.")
    
    elif args.list_rules:
        rules = system.rules
        if rules:
            print(f"\n📋 Notification Rules ({len(rules)}):")
            print("-" * 50)
            for rule_id, rule in rules.items():
                print(f"{rule.name} ({rule_id}):")
                print(f"  Condition: {rule.condition}")
                print(f"  Template: {rule.template_id}")
                print(f"  Recipients: {len(rule.recipients)}")
                print(f"  Enabled: {rule.enabled}")
                print()
        else:
            print("No rules found.")
    
    elif args.history:
        history = system.get_notification_history(20)
        if history:
            print(f"\n📜 Notification History ({len(history)}):")
            print("-" * 50)
            for notification in history:
                print(f"{notification['timestamp']}: {notification['notification_type']} to {notification['recipient']}")
                print(f"  Subject: {notification['subject']}")
                print(f"  Status: {notification['status']}")
                print(f"  Priority: {notification['priority']}")
                print()
        else:
            print("No notification history found.")
    
    elif args.report:
        report = system.generate_notification_report()
        print(report)
        
        # Save report
        report_file = f"notification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        print(f"📋 Templates: {len(system.templates)}")
        print(f"📋 Rules: {len(system.rules)}")
        print(f"📜 History: {len(system.get_notification_history())}")
        print(f"\n💡 Use --list-templates to see all templates")
        print(f"💡 Use --send to send a notification")
        print(f"💡 Use --template to send using a template")
        print(f"💡 Use --report to generate notification report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
