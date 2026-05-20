#!/usr/bin/env python3
"""
API Notifier
============
Notification system for API alerts and events.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with notification capabilities
"""
import warnings

warnings.warn(
    "api_notifier.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Notification:
    """Notification data structure."""
    level: str  # info, warning, critical
    title: str
    message: str
    timestamp: str
    metadata: Dict[str, Any]


class NotificationHandler:
    """Base class for notification handlers."""
    
    def send(self, notification: Notification):
        """Send notification."""
        raise NotImplementedError


class ConsoleNotificationHandler(NotificationHandler):
    """Console notification handler."""
    
    def send(self, notification: Notification):
        """Print notification to console."""
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        
        print(f"\n{emoji.get(notification.level, '📢')} {notification.title}")
        print(f"   {notification.message}")
        print(f"   Time: {notification.timestamp}")


class FileNotificationHandler(NotificationHandler):
    """File notification handler."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
    
    def send(self, notification: Notification):
        """Write notification to file."""
        notifications = []
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                notifications = json.load(f)
        
        notifications.append(asdict(notification))
        
        with open(self.file_path, "w") as f:
            json.dump(notifications, f, indent=2)


class APINotifier:
    """API notification system."""
    
    def __init__(self):
        self.handlers: List[NotificationHandler] = [ConsoleNotificationHandler()]
        self.notifications: List[Notification] = []
    
    def add_handler(self, handler: NotificationHandler):
        """Add notification handler."""
        self.handlers.append(handler)
    
    def notify(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send notification."""
        notification = Notification(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.notifications.append(notification)
        
        for handler in self.handlers:
            handler.send(notification)
        
        return notification
    
    def notify_health_status(self, status: str, details: Dict[str, Any]):
        """Notify health status change."""
        if status == "unhealthy":
            self.notify(
                level="critical",
                title="API Health Critical",
                message=f"API is unhealthy: {details.get('error', 'Unknown error')}",
                metadata=details
            )
        elif status == "degraded":
            self.notify(
                level="warning",
                title="API Health Degraded",
                message="API is in degraded state",
                metadata=details
            )
    
    def notify_performance(self, metric: str, value: float, threshold: float):
        """Notify performance issue."""
        if value > threshold * 2:
            self.notify(
                level="critical",
                title="Performance Critical",
                message=f"{metric} is critically high: {value:.2f} (threshold: {threshold:.2f})",
                metadata={"metric": metric, "value": value, "threshold": threshold}
            )
        elif value > threshold:
            self.notify(
                level="warning",
                title="Performance Warning",
                message=f"{metric} exceeds threshold: {value:.2f} (threshold: {threshold:.2f})",
                metadata={"metric": metric, "value": value, "threshold": threshold}
            )
    
    def notify_test_results(self, passed: int, failed: int, total: int):
        """Notify test results."""
        if failed > 0:
            self.notify(
                level="warning" if failed < total * 0.1 else "critical",
                title="Test Results",
                message=f"Tests: {passed} passed, {failed} failed out of {total}",
                metadata={"passed": passed, "failed": failed, "total": total}
            )
        else:
            self.notify(
                level="info",
                title="Test Results",
                message=f"All {total} tests passed",
                metadata={"passed": passed, "failed": failed, "total": total}
            )
    
    def export_notifications(self, file_path: Path):
        """Export notifications."""
        data = {
            "notifications": [asdict(n) for n in self.notifications],
            "total": len(self.notifications),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Notifications exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Notifier")
    parser.add_argument("--level", choices=["info", "warning", "critical"], default="info")
    parser.add_argument("--title", required=True, help="Notification title")
    parser.add_argument("--message", required=True, help="Notification message")
    parser.add_argument("--export", help="Export notifications to file")
    
    args = parser.parse_args()
    
    notifier = APINotifier()
    
    if args.export:
        file_handler = FileNotificationHandler(Path(args.export))
        notifier.add_handler(file_handler)
    
    notifier.notify(
        level=args.level,
        title=args.title,
        message=args.message
    )
    
    if args.export:
        notifier.export_notifications(Path(args.export))


if __name__ == "__main__":
    main()



