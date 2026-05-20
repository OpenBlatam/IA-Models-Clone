#!/usr/bin/env python3
"""
API Alerts
==========
Alert system for API monitoring and health checks.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with alert capabilities
"""
import warnings

warnings.warn(
    "api_alerts.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
import time
import requests
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: str
    endpoint: Optional[str] = None


class AlertHandler:
    """Base class for alert handlers."""
    
    def handle(self, alert: Alert):
        """Handle an alert."""
        raise NotImplementedError


class ConsoleAlertHandler(AlertHandler):
    """Console alert handler."""
    
    def handle(self, alert: Alert):
        """Print alert to console."""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨"
        }
        
        color = {
            AlertLevel.INFO: "\033[94m",  # Blue
            AlertLevel.WARNING: "\033[93m",  # Yellow
            AlertLevel.CRITICAL: "\033[91m"  # Red
        }
        
        reset = "\033[0m"
        
        print(f"{emoji[alert.level]} {color[alert.level]}[{alert.level.value.upper()}]{reset} {alert.message}")
        print(f"   Metric: {alert.metric}")
        print(f"   Value: {alert.value:.2f} (Threshold: {alert.threshold:.2f})")
        if alert.endpoint:
            print(f"   Endpoint: {alert.endpoint}")
        print(f"   Time: {alert.timestamp}")
        print()


class FileAlertHandler(AlertHandler):
    """File alert handler."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
    
    def handle(self, alert: Alert):
        """Write alert to file."""
        alerts = []
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                alerts = json.load(f)
        
        alerts.append(asdict(alert))
        
        with open(self.file_path, "w") as f:
            json.dump(alerts, f, indent=2)


class APIAlertSystem:
    """API alert system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.handlers: List[AlertHandler] = [ConsoleAlertHandler()]
        self.alerts: List[Alert] = []
        self.running = False
    
    def add_handler(self, handler: AlertHandler):
        """Add alert handler."""
        self.handlers.append(handler)
    
    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric: str,
        value: float,
        threshold: float,
        endpoint: Optional[str] = None
    ):
        """Create and handle alert."""
        alert = Alert(
            level=level,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            timestamp=datetime.now().isoformat(),
            endpoint=endpoint
        )
        
        self.alerts.append(alert)
        
        for handler in self.handlers:
            handler.handle(alert)
        
        return alert
    
    def check_response_time(self, endpoint: str = "/health", threshold: float = 1000.0):
        """Check response time and alert if exceeded."""
        start = time.time()
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            response_time = (time.time() - start) * 1000
            
            if response_time > threshold:
                level = AlertLevel.CRITICAL if response_time > threshold * 2 else AlertLevel.WARNING
                self.create_alert(
                    level=level,
                    message=f"Response time exceeded threshold",
                    metric="response_time",
                    value=response_time,
                    threshold=threshold,
                    endpoint=endpoint
                )
                return False
            
            return True
        except Exception as e:
            self.create_alert(
                level=AlertLevel.CRITICAL,
                message=f"Endpoint unreachable: {str(e)}",
                metric="availability",
                value=0,
                threshold=1,
                endpoint=endpoint
            )
            return False
    
    def check_health_status(self, threshold_healthy: float = 95.0):
        """Check health status and alert if degraded."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code != 200:
                self.create_alert(
                    level=AlertLevel.CRITICAL,
                    message="Health endpoint returned non-200 status",
                    metric="health_status",
                    value=0,
                    threshold=threshold_healthy,
                    endpoint="/health"
                )
                return False
            
            data = response.json()
            status = data.get("status", "unknown")
            
            if status not in ["healthy", "ok", "up"]:
                self.create_alert(
                    level=AlertLevel.WARNING,
                    message=f"Health status is {status}",
                    metric="health_status",
                    value=50 if status == "degraded" else 0,
                    threshold=threshold_healthy,
                    endpoint="/health"
                )
                return False
            
            return True
        except Exception as e:
            self.create_alert(
                level=AlertLevel.CRITICAL,
                message=f"Health check failed: {str(e)}",
                metric="health_availability",
                value=0,
                threshold=1,
                endpoint="/health"
            )
            return False
    
    def monitor_continuous(
        self,
        interval: float = 60.0,
        checks: Optional[List[Callable]] = None
    ):
        """Monitor continuously with alerts."""
        self.running = True
        
        if checks is None:
            checks = [
                lambda: self.check_health_status(),
                lambda: self.check_response_time()
            ]
        
        print(f"🔔 Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while self.running:
                for check in checks:
                    check()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.running = False
            print("\n🛑 Monitoring stopped")
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary."""
        critical = [a for a in self.alerts if a.level == AlertLevel.CRITICAL]
        warnings = [a for a in self.alerts if a.level == AlertLevel.WARNING]
        info = [a for a in self.alerts if a.level == AlertLevel.INFO]
        
        return {
            "total": len(self.alerts),
            "critical": len(critical),
            "warnings": len(warnings),
            "info": len(info),
            "recent_alerts": [asdict(a) for a in self.alerts[-10:]]
        }
    
    def export_alerts(self, file_path: Path):
        """Export alerts to file."""
        data = {
            "alerts": [asdict(a) for a in self.alerts],
            "summary": self.get_alerts_summary(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Alerts exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Alert System")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--interval", type=float, default=60.0, help="Monitoring interval (seconds)")
    parser.add_argument("--check", action="store_true", help="Run single check")
    parser.add_argument("--export", help="Export alerts to file")
    
    args = parser.parse_args()
    
    alert_system = APIAlertSystem(base_url=args.url)
    
    if args.export:
        file_handler = FileAlertHandler(Path(args.export))
        alert_system.add_handler(file_handler)
    
    if args.check:
        print("🔍 Running health checks...")
        alert_system.check_health_status()
        alert_system.check_response_time()
    elif args.monitor:
        alert_system.monitor_continuous(interval=args.interval)
    else:
        # Default: single check
        alert_system.check_health_status()
        alert_system.check_response_time()
    
    if args.export:
        alert_system.export_alerts(Path(args.export))


if __name__ == "__main__":
    main()



