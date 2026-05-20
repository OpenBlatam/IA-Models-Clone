"""
Monitoring Utilities
===================

Utilities for monitoring and alerting.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert data"""
    level: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AlertManager:
    """
    Manager for alerts and notifications.
    
    Features:
    - Alert tracking
    - Alert filtering
    - Alert callbacks
    """
    
    def __init__(self):
        """Initialize alert manager"""
        self.alerts: deque = deque(maxlen=1000)
        self.callbacks: Dict[str, list[Callable]] = {
            "info": [],
            "warning": [],
            "error": [],
            "critical": []
        }
    
    def register_callback(
        self,
        level: str,
        callback: Callable[[Alert], None]
    ) -> None:
        """
        Register callback for alert level.
        
        Args:
            level: Alert level
            callback: Callback function
        """
        if level in self.callbacks:
            self.callbacks[level].append(callback)
    
    def alert(
        self,
        level: str,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Create and process an alert.
        
        Args:
            level: Alert level
            message: Alert message
            metadata: Optional metadata
        """
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Call registered callbacks
        for callback in self.callbacks.get(level, []):
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Log alert
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, f"Alert [{level}]: {message}")
    
    def get_recent_alerts(
        self,
        level: Optional[str] = None,
        limit: int = 10
    ) -> list[Alert]:
        """
        Get recent alerts.
        
        Args:
            level: Optional level filter
            limit: Maximum number of alerts
            
        Returns:
            List of recent alerts
        """
        alerts = list(self.alerts)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    def clear_alerts(self) -> int:
        """
        Clear all alerts.
        
        Returns:
            Number of alerts cleared
        """
        count = len(self.alerts)
        self.alerts.clear()
        return count


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


class SystemMonitor:
    """
    System monitor for tracking system health.
    
    Features:
    - Resource monitoring
    - Performance tracking
    - Alert generation
    """
    
    def __init__(self):
        """Initialize system monitor"""
        self.alert_manager = get_alert_manager()
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check system resources.
        
        Returns:
            Dictionary with resource information
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Alert if resources are high
            if cpu_percent > 80:
                self.alert_manager.alert(
                    "warning",
                    f"High CPU usage: {cpu_percent}%",
                    {"cpu_percent": cpu_percent}
                )
            
            if memory.percent > 85:
                self.alert_manager.alert(
                    "warning",
                    f"High memory usage: {memory.percent}%",
                    {"memory_percent": memory.percent}
                )
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                }
            }
        except ImportError:
            logger.warning("psutil not available, skipping resource check")
            return {}
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return {}


# Global system monitor
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor

