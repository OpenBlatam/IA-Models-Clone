"""
Alert Manager for Piel Mejorador AI SAM3
=========================================

Manages alerts and notifications for system events.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types."""
    MEMORY_HIGH = "memory_high"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TASK_FAILURE_RATE = "task_failure_rate"
    CACHE_MISS_RATE = "cache_miss_rate"
    WEBHOOK_DELIVERY_FAILED = "webhook_delivery_failed"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Alert data structure."""
    type: AlertType
    level: AlertLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class AlertManager:
    """
    Manages system alerts and notifications.
    
    Features:
    - Alert threshold management
    - Alert handlers
    - Alert history
    - Auto-resolution
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self._alerts: List[Alert] = []
        self._handlers: Dict[AlertType, List[Callable]] = {}
        self._thresholds: Dict[AlertType, Dict[str, float]] = {
            AlertType.MEMORY_HIGH: {"threshold": 85.0},
            AlertType.RATE_LIMIT_EXCEEDED: {"threshold": 0.1},  # 10% rate limit rate
            AlertType.TASK_FAILURE_RATE: {"threshold": 0.2},  # 20% failure rate
            AlertType.CACHE_MISS_RATE: {"threshold": 0.5},  # 50% miss rate
            AlertType.WEBHOOK_DELIVERY_FAILED: {"threshold": 0.3},  # 30% failure rate
        }
    
    def register_handler(self, alert_type: AlertType, handler: Callable):
        """
        Register alert handler.
        
        Args:
            alert_type: Type of alert
            handler: Handler function (alert: Alert) -> None
        """
        if alert_type not in self._handlers:
            self._handlers[alert_type] = []
        self._handlers[alert_type].append(handler)
        logger.info(f"Registered handler for {alert_type.value}")
    
    def check_and_alert(
        self,
        alert_type: AlertType,
        current_value: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Check threshold and create alert if needed.
        
        Args:
            alert_type: Type of alert
            current_value: Current metric value
            message: Alert message
            details: Additional details
            
        Returns:
            Alert if created, None otherwise
        """
        threshold_config = self._thresholds.get(alert_type, {})
        threshold = threshold_config.get("threshold", 0)
        
        # Determine level based on severity
        if current_value >= threshold:
            level = AlertLevel.WARNING
            if current_value >= threshold * 1.5:
                level = AlertLevel.ERROR
            if current_value >= threshold * 2.0:
                level = AlertLevel.CRITICAL
            
            alert = Alert(
                type=alert_type,
                level=level,
                message=message,
                details=details or {}
            )
            
            self._alerts.append(alert)
            
            # Trigger handlers
            handlers = self._handlers.get(alert_type, [])
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.warning(f"Alert triggered: {alert_type.value} - {message}")
            return alert
        
        return None
    
    def check_memory_alert(self, memory_percent: float) -> Optional[Alert]:
        """Check memory usage and alert if high."""
        return self.check_and_alert(
            AlertType.MEMORY_HIGH,
            memory_percent,
            f"Memory usage is high: {memory_percent:.1f}%",
            {"memory_percent": memory_percent}
        )
    
    def check_rate_limit_alert(self, rate_limit_rate: float) -> Optional[Alert]:
        """Check rate limit rate and alert if high."""
        return self.check_and_alert(
            AlertType.RATE_LIMIT_EXCEEDED,
            rate_limit_rate,
            f"Rate limit rate is high: {rate_limit_rate:.1%}",
            {"rate_limit_rate": rate_limit_rate}
        )
    
    def check_task_failure_alert(self, failure_rate: float) -> Optional[Alert]:
        """Check task failure rate and alert if high."""
        return self.check_and_alert(
            AlertType.TASK_FAILURE_RATE,
            failure_rate,
            f"Task failure rate is high: {failure_rate:.1%}",
            {"failure_rate": failure_rate}
        )
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            level: Optional filter by level
            
        Returns:
            List of active alerts
        """
        alerts = [a for a in self._alerts if not a.resolved]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts
    
    def resolve_alert(self, alert: Alert):
        """Mark alert as resolved."""
        alert.resolved = True
        logger.info(f"Alert resolved: {alert.type.value}")
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self._alerts[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self._alerts)
        active = len([a for a in self._alerts if not a.resolved])
        
        by_level = {}
        for level in AlertLevel:
            by_level[level.value] = len([
                a for a in self._alerts
                if a.level == level and not a.resolved
            ])
        
        return {
            "total_alerts": total,
            "active_alerts": active,
            "resolved_alerts": total - active,
            "by_level": by_level,
        }




