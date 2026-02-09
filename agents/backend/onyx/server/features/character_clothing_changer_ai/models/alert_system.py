"""
Alert System for Flux2 Clothing Changer
========================================

Alert and notification system for monitoring.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert information."""
    level: AlertLevel
    title: str
    message: str
    timestamp: float
    source: str
    metadata: Dict[str, Any] = None
    acknowledged: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AlertSystem:
    """Alert and notification system."""
    
    def __init__(
        self,
        history_size: int = 1000,
        enable_notifications: bool = True,
    ):
        """
        Initialize alert system.
        
        Args:
            history_size: Maximum number of alerts to keep
            enable_notifications: Enable notifications
        """
        self.history_size = history_size
        self.enable_notifications = enable_notifications
        
        self.alerts: deque = deque(maxlen=history_size)
        self.handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        
        # Alert thresholds
        self.thresholds = {
            "error_rate": 0.1,  # 10%
            "processing_time": 30.0,  # 30 seconds
            "memory_usage": 8000.0,  # 8GB
            "gpu_memory": 20000.0,  # 20GB
            "quality_score": 0.5,  # Below 0.5
        }
    
    def register_handler(
        self,
        level: AlertLevel,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register an alert handler.
        
        Args:
            level: Alert level
            handler: Handler function
        """
        self.handlers[level].append(handler)
        logger.info(f"Registered handler for {level.value} alerts")
    
    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Send an alert.
        
        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            source: Alert source
            metadata: Optional metadata
            
        Returns:
            Created alert
        """
        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {},
        )
        
        self.alerts.append(alert)
        
        # Trigger handlers
        if self.enable_notifications:
            for handler in self.handlers[level]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
        
        logger.log(
            getattr(logging, level.value.upper(), logging.INFO),
            f"Alert [{level.value}]: {title} - {message}"
        )
        
        return alert
    
    def check_thresholds(
        self,
        metrics: Dict[str, float],
        source: str = "monitoring",
    ) -> List[Alert]:
        """
        Check metrics against thresholds and send alerts.
        
        Args:
            metrics: Dictionary of metric names and values
            source: Alert source
            
        Returns:
            List of created alerts
        """
        alerts = []
        
        # Check error rate
        if "error_rate" in metrics:
            if metrics["error_rate"] > self.thresholds["error_rate"]:
                alerts.append(self.send_alert(
                    level=AlertLevel.ERROR,
                    title="High Error Rate",
                    message=f"Error rate is {metrics['error_rate']:.2%}, threshold: {self.thresholds['error_rate']:.2%}",
                    source=source,
                    metadata={"error_rate": metrics["error_rate"]},
                ))
        
        # Check processing time
        if "processing_time" in metrics:
            if metrics["processing_time"] > self.thresholds["processing_time"]:
                alerts.append(self.send_alert(
                    level=AlertLevel.WARNING,
                    title="Slow Processing",
                    message=f"Processing time is {metrics['processing_time']:.2f}s, threshold: {self.thresholds['processing_time']:.2f}s",
                    source=source,
                    metadata={"processing_time": metrics["processing_time"]},
                ))
        
        # Check memory usage
        if "memory_usage_mb" in metrics:
            if metrics["memory_usage_mb"] > self.thresholds["memory_usage"]:
                alerts.append(self.send_alert(
                    level=AlertLevel.WARNING,
                    title="High Memory Usage",
                    message=f"Memory usage is {metrics['memory_usage_mb']:.2f}MB, threshold: {self.thresholds['memory_usage']:.2f}MB",
                    source=source,
                    metadata={"memory_usage": metrics["memory_usage_mb"]},
                ))
        
        # Check GPU memory
        if "gpu_memory_mb" in metrics:
            if metrics["gpu_memory_mb"] > self.thresholds["gpu_memory"]:
                alerts.append(self.send_alert(
                    level=AlertLevel.WARNING,
                    title="High GPU Memory Usage",
                    message=f"GPU memory is {metrics['gpu_memory_mb']:.2f}MB, threshold: {self.thresholds['gpu_memory']:.2f}MB",
                    source=source,
                    metadata={"gpu_memory": metrics["gpu_memory_mb"]},
                ))
        
        # Check quality score
        if "quality_score" in metrics:
            if metrics["quality_score"] < self.thresholds["quality_score"]:
                alerts.append(self.send_alert(
                    level=AlertLevel.ERROR,
                    title="Low Quality Score",
                    message=f"Quality score is {metrics['quality_score']:.2f}, threshold: {self.thresholds['quality_score']:.2f}",
                    source=source,
                    metadata={"quality_score": metrics["quality_score"]},
                ))
        
        return alerts
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        unacknowledged_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Alert]:
        """
        Get alerts.
        
        Args:
            level: Filter by level
            unacknowledged_only: Only unacknowledged alerts
            limit: Maximum number of alerts
            
        Returns:
            List of alerts
        """
        alerts = list(self.alerts)
        
        # Filter by level
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        # Filter by acknowledged
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def acknowledge_alert(self, alert: Alert) -> None:
        """Acknowledge an alert."""
        alert.acknowledged = True
        logger.info(f"Alert acknowledged: {alert.title}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self.alerts)
        
        by_level = {}
        for level in AlertLevel:
            by_level[level.value] = sum(
                1 for a in self.alerts if a.level == level
            )
        
        unacknowledged = sum(1 for a in self.alerts if not a.acknowledged)
        
        return {
            "total_alerts": total,
            "by_level": by_level,
            "unacknowledged": unacknowledged,
            "acknowledged": total - unacknowledged,
        }
    
    def set_threshold(
        self,
        metric_name: str,
        threshold: float,
    ) -> None:
        """
        Set threshold for a metric.
        
        Args:
            metric_name: Metric name
            threshold: Threshold value
        """
        if metric_name in self.thresholds:
            self.thresholds[metric_name] = threshold
            logger.info(f"Set threshold for {metric_name}: {threshold}")
        else:
            logger.warning(f"Unknown metric: {metric_name}")


