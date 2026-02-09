"""
Performance Monitor for Color Grading AI
=========================================

Advanced performance monitoring and optimization.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    operation: str
    duration: float
    timestamp: datetime
    success: bool
    resource_usage: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Advanced performance monitoring.
    
    Features:
    - Real-time performance tracking
    - Anomaly detection
    - Performance trends
    - Resource usage analysis
    - Alerting
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self._metrics: Dict[str, deque] = {}  # operation -> metrics queue
        self._alerts: List[Dict[str, Any]] = []
        self._thresholds = {
            "slow_operation": 10.0,  # seconds
            "high_cpu": 80.0,  # percent
            "high_memory": 80.0,  # percent
        }
    
    def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        resource_usage: Optional[Dict[str, float]] = None
    ):
        """
        Record a performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            resource_usage: Optional resource usage data
        """
        if operation not in self._metrics:
            self._metrics[operation] = deque(maxlen=self.window_size)
        
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=datetime.now(),
            success=success,
            resource_usage=resource_usage or {}
        )
        
        self._metrics[operation].append(metric)
        
        # Check for anomalies
        self._check_anomalies(operation, metric)
    
    def _check_anomalies(self, operation: str, metric: PerformanceMetric):
        """Check for performance anomalies."""
        # Check slow operation
        if metric.duration > self._thresholds["slow_operation"]:
            self._add_alert(
                "slow_operation",
                f"Operation {operation} took {metric.duration:.2f}s (threshold: {self._thresholds['slow_operation']}s)",
                {"operation": operation, "duration": metric.duration}
            )
        
        # Check resource usage
        if metric.resource_usage:
            cpu = metric.resource_usage.get("cpu_percent", 0)
            memory = metric.resource_usage.get("memory_percent", 0)
            
            if cpu > self._thresholds["high_cpu"]:
                self._add_alert(
                    "high_cpu",
                    f"High CPU usage: {cpu:.1f}% during {operation}",
                    {"operation": operation, "cpu": cpu}
                )
            
            if memory > self._thresholds["high_memory"]:
                self._add_alert(
                    "high_memory",
                    f"High memory usage: {memory:.1f}% during {operation}",
                    {"operation": operation, "memory": memory}
                )
    
    def _add_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """Add performance alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self._alerts.append(alert)
        logger.warning(f"Performance alert: {message}")
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Statistics dictionary
        """
        metrics = list(self._metrics.get(operation, []))
        
        if not metrics:
            return {}
        
        durations = [m.duration for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        
        return {
            "operation": operation,
            "count": len(metrics),
            "success_rate": success_count / len(metrics) if metrics else 0,
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
            "p95_duration": self._percentile(durations, 95),
            "p99_duration": self._percentile(durations, 99),
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations."""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self._metrics.keys()
        }
    
    def get_trends(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends.
        
        Args:
            operation: Operation name
            hours: Number of hours to analyze
            
        Returns:
            Trends dictionary
        """
        metrics = list(self._metrics.get(operation, []))
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in metrics if m.timestamp > cutoff]
        
        if len(recent_metrics) < 2:
            return {}
        
        # Calculate trend
        recent_avg = statistics.mean([m.duration for m in recent_metrics[-10:]])
        older_avg = statistics.mean([m.duration for m in recent_metrics[:-10]]) if len(recent_metrics) > 10 else recent_avg
        
        trend = "stable"
        if recent_avg > older_avg * 1.1:
            trend = "degrading"
        elif recent_avg < older_avg * 0.9:
            trend = "improving"
        
        return {
            "operation": operation,
            "trend": trend,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "change_percent": ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0,
        }
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self._alerts[-limit:]
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set performance threshold."""
        if threshold_name in self._thresholds:
            self._thresholds[threshold_name] = value
            logger.info(f"Updated threshold {threshold_name} to {value}")




