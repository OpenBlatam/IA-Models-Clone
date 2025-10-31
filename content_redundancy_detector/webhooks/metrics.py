"""
Webhook Metrics - Advanced metrics collection and analysis
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a new metric point"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get the latest metric point"""
        return self.points[-1] if self.points else None
    
    def get_average(self, duration_seconds: int = 300) -> float:
        """Get average value over duration"""
        cutoff = time.time() - duration_seconds
        recent_points = [p for p in self.points if p.timestamp >= cutoff]
        if not recent_points:
            return 0.0
        return sum(p.value for p in recent_points) / len(recent_points)
    
    def get_count(self, duration_seconds: int = 300) -> int:
        """Get count of points over duration"""
        cutoff = time.time() - duration_seconds
        return sum(1 for p in self.points if p.timestamp >= cutoff)


class WebhookMetricsCollector:
    """
    Advanced metrics collector for webhook system
    
    Features:
    - Time series metrics
    - Real-time calculations
    - Performance analysis
    - Alerting thresholds
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to keep metrics data
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, MetricSeries] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Initialize core metrics
        self._init_core_metrics()
    
    def _init_core_metrics(self):
        """Initialize core metric series"""
        core_metrics = [
            "webhook_deliveries_total",
            "webhook_delivery_duration_seconds",
            "webhook_queue_size",
            "webhook_circuit_breaker_state",
            "webhook_rate_limit_hits",
            "webhook_validation_errors",
            "webhook_retry_attempts",
            "webhook_worker_utilization"
        ]
        
        for metric in core_metrics:
            self.metrics[metric] = MetricSeries(name=metric)
        
    def record_delivery(
        self,
        status: str,
        event_type: str,
        duration: float,
        endpoint_id: str
    ) -> None:
        """Record webhook delivery metric"""
        labels = {
            "status": status,
            "event_type": event_type,
            "endpoint_id": endpoint_id
        }
        
        # Record total deliveries
        self.metrics["webhook_deliveries_total"].add_point(1.0, labels)
        
        # Record duration
        self.metrics["webhook_delivery_duration_seconds"].add_point(duration, labels)
        
        # Check thresholds
        self._check_delivery_thresholds(status, duration, endpoint_id)
    
    def record_queue_size(self, size: int) -> None:
        """Record current queue size"""
        self.metrics["webhook_queue_size"].add_point(float(size))
        
        # Check queue threshold
        if size > self.thresholds.get("queue_size", {}).get("warning", 100):
            self._create_alert("queue_size_warning", f"Queue size high: {size}")
    
    def record_circuit_breaker_state(
        self,
        endpoint_id: str,
        state: str,
        failure_rate: float
    ) -> None:
        """Record circuit breaker state"""
        labels = {
            "endpoint_id": endpoint_id,
            "state": state
        }
        
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
        self.metrics["webhook_circuit_breaker_state"].add_point(
            float(state_value), labels
        )
        
        # Check circuit breaker threshold
        if state == "open" and failure_rate > 0.5:
            self._create_alert(
                "circuit_breaker_open",
                f"Circuit breaker open for {endpoint_id}: {failure_rate:.1%} failure rate"
            )
    
    def record_rate_limit_hit(self, endpoint_id: str, limit_type: str) -> None:
        """Record rate limit hit"""
        labels = {
            "endpoint_id": endpoint_id,
            "limit_type": limit_type
        }
        self.metrics["webhook_rate_limit_hits"].add_point(1.0, labels)
    
    def record_validation_error(self, error_type: str, endpoint_id: str) -> None:
        """Record validation error"""
        labels = {
            "error_type": error_type,
            "endpoint_id": endpoint_id
        }
        self.metrics["webhook_validation_errors"].add_point(1.0, labels)
    
    def record_retry_attempt(self, endpoint_id: str, attempt_number: int) -> None:
        """Record retry attempt"""
        labels = {
            "endpoint_id": endpoint_id,
            "attempt": str(attempt_number)
        }
        self.metrics["webhook_retry_attempts"].add_point(1.0, labels)
    
    def record_worker_utilization(self, worker_id: str, utilization: float) -> None:
        """Record worker utilization"""
        labels = {"worker_id": worker_id}
        self.metrics["webhook_worker_utilization"].add_point(utilization, labels)
    
    def set_threshold(
        self,
        metric_name: str,
        threshold_type: str,
        value: float
    ) -> None:
        """Set alert threshold for metric"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][threshold_type] = value
        logger.info(f"Set {threshold_type} threshold for {metric_name}: {value}")
    
    def _check_delivery_thresholds(
        self,
        status: str,
        duration: float,
        endpoint_id: str
    ) -> None:
        """Check delivery-related thresholds"""
        # Check duration threshold
        duration_threshold = self.thresholds.get("webhook_delivery_duration_seconds", {})
        if duration > duration_threshold.get("warning", 5.0):
            self._create_alert(
                "slow_delivery",
                f"Slow webhook delivery: {duration:.2f}s for {endpoint_id}"
            )
        
        # Check failure rate
        if status == "failed":
            failure_rate = self._calculate_failure_rate(endpoint_id)
            if failure_rate > self.thresholds.get("failure_rate", {}).get("critical", 0.8):
                self._create_alert(
                    "high_failure_rate",
                    f"High failure rate for {endpoint_id}: {failure_rate:.1%}"
                )
    
    def _calculate_failure_rate(self, endpoint_id: str, duration_seconds: int = 300) -> float:
        """Calculate failure rate for endpoint"""
        cutoff = time.time() - duration_seconds
        deliveries = self.metrics["webhook_deliveries_total"]
        
        recent_deliveries = [
            p for p in deliveries.points
            if p.timestamp >= cutoff and p.labels.get("endpoint_id") == endpoint_id
        ]
        
        if not recent_deliveries:
            return 0.0
        
        failed_count = sum(
            1 for p in recent_deliveries
            if p.labels.get("status") == "failed"
        )
        
        return failed_count / len(recent_deliveries)
    
    def _create_alert(self, alert_type: str, message: str) -> None:
        """Create alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "severity": "warning"
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")
    
    def get_metrics_summary(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = {
            "timestamp": time.time(),
            "duration_seconds": duration_seconds,
            "metrics": {},
            "alerts": self.get_recent_alerts(duration_seconds),
            "health_score": self._calculate_health_score()
        }
        
        for name, series in self.metrics.items():
            summary["metrics"][name] = {
                "count": series.get_count(duration_seconds),
                "average": series.get_average(duration_seconds),
                "latest": series.get_latest().value if series.get_latest() else 0
            }
        
        return summary
    
    def get_recent_alerts(self, duration_seconds: int = 300) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = time.time() - duration_seconds
        return [
            alert for alert in self.alerts
            if alert["timestamp"] >= cutoff
        ]
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        # Deduct for high failure rate
        failure_rate = self._get_overall_failure_rate()
        score -= failure_rate * 30  # Up to 30 points for failure rate
        
        # Deduct for circuit breakers open
        open_circuits = self._count_open_circuit_breakers()
        score -= open_circuits * 10  # 10 points per open circuit
        
        # Deduct for high queue size
        queue_size = self.metrics["webhook_queue_size"].get_latest()
        if queue_size and queue_size.value > 50:
            score -= min(20, (queue_size.value - 50) * 0.5)
        
        return max(0.0, min(100.0, score))
    
    def _get_overall_failure_rate(self, duration_seconds: int = 300) -> float:
        """Get overall failure rate"""
        cutoff = time.time() - duration_seconds
        deliveries = self.metrics["webhook_deliveries_total"]
        
        recent_deliveries = [
            p for p in deliveries.points if p.timestamp >= cutoff
        ]
        
        if not recent_deliveries:
            return 0.0
        
        failed_count = sum(
            1 for p in recent_deliveries
            if p.labels.get("status") == "failed"
        )
        
        return failed_count / len(recent_deliveries)
    
    def _count_open_circuit_breakers(self) -> int:
        """Count open circuit breakers"""
        cb_metrics = self.metrics["webhook_circuit_breaker_state"]
        latest_points = {}
        
        # Get latest state for each endpoint
        for point in cb_metrics.points:
            endpoint_id = point.labels.get("endpoint_id")
            if endpoint_id:
                latest_points[endpoint_id] = point.value
        
        # Count open circuits (value = 2)
        return sum(1 for value in latest_points.values() if value == 2)
    
    def cleanup_old_data(self) -> None:
        """Clean up old metric data"""
        cutoff = time.time() - (self.retention_hours * 3600)
        
        for series in self.metrics.values():
            # Remove old points
            while series.points and series.points[0].timestamp < cutoff:
                series.points.popleft()
        
        # Remove old alerts
        self.alerts = [
            alert for alert in self.alerts
            if alert["timestamp"] >= cutoff
        ]
        
        logger.debug(f"Cleaned up metrics data older than {self.retention_hours} hours")


# Global metrics collector instance
metrics_collector = WebhookMetricsCollector()