"""
Real-time monitoring system for KV cache.

This module provides real-time monitoring, alerting, and dashboards
for cache performance and health.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert."""
    alert_id: str
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class MetricSnapshot:
    """A metric snapshot."""
    timestamp: float
    metrics: Dict[str, float]
    cache_size: int
    hit_rate: float
    miss_rate: float


class RealTimeMonitor:
    """Real-time cache monitor."""
    
    def __init__(self, cache: Any, update_interval: float = 1.0):
        self.cache = cache
        self.update_interval = update_interval
        self._snapshots: deque = deque(maxlen=1000)
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Thresholds
        self._thresholds = {
            'hit_rate_min': 0.7,
            'memory_usage_max': 0.9,
            'latency_max': 0.1,
            'error_rate_max': 0.01
        }
        
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self) -> None:
        """Monitoring loop."""
        while self._running:
            try:
                snapshot = self._capture_snapshot()
                self._snapshots.append(snapshot)
                self._check_thresholds(snapshot)
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
                
    def _capture_snapshot(self) -> MetricSnapshot:
        """Capture current metrics snapshot."""
        metrics = {}
        
        if hasattr(self.cache, 'stats'):
            stats = self.cache.stats
            metrics['hit_rate'] = getattr(stats, 'hit_rate', 0.0)
            metrics['miss_rate'] = getattr(stats, 'miss_rate', 0.0)
            metrics['total_requests'] = getattr(stats, 'total_requests', 0)
            
        if hasattr(self.cache, '_cache'):
            cache_size = len(self.cache._cache)
            if hasattr(self.cache, 'max_size'):
                max_size = self.cache.max_size
                metrics['memory_usage'] = cache_size / max_size if max_size > 0 else 0.0
        else:
            cache_size = 0
            
        return MetricSnapshot(
            timestamp=time.time(),
            metrics=metrics,
            cache_size=cache_size,
            hit_rate=metrics.get('hit_rate', 0.0),
            miss_rate=metrics.get('miss_rate', 0.0)
        )
        
    def _check_thresholds(self, snapshot: MetricSnapshot) -> None:
        """Check if metrics exceed thresholds."""
        # Check hit rate
        if snapshot.hit_rate < self._thresholds['hit_rate_min']:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Low hit rate: {snapshot.hit_rate:.2%}",
                "hit_rate",
                snapshot.hit_rate,
                self._thresholds['hit_rate_min']
            )
            
        # Check memory usage
        memory_usage = snapshot.metrics.get('memory_usage', 0.0)
        if memory_usage > self._thresholds['memory_usage_max']:
            self._trigger_alert(
                AlertLevel.ERROR,
                f"High memory usage: {memory_usage:.2%}",
                "memory_usage",
                memory_usage,
                self._thresholds['memory_usage_max']
            )
            
    def _trigger_alert(
        self,
        level: AlertLevel,
        message: str,
        metric: str,
        value: float,
        threshold: float
    ) -> None:
        """Trigger an alert."""
        import uuid
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            level=level,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold
        )
        
        with self._lock:
            self._alerts.append(alert)
            # Keep only last 1000 alerts
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]
                
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
                
    def subscribe_alerts(self, callback: Callable[[Alert], None]) -> None:
        """Subscribe to alerts."""
        with self._lock:
            self._alert_callbacks.append(callback)
            
    def get_snapshots(self, window_seconds: Optional[float] = None) -> List[MetricSnapshot]:
        """Get metric snapshots."""
        with self._lock:
            snapshots = list(self._snapshots)
            
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
            
        return snapshots
        
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """Get alerts."""
        with self._lock:
            alerts = list(self._alerts)
            
        if level:
            alerts = [a for a in alerts if a.level == level]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
            
        return alerts
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False
            
    def set_threshold(self, metric: str, threshold: float) -> None:
        """Set threshold for a metric."""
        self._thresholds[metric] = threshold
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        snapshot = self._capture_snapshot()
        return snapshot.metrics


class CacheMonitorAdvanced:
    """Advanced cache monitor wrapper."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.monitor = RealTimeMonitor(cache)
        
    def start(self) -> None:
        """Start monitoring."""
        self.monitor.start_monitoring()
        
    def stop(self) -> None:
        """Stop monitoring."""
        self.monitor.stop_monitoring()
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard."""
        current_metrics = self.monitor.get_current_metrics()
        snapshots = self.monitor.get_snapshots(window_seconds=300)  # Last 5 minutes
        alerts = self.monitor.get_alerts(unacknowledged_only=True)
        
        return {
            'current_metrics': current_metrics,
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'hit_rate': s.hit_rate,
                    'cache_size': s.cache_size
                }
                for s in snapshots
            ],
            'active_alerts': len(alerts),
            'alerts': [
                {
                    'level': a.level.value,
                    'message': a.message,
                    'metric': a.metric,
                    'value': a.value
                }
                for a in alerts[:10]  # Last 10 alerts
            ]
        }


