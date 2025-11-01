"""
Monitoring and observability for KV Cache.

Provides metrics collection and monitoring capabilities.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field
import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    timestamp: float = field(default_factory=time.time)
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    memory_usage_mb: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "eviction_rate": self.eviction_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "error_count": self.error_count,
        }


class CacheMonitor:
    """
    Monitors cache performance and collects metrics.
    
    Provides real-time monitoring, alerts, and metrics collection.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize cache monitor.
        
        Args:
            window_size: Size of sliding window for metrics
            alert_thresholds: Thresholds for alerts (e.g., {"hit_rate": 0.5})
        """
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            "hit_rate": 0.3,  # Alert if hit rate < 30%
            "memory_usage_mb": 8000,  # Alert if memory > 8GB
        }
        
        # Metrics history
        self._metrics_history: deque = deque(maxlen=window_size)
        self._operation_times: deque = deque(maxlen=window_size)
        
        # Current state
        self._last_metrics: Optional[CacheMetrics] = None
        self._alerts: List[str] = []
        
        logger.info(f"Initialized CacheMonitor with window_size={window_size}")
    
    def record_operation(self, operation_time: float) -> None:
        """Record operation time."""
        self._operation_times.append(operation_time)
    
    def update_metrics(
        self,
        stats: Dict[str, Any],
        error_count: int = 0
    ) -> CacheMetrics:
        """
        Update metrics from cache stats.
        
        Args:
            stats: Cache statistics dictionary
            error_count: Number of errors since last update
            
        Returns:
            Updated CacheMetrics
        """
        # Calculate throughput
        if len(self._operation_times) > 0:
            avg_time = sum(self._operation_times) / len(self._operation_times)
            throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        else:
            throughput = 0.0
        
        metrics = CacheMetrics(
            timestamp=time.time(),
            hit_rate=stats.get("hit_rate", 0.0),
            miss_rate=1.0 - stats.get("hit_rate", 0.0),
            eviction_rate=stats.get("evictions", 0) / max(stats.get("total_requests", 1), 1),
            memory_usage_mb=stats.get("storage_memory_mb", 0.0),
            avg_latency_ms=avg_time * 1000 if len(self._operation_times) > 0 else 0.0,
            throughput_ops_per_sec=throughput,
            error_count=error_count,
        )
        
        self._metrics_history.append(metrics)
        self._last_metrics = metrics
        
        # Check alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: CacheMetrics) -> None:
        """Check if metrics exceed alert thresholds."""
        alerts = []
        
        if metrics.hit_rate < self.alert_thresholds.get("hit_rate", 0.3):
            alerts.append(
                f"Low hit rate: {metrics.hit_rate:.2%} < {self.alert_thresholds.get('hit_rate', 0.3):.2%}"
            )
        
        if metrics.memory_usage_mb > self.alert_thresholds.get("memory_usage_mb", 8000):
            alerts.append(
                f"High memory usage: {metrics.memory_usage_mb:.2f} MB > {self.alert_thresholds.get('memory_usage_mb', 8000):.2f} MB"
            )
        
        if metrics.error_count > 0:
            alerts.append(f"Errors detected: {metrics.error_count}")
        
        if alerts:
            self._alerts.extend(alerts)
            for alert in alerts:
                logger.warning(f"Cache Alert: {alert}")
    
    def get_current_metrics(self) -> Optional[CacheMetrics]:
        """Get current metrics."""
        return self._last_metrics
    
    def get_metrics_history(self) -> List[CacheMetrics]:
        """Get metrics history."""
        return list(self._metrics_history)
    
    def get_alerts(self) -> List[str]:
        """Get recent alerts."""
        return list(self._alerts[-10:])  # Last 10 alerts
    
    def clear_alerts(self) -> None:
        """Clear alerts."""
        self._alerts.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self._metrics_history:
            return {"status": "no_data"}
        
        metrics_list = list(self._metrics_history)
        
        return {
            "total_samples": len(metrics_list),
            "avg_hit_rate": sum(m.hit_rate for m in metrics_list) / len(metrics_list),
            "avg_memory_mb": sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list),
            "avg_throughput": sum(m.throughput_ops_per_sec for m in metrics_list) / len(metrics_list),
            "current_metrics": self._last_metrics.to_dict() if self._last_metrics else None,
            "active_alerts": len(self._alerts),
            "recent_alerts": self.get_alerts(),
        }


class MetricsExporter:
    """Exports metrics to various backends."""
    
    @staticmethod
    def export_to_dict(monitor: CacheMonitor) -> Dict[str, Any]:
        """Export to dictionary."""
        return monitor.get_summary()
    
    @staticmethod
    def export_to_prometheus_format(monitor: CacheMonitor) -> str:
        """Export to Prometheus format."""
        metrics = monitor.get_current_metrics()
        if metrics is None:
            return ""
        
        lines = []
        lines.append(f"cache_hit_rate {metrics.hit_rate}")
        lines.append(f"cache_memory_usage_mb {metrics.memory_usage_mb}")
        lines.append(f"cache_throughput_ops_per_sec {metrics.throughput_ops_per_sec}")
        lines.append(f"cache_errors_total {metrics.error_count}")
        
        return "\n".join(lines)

