"""
Advanced Performance Monitor
============================

Comprehensive performance monitoring with real-time metrics, alerts, and optimization suggestions.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    value: float
    label: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert."""
    level: str  # warning, critical
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: datetime

class PerformanceMonitor:
    """
    Advanced performance monitor with:
    - Real-time metrics collection
    - Alert system
    - Performance trends
    - Optimization suggestions
    - Resource usage tracking
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_callbacks: List[Callable] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        # Default thresholds
        self.set_default_thresholds()
    
    def set_default_thresholds(self):
        """Set default performance thresholds."""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 75.0, "critical": 90.0},
            "disk_usage": {"warning": 80.0, "critical": 95.0},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
            "queue_size": {"warning": 100, "critical": 500}
        }
    
    def set_threshold(self, metric: str, warning: float, critical: float):
        """Set custom threshold for a metric."""
        self.thresholds[metric] = {
            "warning": warning,
            "critical": critical
        }
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            value=value,
            label=name,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # Check thresholds
        self._check_thresholds(name, value)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Check if metric exceeds thresholds."""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get("critical", float('inf')):
            alert = PerformanceAlert(
                level="critical",
                message=f"{metric_name} is critically high: {value}",
                metric=metric_name,
                threshold=thresholds["critical"],
                current_value=value,
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)
        
        elif value >= thresholds.get("warning", float('inf')):
            alert = PerformanceAlert(
                level="warning",
                message=f"{metric_name} is high: {value}",
                metric=metric_name,
                threshold=thresholds["warning"],
                current_value=value,
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger alert callbacks."""
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(alert))
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Performance Alert [{alert.level}]: {alert.message}")
    
    def get_metric_stats(self, name: str, window: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        metrics = list(self.metrics[name])
        if window:
            metrics = metrics[-window:]
        
        values = [m.value for m in metrics]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
            "latest": values[-1] if values else None,
            "trend": self._calculate_trend(values)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend (increasing, decreasing, stable)."""
        if len(values) < 2:
            return "stable"
        
        recent = values[-10:] if len(values) >= 10 else values
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = statistics.mean(first_half) if first_half else 0
        second_avg = statistics.mean(second_half) if second_half else 0
        
        diff = second_avg - first_avg
        percent_change = (diff / first_avg * 100) if first_avg > 0 else 0
        
        if percent_change > 5:
            return "increasing"
        elif percent_change < -5:
            return "decreasing"
        else:
            return "stable"
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record metrics
            self.record_metric("cpu_usage", cpu_percent)
            self.record_metric("memory_usage", memory.percent)
            self.record_metric("disk_usage", disk.percent)
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "per_cpu": psutil.cpu_percent(interval=0.1, percpu=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": self._get_network_stats(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except:
            return {}
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on metrics."""
        suggestions = []
        
        # Check CPU usage
        cpu_stats = self.get_metric_stats("cpu_usage", window=100)
        if cpu_stats.get("mean", 0) > 80:
            suggestions.append({
                "type": "cpu",
                "severity": "high",
                "message": "High CPU usage detected. Consider scaling horizontally or optimizing algorithms.",
                "current_value": cpu_stats.get("mean", 0),
                "recommendation": "Add more workers or optimize CPU-intensive operations"
            })
        
        # Check memory usage
        memory_stats = self.get_metric_stats("memory_usage", window=100)
        if memory_stats.get("mean", 0) > 80:
            suggestions.append({
                "type": "memory",
                "severity": "high",
                "message": "High memory usage detected. Consider memory optimization or increasing resources.",
                "current_value": memory_stats.get("mean", 0),
                "recommendation": "Implement memory caching, cleanup unused objects, or increase memory allocation"
            })
        
        # Check response time
        response_stats = self.get_metric_stats("response_time", window=100)
        if response_stats.get("p95", 0) > 2.0:
            suggestions.append({
                "type": "performance",
                "severity": "medium",
                "message": "High response times detected. Consider optimizing queries or adding caching.",
                "current_value": response_stats.get("p95", 0),
                "recommendation": "Add caching layer, optimize database queries, or use async operations"
            })
        
        # Check error rate
        error_stats = self.get_metric_stats("error_rate", window=100)
        if error_stats.get("mean", 0) > 5:
            suggestions.append({
                "type": "reliability",
                "severity": "high",
                "message": "High error rate detected. Review error logs and fix issues.",
                "current_value": error_stats.get("mean", 0),
                "recommendation": "Review error logs, improve error handling, or add circuit breakers"
            })
        
        return suggestions
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring."""
        if self.is_monitoring:
            return
        
        self.monitor_interval = interval
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.get_system_metrics()
                    time.sleep(self.monitor_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(self.monitor_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        stats = {
            "system": self.get_system_metrics(),
            "metrics": {},
            "alerts": {
                "total": len(self.alerts),
                "recent": [
                    {
                        "level": a.level,
                        "message": a.message,
                        "metric": a.metric,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in self.alerts[-10:]
                ]
            },
            "suggestions": self.get_optimization_suggestions(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add stats for all metrics
        for metric_name in self.metrics.keys():
            stats["metrics"][metric_name] = self.get_metric_stats(metric_name)
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.alerts.clear()
        logger.info("Performance metrics reset")

# Global instance
performance_monitor = PerformanceMonitor()
































