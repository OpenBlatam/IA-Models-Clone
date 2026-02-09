"""
Performance Monitor for Piel Mejorador AI SAM3
==============================================

Advanced performance monitoring and optimization.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric."""
    name: str
    duration: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring system.
    
    Features:
    - Function timing
    - Performance metrics collection
    - Bottleneck detection
    - Performance reports
    """
    
    def __init__(self, max_metrics: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_metrics: Maximum metrics to keep
        """
        self.max_metrics = max_metrics
        self._metrics: Dict[str, deque] = {}
        self._active_timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start performance timer."""
        self._active_timers[name] = time.time()
    
    def stop_timer(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Stop performance timer.
        
        Args:
            name: Timer name
            metadata: Optional metadata
            
        Returns:
            Duration in seconds
        """
        if name not in self._active_timers:
            logger.warning(f"Timer {name} was not started")
            return 0.0
        
        start_time = self._active_timers.pop(name)
        duration = time.time() - start_time
        
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.max_metrics)
        
        metric = PerformanceMetric(
            name=name,
            duration=duration,
            metadata=metadata or {}
        )
        self._metrics[name].append(metric)
        
        return duration
    
    def record_metric(self, name: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.max_metrics)
        
        metric = PerformanceMetric(
            name=name,
            duration=duration,
            metadata=metadata or {}
        )
        self._metrics[name].append(metric)
    
    def get_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Statistics dictionary or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        
        metrics = list(self._metrics[name])
        durations = [m.duration for m in metrics]
        
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations),
            "p50": self._percentile(durations, 50),
            "p95": self._percentile(durations, 95),
            "p99": self._percentile(durations, 99),
        }
    
    def _percentile(self, data: list, percentile: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        return {
            name: self.get_statistics(name)
            for name in self._metrics.keys()
            if self._metrics[name]
        }
    
    def detect_bottlenecks(self, threshold_seconds: float = 1.0) -> list:
        """
        Detect performance bottlenecks.
        
        Args:
            threshold_seconds: Threshold in seconds
            
        Returns:
            List of bottleneck metrics
        """
        bottlenecks = []
        
        for name, stats in self.get_all_statistics().items():
            if stats and stats["avg"] > threshold_seconds:
                bottlenecks.append({
                    "name": name,
                    "avg_duration": stats["avg"],
                    "max_duration": stats["max"],
                    "count": stats["count"],
                })
        
        return sorted(bottlenecks, key=lambda x: x["avg_duration"], reverse=True)
    
    def monitor_function(self, func: Callable):
        """Decorator to monitor function performance."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            self.start_timer(func_name)
            try:
                result = await func(*args, **kwargs)
                duration = self.stop_timer(func_name)
                logger.debug(f"{func_name} took {duration:.3f}s")
                return result
            except Exception as e:
                self.stop_timer(func_name, {"error": str(e)})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            self.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                duration = self.stop_timer(func_name)
                logger.debug(f"{func_name} took {duration:.3f}s")
                return result
            except Exception as e:
                self.stop_timer(func_name, {"error": str(e)})
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _global_monitor




