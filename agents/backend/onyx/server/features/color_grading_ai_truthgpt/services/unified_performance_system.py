"""
Unified Performance System for Color Grading AI
================================================

Consolidated performance system combining:
- PerformanceMonitor (metrics tracking, alerts)
- PerformanceProfiler (cProfile-based profiling)

Features:
- Real-time performance monitoring
- Detailed profiling with cProfile
- Anomaly detection
- Performance trends
- Resource usage analysis
- Alerting
- Context managers and decorators
"""

import logging
import time
import cProfile
import pstats
import io
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from contextlib import contextmanager
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ProfilerMode(Enum):
    """Profiler modes."""
    SIMPLE = "simple"  # Simple timing
    DETAILED = "detailed"  # Detailed profiling
    MEMORY = "memory"  # Memory profiling
    FULL = "full"  # Full profiling


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    operation: str
    duration: float
    timestamp: datetime
    success: bool
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Profile result."""
    function_name: str
    total_time: float
    call_count: int
    cumulative_time: float
    per_call_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedPerformanceSystem:
    """
    Unified performance system combining monitoring and profiling.
    
    Features:
    - Real-time performance tracking
    - Detailed profiling with cProfile
    - Anomaly detection
    - Performance trends
    - Resource usage analysis
    - Alerting
    - Context managers and decorators
    """
    
    def __init__(
        self,
        window_size: int = 100,
        profiler_mode: ProfilerMode = ProfilerMode.SIMPLE
    ):
        """
        Initialize unified performance system.
        
        Args:
            window_size: Size of sliding window for metrics
            profiler_mode: Profiler mode
        """
        self.window_size = window_size
        self.profiler_mode = profiler_mode
        
        # Monitoring
        self._metrics: Dict[str, deque] = {}  # operation -> metrics queue
        self._alerts: List[Dict[str, Any]] = []
        self._thresholds = {
            "slow_operation": 10.0,  # seconds
            "high_cpu": 80.0,  # percent
            "high_memory": 80.0,  # percent
        }
        
        # Profiling
        self._profiler: Optional[cProfile.Profile] = None
        self._profile_results: Dict[str, ProfileResult] = {}
        self._active_profiles: Dict[str, float] = {}
    
    # =========================================================================
    # Monitoring Methods
    # =========================================================================
    
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
        """Get statistics for an operation."""
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
        """Get performance trends."""
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
    
    # =========================================================================
    # Profiling Methods
    # =========================================================================
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling.
        
        Args:
            name: Profile name
            metadata: Optional metadata
        """
        start_time = time.time()
        
        if self.profiler_mode in [ProfilerMode.DETAILED, ProfilerMode.FULL]:
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            if profiler:
                profiler.disable()
                stats = self._extract_stats(profiler, name)
            else:
                stats = ProfileResult(
                    function_name=name,
                    total_time=elapsed,
                    call_count=1,
                    cumulative_time=elapsed,
                    per_call_time=elapsed,
                    metadata=metadata or {}
                )
            
            self._profile_results[name] = stats
            logger.debug(f"Profiled {name}: {elapsed:.4f}s")
    
    def _extract_stats(self, profiler: cProfile.Profile, name: str) -> ProfileResult:
        """Extract statistics from profiler."""
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Get top function stats
        total_time = stats.total_tt
        call_count = stats.total_calls
        
        return ProfileResult(
            function_name=name,
            total_time=total_time,
            call_count=call_count,
            cumulative_time=total_time,
            per_call_time=total_time / call_count if call_count > 0 else 0.0,
            metadata={"stats_output": stream.getvalue()}
        )
    
    def profile_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator for profiling functions.
        
        Args:
            func: Function to profile
            name: Optional profile name
            metadata: Optional metadata
        """
        profile_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with self.profile(profile_name, metadata):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with self.profile(profile_name, metadata):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    def get_profile_results(self, name: Optional[str] = None) -> Union[ProfileResult, Dict[str, ProfileResult]]:
        """Get profiling results."""
        if name:
            return self._profile_results.get(name)
        return self._profile_results.copy()
    
    def get_slowest_functions(self, limit: int = 10) -> List[ProfileResult]:
        """Get slowest functions."""
        results = sorted(
            self._profile_results.values(),
            key=lambda x: x.total_time,
            reverse=True
        )
        return results[:limit]
    
    def generate_report(self) -> str:
        """Generate performance report."""
        report = ["Unified Performance Report", "=" * 50, ""]
        
        # Monitoring summary
        if self._metrics:
            report.append("Performance Monitoring:")
            report.append("-" * 50)
            total_time = sum(
                sum(m.duration for m in metrics)
                for metrics in self._metrics.values()
            )
            report.append(f"Total Operations Tracked: {len(self._metrics)}")
            report.append(f"Total Time: {total_time:.4f}s")
            report.append("")
        
        # Profiling summary
        if self._profile_results:
            report.append("Performance Profiling:")
            report.append("-" * 50)
            total_time = sum(r.total_time for r in self._profile_results.values())
            report.append(f"Total Functions Profiled: {len(self._profile_results)}")
            report.append(f"Total Time: {total_time:.4f}s")
            report.append("")
            
            # Slowest functions
            report.append("Slowest Functions:")
            report.append("-" * 50)
            for result in self.get_slowest_functions(10):
                report.append(
                    f"{result.function_name}: {result.total_time:.4f}s "
                    f"({result.call_count} calls, {result.per_call_time:.4f}s/call)"
                )
            report.append("")
        
        # Alerts
        if self._alerts:
            report.append("Recent Alerts:")
            report.append("-" * 50)
            for alert in self.get_alerts(10):
                report.append(f"[{alert['type']}] {alert['message']}")
        
        return "\n".join(report)
    
    def clear(self):
        """Clear all data."""
        self._metrics.clear()
        self._profile_results.clear()
        self._alerts.clear()
        logger.info("Performance data cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "monitoring": {
                "operations_tracked": len(self._metrics),
                "total_metrics": sum(len(m) for m in self._metrics.values()),
                "alerts_count": len(self._alerts),
            },
            "profiling": {
                "profiles_count": len(self._profile_results),
                "total_time": sum(r.total_time for r in self._profile_results.values()) if self._profile_results else 0.0,
            }
        }
        
        if self._profile_results:
            stats["profiling"]["avg_time"] = (
                sum(r.total_time for r in self._profile_results.values()) / len(self._profile_results)
            )
            stats["profiling"]["slowest_function"] = (
                max(self._profile_results.values(), key=lambda x: x.total_time).function_name
            )
        
        return stats


