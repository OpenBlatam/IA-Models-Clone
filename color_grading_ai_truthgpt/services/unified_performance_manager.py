"""
Unified Performance Manager for Color Grading AI
================================================

Consolidates PerformanceMonitor, PerformanceOptimizer, and PerformanceProfiler
into a single unified performance management system.
"""

import logging
import time
import asyncio
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from contextlib import contextmanager
from enum import Enum
import statistics
import psutil

from .performance_monitor import PerformanceMonitor, PerformanceMetric
from .performance_optimizer import PerformanceOptimizer, SystemResources
from .performance_profiler import PerformanceProfiler, ProfileResult, ProfilerMode

logger = logging.getLogger(__name__)


class PerformanceMode(Enum):
    """Performance management modes."""
    MONITORING = "monitoring"  # Track metrics only
    OPTIMIZATION = "optimization"  # Monitor + optimize resources
    PROFILING = "profiling"  # Monitor + detailed profiling
    FULL = "full"  # All features enabled


@dataclass
class UnifiedPerformanceReport:
    """Unified performance report."""
    timestamp: datetime
    metrics: Dict[str, Any]
    resource_stats: Dict[str, Any]
    profile_results: Optional[Dict[str, ProfileResult]] = None
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class UnifiedPerformanceManager:
    """
    Unified performance management system.
    
    Consolidates:
    - PerformanceMonitor: Metrics tracking and alerts
    - PerformanceOptimizer: Resource optimization and throttling
    - PerformanceProfiler: Detailed code profiling
    
    Features:
    - Real-time performance monitoring
    - Resource optimization
    - Detailed profiling
    - Anomaly detection
    - Performance recommendations
    - Unified reporting
    """
    
    def __init__(
        self,
        mode: PerformanceMode = PerformanceMode.OPTIMIZATION,
        window_size: int = 100,
        enable_profiling: bool = False
    ):
        """
        Initialize unified performance manager.
        
        Args:
            mode: Performance management mode
            window_size: Size of sliding window for metrics
            enable_profiling: Enable detailed profiling
        """
        self.mode = mode
        self.enable_profiling = enable_profiling
        
        # Initialize components
        self.monitor = PerformanceMonitor(window_size=window_size)
        self.optimizer = PerformanceOptimizer()
        self.profiler = PerformanceProfiler(
            mode=ProfilerMode.DETAILED if enable_profiling else ProfilerMode.SIMPLE
        ) if enable_profiling else None
        
        logger.info(f"Initialized UnifiedPerformanceManager (mode={mode.value})")
    
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
        # Record in monitor
        self.monitor.record_metric(operation, duration, success, resource_usage)
        
        # Auto-optimize if in optimization mode
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            if self.optimizer.should_throttle():
                logger.warning(f"High resource usage detected for {operation}")
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling.
        
        Args:
            name: Profile name
            metadata: Optional metadata
        """
        start_time = time.time()
        start_resources = self.optimizer.get_system_resources() if self.profiler else None
        
        if self.profiler:
            with self.profiler.profile(name, metadata):
                yield
        else:
            yield
        
        # Record metric
        duration = time.time() - start_time
        resource_usage = None
        
        if start_resources:
            end_resources = self.optimizer.get_system_resources()
            resource_usage = {
                "cpu_percent": end_resources.cpu_percent,
                "memory_percent": end_resources.memory_percent,
            }
        
        self.record_metric(name, duration, success=True, resource_usage=resource_usage)
    
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
        
        if self.profiler:
            decorated = self.profiler.profile_function(func, profile_name, metadata)
        else:
            # Simple wrapper without detailed profiling
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
        
        return decorated
    
    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled.
        
        Returns:
            True if should throttle
        """
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            return self.optimizer.should_throttle()
        return False
    
    def get_optimal_workers(self, base_workers: int = 3) -> int:
        """
        Get optimal number of workers.
        
        Args:
            base_workers: Base number of workers
            
        Returns:
            Optimal number of workers
        """
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            return self.optimizer.get_optimal_workers(base_workers)
        return base_workers
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Statistics dictionary
        """
        stats = self.monitor.get_operation_stats(operation)
        
        # Add profiling data if available
        if self.profiler:
            profile_result = self.profiler.get_results(operation)
            if profile_result:
                stats["profiling"] = {
                    "total_time": profile_result.total_time,
                    "call_count": profile_result.call_count,
                    "per_call_time": profile_result.per_call_time,
                }
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations."""
        return self.monitor.get_all_stats()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            return self.optimizer.get_resource_stats()
        return {}
    
    def get_trends(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends.
        
        Args:
            operation: Operation name
            hours: Number of hours to analyze
            
        Returns:
            Trends dictionary
        """
        return self.monitor.get_trends(operation, hours)
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.monitor.get_alerts(limit)
    
    def get_slowest_functions(self, limit: int = 10) -> List[ProfileResult]:
        """
        Get slowest functions.
        
        Args:
            limit: Number of functions to return
            
        Returns:
            List of slowest functions
        """
        if self.profiler:
            return self.profiler.get_slowest_functions(limit)
        return []
    
    def generate_report(self) -> UnifiedPerformanceReport:
        """
        Generate unified performance report.
        
        Returns:
            Unified performance report
        """
        # Collect metrics
        metrics = self.get_all_stats()
        
        # Collect resource stats
        resource_stats = self.get_resource_stats()
        
        # Collect profile results
        profile_results = None
        if self.profiler:
            profile_results = self.profiler.get_results()
        
        # Collect alerts
        alerts = self.get_alerts()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, resource_stats, alerts)
        
        return UnifiedPerformanceReport(
            timestamp=datetime.now(),
            metrics=metrics,
            resource_stats=resource_stats,
            profile_results=profile_results,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        resource_stats: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check for slow operations
        for operation, stats in metrics.items():
            if stats.get("avg_duration", 0) > 5.0:
                recommendations.append(
                    f"Operation '{operation}' is slow (avg: {stats['avg_duration']:.2f}s). "
                    "Consider optimization or caching."
                )
        
        # Check resource usage
        if resource_stats:
            current = resource_stats.get("current", {})
            if current.get("cpu_percent", 0) > 80:
                recommendations.append(
                    f"High CPU usage ({current['cpu_percent']:.1f}%). "
                    "Consider reducing concurrent operations."
                )
            if current.get("memory_percent", 0) > 80:
                recommendations.append(
                    f"High memory usage ({current['memory_percent']:.1f}%). "
                    "Consider cache cleanup or memory optimization."
                )
        
        # Check alerts
        if len(alerts) > 10:
            recommendations.append(
                f"High number of alerts ({len(alerts)}). "
                "Review system configuration and thresholds."
            )
        
        return recommendations
    
    async def wait_if_throttled(self, delay: float = 1.0):
        """Wait if system is throttled."""
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            await self.optimizer.wait_if_throttled(delay)
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set performance threshold."""
        self.monitor.set_threshold(threshold_name, value)
    
    def clear(self):
        """Clear all performance data."""
        self.monitor._metrics.clear()
        self.monitor._alerts.clear()
        if self.profiler:
            self.profiler.clear()
        logger.info("Performance data cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        stats = {
            "mode": self.mode.value,
            "profiling_enabled": self.profiler is not None,
            "monitor_stats": {
                "operations_tracked": len(self.monitor._metrics),
                "alerts_count": len(self.monitor._alerts),
            },
        }
        
        if self.profiler:
            stats["profiler_stats"] = self.profiler.get_statistics()
        
        if self.mode in [PerformanceMode.OPTIMIZATION, PerformanceMode.FULL]:
            stats["resource_stats"] = self.get_resource_stats()
        
        return stats


