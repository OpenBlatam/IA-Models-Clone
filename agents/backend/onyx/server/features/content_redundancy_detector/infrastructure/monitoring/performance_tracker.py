"""
Advanced Performance Tracking System - Real-time performance monitoring
Production-ready performance tracking and optimization
"""

import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSummary:
    """Summary of performance metrics"""
    name: str
    count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    std_dev: float
    throughput: float  # operations per second
    error_rate: float
    tags: Dict[str, str] = field(default_factory=dict)

class PerformanceTracker:
    """Advanced performance tracking and monitoring system"""
    
    def __init__(
        self,
        retention_period: int = 3600,  # 1 hour
        aggregation_interval: int = 60,  # 1 minute
        max_metrics: int = 10000
    ):
        self.retention_period = retention_period
        self.aggregation_interval = aggregation_interval
        self.max_metrics = max_metrics
        
        # Performance data
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.summaries: Dict[str, PerformanceSummary] = {}
        self.errors: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.aggregation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Callbacks
        self.callbacks: List[Callable[[PerformanceMetrics], None]] = []
        self.alert_callbacks: List[Callable[[str, float], None]] = []

    async def start(self):
        """Start performance tracking"""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.aggregation_task = asyncio.create_task(self._aggregation_worker())
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())

    async def stop(self):
        """Stop performance tracking"""
        self.running = False
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        await asyncio.gather(
            self.aggregation_task,
            self.cleanup_task,
            return_exceptions=True
        )

    def add_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for performance events"""
        self.callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[str, float], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def track_operation(
        self,
        name: str,
        duration: float,
        success: bool = True,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Track a performance operation"""
        with self.lock:
            # Create metrics
            metric = PerformanceMetrics(
                name=name,
                value=duration,
                timestamp=time.time(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Store metric
            self.metrics[name].append(metric)
            
            # Track errors
            if not success:
                self.errors[name] += 1
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    print(f"Performance callback error: {e}")
            
            # Check for alerts
            self._check_performance_alerts(name, duration)

    def track_async_operation(
        self,
        name: str,
        coro: Callable,
        *args,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ):
        """Track an async operation"""
        async def _tracked_operation():
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = await coro(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                self.track_operation(
                    name, duration, success, tags, 
                    {**(metadata or {}), "error": str(error) if error else None}
                )
        
        return _tracked_operation()

    def track_sync_operation(
        self,
        name: str,
        func: Callable,
        *args,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ):
        """Track a sync operation"""
        start_time = time.time()
        success = True
        error = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = e
            raise
        finally:
            duration = time.time() - start_time
            self.track_operation(
                name, duration, success, tags,
                {**(metadata or {}), "error": str(error) if error else None}
            )

    def _check_performance_alerts(self, name: str, duration: float):
        """Check for performance alerts"""
        # Get recent metrics for this operation
        recent_metrics = [
            m for m in self.metrics[name]
            if time.time() - m.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_metrics) < 10:  # Need at least 10 samples
            return
        
        # Calculate percentiles
        durations = [m.value for m in recent_metrics]
        p95 = statistics.quantiles(durations, n=20)[18]  # 95th percentile
        
        # Alert if current duration is significantly higher than p95
        if duration > p95 * 2:  # 2x p95
            for callback in self.alert_callbacks:
                try:
                    callback(name, duration)
                except Exception as e:
                    print(f"Alert callback error: {e}")

    def get_operation_summary(self, name: str) -> Optional[PerformanceSummary]:
        """Get performance summary for an operation"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            # Get recent metrics (last hour)
            cutoff_time = time.time() - self.retention_period
            recent_metrics = [
                m for m in self.metrics[name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            # Calculate statistics
            durations = [m.value for m in recent_metrics]
            total_time = sum(durations)
            count = len(durations)
            
            # Calculate percentiles
            sorted_durations = sorted(durations)
            p50 = sorted_durations[int(count * 0.5)] if count > 0 else 0
            p95 = sorted_durations[int(count * 0.95)] if count > 0 else 0
            p99 = sorted_durations[int(count * 0.99)] if count > 0 else 0
            
            # Calculate throughput (operations per second)
            time_span = max(m.timestamp for m in recent_metrics) - min(m.timestamp for m in recent_metrics)
            throughput = count / max(time_span, 1)
            
            # Calculate error rate
            error_count = self.errors.get(name, 0)
            error_rate = error_count / max(count, 1)
            
            return PerformanceSummary(
                name=name,
                count=count,
                total_time=total_time,
                avg_time=total_time / count,
                min_time=min(durations),
                max_time=max(durations),
                p50_time=p50,
                p95_time=p95,
                p99_time=p99,
                std_dev=statistics.stdev(durations) if count > 1 else 0,
                throughput=throughput,
                error_rate=error_rate,
                tags=recent_metrics[0].tags if recent_metrics else {}
            )

    def get_all_summaries(self) -> Dict[str, PerformanceSummary]:
        """Get performance summaries for all operations"""
        with self.lock:
            summaries = {}
            for name in self.metrics.keys():
                summary = self.get_operation_summary(name)
                if summary:
                    summaries[name] = summary
            return summaries

    def get_top_slow_operations(self, limit: int = 10) -> List[PerformanceSummary]:
        """Get top slowest operations by p95 time"""
        summaries = self.get_all_summaries()
        return sorted(
            summaries.values(),
            key=lambda s: s.p95_time,
            reverse=True
        )[:limit]

    def get_top_error_operations(self, limit: int = 10) -> List[PerformanceSummary]:
        """Get operations with highest error rates"""
        summaries = self.get_all_summaries()
        return sorted(
            summaries.values(),
            key=lambda s: s.error_rate,
            reverse=True
        )[:limit]

    def get_throughput_ranking(self, limit: int = 10) -> List[PerformanceSummary]:
        """Get operations ranked by throughput"""
        summaries = self.get_all_summaries()
        return sorted(
            summaries.values(),
            key=lambda s: s.throughput,
            reverse=True
        )[:limit]

    async def _aggregation_worker(self):
        """Background worker for metric aggregation"""
        while self.running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Performance aggregation error: {e}")

    async def _cleanup_worker(self):
        """Background worker for metric cleanup"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Performance cleanup error: {e}")

    async def _aggregate_metrics(self):
        """Aggregate metrics into summaries"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.aggregation_interval
            
            # Aggregate all metrics
            for name, metric_deque in self.metrics.items():
                if not metric_deque:
                    continue
                
                # Get recent metrics
                recent_metrics = [
                    m for m in metric_deque
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_metrics:
                    # Calculate summary
                    summary = self._calculate_summary(name, recent_metrics)
                    self.summaries[name] = summary

    async def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.retention_period
            
            # Clean up old metrics
            for name, metric_deque in self.metrics.items():
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()

    def _calculate_summary(self, name: str, metrics: List[PerformanceMetrics]) -> PerformanceSummary:
        """Calculate performance summary from metrics"""
        if not metrics:
            return PerformanceSummary(
                name=name, count=0, total_time=0, avg_time=0,
                min_time=0, max_time=0, p50_time=0, p95_time=0, p99_time=0,
                std_dev=0, throughput=0, error_rate=0
            )
        
        durations = [m.value for m in metrics]
        total_time = sum(durations)
        count = len(durations)
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50 = sorted_durations[int(count * 0.5)] if count > 0 else 0
        p95 = sorted_durations[int(count * 0.95)] if count > 0 else 0
        p99 = sorted_durations[int(count * 0.99)] if count > 0 else 0
        
        # Calculate throughput
        time_span = max(m.timestamp for m in metrics) - min(m.timestamp for m in metrics)
        throughput = count / max(time_span, 1)
        
        # Calculate error rate
        error_count = sum(1 for m in metrics if not m.metadata.get("success", True))
        error_rate = error_count / max(count, 1)
        
        return PerformanceSummary(
            name=name,
            count=count,
            total_time=total_time,
            avg_time=total_time / count,
            min_time=min(durations),
            max_time=max(durations),
            p50_time=p50,
            p95_time=p95,
            p99_time=p99,
            std_dev=statistics.stdev(durations) if count > 1 else 0,
            throughput=throughput,
            error_rate=error_rate,
            tags=metrics[0].tags if metrics else {}
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        summaries = self.get_all_summaries()
        
        if not summaries:
            return {"message": "No performance data available"}
        
        # Overall statistics
        total_operations = sum(s.count for s in summaries.values())
        total_time = sum(s.total_time for s in summaries.values())
        avg_response_time = total_time / max(total_operations, 1)
        
        # Top operations
        top_slow = self.get_top_slow_operations(5)
        top_errors = self.get_top_error_operations(5)
        top_throughput = self.get_throughput_ranking(5)
        
        return {
            "overview": {
                "total_operations": total_operations,
                "total_time": total_time,
                "avg_response_time": avg_response_time,
                "unique_operations": len(summaries)
            },
            "top_slow_operations": [
                {
                    "name": s.name,
                    "p95_time": s.p95_time,
                    "avg_time": s.avg_time,
                    "count": s.count
                } for s in top_slow
            ],
            "top_error_operations": [
                {
                    "name": s.name,
                    "error_rate": s.error_rate,
                    "error_count": int(s.error_rate * s.count),
                    "count": s.count
                } for s in top_errors
            ],
            "top_throughput_operations": [
                {
                    "name": s.name,
                    "throughput": s.throughput,
                    "count": s.count,
                    "avg_time": s.avg_time
                } for s in top_throughput
            ],
            "all_operations": {
                name: {
                    "count": summary.count,
                    "avg_time": summary.avg_time,
                    "p95_time": summary.p95_time,
                    "throughput": summary.throughput,
                    "error_rate": summary.error_rate
                } for name, summary in summaries.items()
            }
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics"""
        report = self.get_performance_report()
        
        if format == "json":
            return json.dumps(report, indent=2)
        else:
            return str(report)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of performance tracking"""
        with self.lock:
            total_metrics = sum(len(deque) for deque in self.metrics.values())
            unique_operations = len(self.metrics)
            
            return {
                "status": "healthy" if self.running else "stopped",
                "total_metrics": total_metrics,
                "unique_operations": unique_operations,
                "aggregation_running": self.aggregation_task is not None,
                "cleanup_running": self.cleanup_task is not None,
                "callbacks_registered": len(self.callbacks),
                "alert_callbacks_registered": len(self.alert_callbacks)
            }





