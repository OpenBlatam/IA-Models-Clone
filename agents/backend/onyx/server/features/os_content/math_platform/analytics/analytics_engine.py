from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import statistics
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from ..core.math_service import MathService, MathOperation, MathResult, OperationType, CalculationMethod
            import psutil
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Analytics Engine
Advanced analytics, monitoring, and insights for mathematical operations with production features.
"""



logger = logging.getLogger(__name__)


class AnalyticsMetric(Enum):
    """Types of analytics metrics."""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    RESPONSE_TIME = "response_time"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    QUEUE_SIZE = "queue_size"


class TimeWindow(Enum):
    """Time windows for analytics."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AnalyticsDataPoint:
    """Single data point for analytics with enhanced metadata."""
    timestamp: datetime
    metric: AnalyticsMetric
    value: float
    operation_type: Optional[str] = None
    method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics with enhanced statistics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_execution_time: float = 0.0
    median_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    std_dev_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    throughput_ops_per_second: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    cache_hit_rate: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    concurrent_operations: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class OperationAnalytics:
    """Analytics for specific operation types with enhanced metrics."""
    operation_type: str
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_time: float = 0.0
    median_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev_time: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    time_distribution: Dict[str, int] = field(default_factory=dict)
    p95_time: float = 0.0
    p99_time: float = 0.0
    throughput_per_second: float = 0.0


@dataclass
class Alert:
    """Alert definition with severity and conditions."""
    alert_id: str
    name: str
    metric: AnalyticsMetric
    threshold: float
    severity: AlertSeverity
    condition: str  # "above", "below", "equals"
    enabled: bool = True
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AlertEvent:
    """Alert event with context."""
    alert_id: str
    alert_name: str
    severity: AlertSeverity
    metric: AnalyticsMetric
    current_value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class MathAnalyticsEngine:
    """Advanced analytics engine for mathematical operations with production features."""
    
    def __init__(self, math_service: Optional[MathService] = None, 
                 max_data_points: int = 50000,
                 retention_days: int = 30):
        
    """__init__ function."""
self.math_service = math_service
        self.max_data_points = max_data_points
        self.retention_days = retention_days
        
        # Data storage
        self.data_points: deque = deque(maxlen=max_data_points)
        self.operation_history: deque = deque(maxlen=max_data_points)
        self.real_time_metrics: Dict[str, Any] = defaultdict(float)
        
        # Analytics components
        self.alerts: Dict[str, Alert] = {}
        self.analytics_callbacks: List[callable] = []
        self.alert_callbacks: List[callable] = []
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.cache_stats: Dict[str, int] = defaultdict(int)
        
        # Threading and concurrency
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # System monitoring
        self.system_metrics = {
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "concurrent_operations": 0
        }
        
        # Initialize default alerts
        self._setup_default_alerts()
        
        logger.info(f"MathAnalyticsEngine initialized with {max_data_points} max data points")
    
    async def initialize(self) -> Any:
        """Initialize the analytics engine."""
        logger.info("Initializing analytics engine...")
        
        # Start background tasks
        self._processing_task = asyncio.create_task(self._data_processing_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Analytics engine initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown the analytics engine gracefully."""
        logger.info("Shutting down analytics engine...")
        
        # Cancel background tasks
        if self._processing_task:
            self._processing_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Analytics engine shutdown completed")
    
    def _setup_default_alerts(self) -> Any:
        """Setup default alert thresholds."""
        default_alerts = [
            Alert("high_error_rate", "High Error Rate", AnalyticsMetric.ERROR_RATE, 0.1, AlertSeverity.WARNING, "above"),
            Alert("slow_execution", "Slow Execution", AnalyticsMetric.EXECUTION_TIME, 5.0, AlertSeverity.WARNING, "above"),
            Alert("low_success_rate", "Low Success Rate", AnalyticsMetric.SUCCESS_RATE, 0.9, AlertSeverity.ERROR, "below"),
            Alert("high_memory_usage", "High Memory Usage", AnalyticsMetric.MEMORY_USAGE, 80.0, AlertSeverity.WARNING, "above"),
            Alert("high_cpu_usage", "High CPU Usage", AnalyticsMetric.CPU_USAGE, 90.0, AlertSeverity.WARNING, "above")
        ]
        
        for alert in default_alerts:
            self.alerts[alert.alert_id] = alert
    
    async def _data_processing_loop(self) -> Any:
        """Background data processing loop."""
        while True:
            try:
                # Process data points in batches
                await self._process_data_batch()
                
                # Update system metrics
                await self._update_system_metrics()
                
                # Check alerts
                await self._check_alerts()
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Data processing error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop for old data."""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _process_data_batch(self) -> Any:
        """Process a batch of data points."""
        with self._lock:
            # Process recent data points
            recent_data = list(self.data_points)[-100:]  # Process last 100 points
            
            for data_point in recent_data:
                # Update operation-specific metrics
                if data_point.operation_type:
                    self.operation_counts[data_point.operation_type] += 1
                    
                    if data_point.metric == AnalyticsMetric.EXECUTION_TIME:
                        self.execution_times[data_point.operation_type].append(data_point.value)
                        
                        # Keep only recent execution times
                        if len(self.execution_times[data_point.operation_type]) > 1000:
                            self.execution_times[data_point.operation_type] = \
                                self.execution_times[data_point.operation_type][-1000:]
    
    async def _update_system_metrics(self) -> Any:
        """Update system-level metrics."""
        try:
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.system_metrics["memory_usage"] = memory_info.rss / 1024 / 1024  # MB
            self.system_metrics["cpu_usage"] = process.cpu_percent()
            
            # Update data points
            self._add_data_point(AnalyticsMetric.MEMORY_USAGE, self.system_metrics["memory_usage"])
            self._add_data_point(AnalyticsMetric.CPU_USAGE, self.system_metrics["cpu_usage"])
            
        except ImportError:
            logger.warning("psutil not available, system metrics disabled")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def _check_alerts(self) -> Any:
        """Check all configured alerts."""
        current_metrics = self.get_real_time_metrics()
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                (datetime.now() - alert.last_triggered).total_seconds() < alert.cooldown_seconds):
                continue
            
            current_value = current_metrics.get(alert.metric.value, 0.0)
            should_trigger = False
            
            if alert.condition == "above" and current_value > alert.threshold:
                should_trigger = True
            elif alert.condition == "below" and current_value < alert.threshold:
                should_trigger = True
            elif alert.condition == "equals" and abs(current_value - alert.threshold) < 0.001:
                should_trigger = True
            
            if should_trigger:
                await self._trigger_alert(alert, current_value)
    
    async def _trigger_alert(self, alert: Alert, current_value: float):
        """Trigger an alert."""
        alert.last_triggered = datetime.now()
        alert.trigger_count += 1
        
        alert_event = AlertEvent(
            alert_id=alert.alert_id,
            alert_name=alert.name,
            severity=alert.severity,
            metric=alert.metric,
            current_value=current_value,
            threshold=alert.threshold,
            timestamp=datetime.now(),
            context=self.get_real_time_metrics()
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_event)
                else:
                    callback(alert_event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.metric.value} = {current_value}")
    
    async def _cleanup_old_data(self) -> Any:
        """Clean up old data points."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        with self._lock:
            # Remove old data points
            old_count = 0
            while self.data_points and self.data_points[0].timestamp < cutoff_time:
                self.data_points.popleft()
                old_count += 1
            
            if old_count > 0:
                logger.info(f"Cleaned up {old_count} old data points")
    
    def record_operation(self, operation: MathOperation, result: MathResult):
        """Record an operation and its result for analytics."""
        timestamp = datetime.now()
        
        with self._lock:
            # Create analytics data points
            self._create_analytics_data_points(timestamp, operation, result)
            
            # Update real-time metrics
            self._update_real_time_metrics(operation, result)
            
            # Store operation history
            self.operation_history.append({
                "timestamp": timestamp,
                "operation": operation,
                "result": result
            })
            
            # Update cache statistics
            if result.cache_hit is not None:
                cache_key = f"{operation.operation_type.value}_{operation.method.value}"
                if result.cache_hit:
                    self.cache_stats[f"{cache_key}_hits"] += 1
                else:
                    self.cache_stats[f"{cache_key}_misses"] += 1
            
            # Update error counts
            if not result.success:
                self.error_counts[operation.operation_type.value] += 1
    
    def _create_analytics_data_points(self, timestamp: datetime, 
                                    operation: MathOperation, result: MathResult):
        """Create analytics data points from operation result."""
        # Execution time data point
        self._add_data_point(
            AnalyticsMetric.EXECUTION_TIME,
            result.execution_time,
            operation_type=operation.operation_type.value,
            method=operation.method.value
        )
        
        # Success rate data point
        self._add_data_point(
            AnalyticsMetric.SUCCESS_RATE,
            1.0 if result.success else 0.0,
            operation_type=operation.operation_type.value
        )
        
        # Cache hit rate data point
        if result.cache_hit is not None:
            self._add_data_point(
                AnalyticsMetric.CACHE_HIT_RATE,
                1.0 if result.cache_hit else 0.0,
                operation_type=operation.operation_type.value
            )
        
        # Throughput data point (operations per second)
        self._add_data_point(
            AnalyticsMetric.THROUGHPUT,
            1.0 / result.execution_time if result.execution_time > 0 else 0.0,
            operation_type=operation.operation_type.value
        )
    
    def _add_data_point(self, metric: AnalyticsMetric, value: float, 
                       operation_type: str = None, method: str = None):
        """Add a data point to the analytics store."""
        data_point = AnalyticsDataPoint(
            timestamp=datetime.now(),
            metric=metric,
            value=value,
            operation_type=operation_type,
            method=method
        )
        
        self.data_points.append(data_point)
    
    def _update_real_time_metrics(self, operation: MathOperation, result: MathResult):
        """Update real-time metrics."""
        # Update operation counts
        self.real_time_metrics[f"operations_{operation.operation_type.value}"] += 1
        self.real_time_metrics["total_operations"] += 1
        
        # Update execution times
        self.real_time_metrics[f"execution_time_{operation.operation_type.value}"] = result.execution_time
        
        # Update success/failure counts
        if result.success:
            self.real_time_metrics["successful_operations"] += 1
        else:
            self.real_time_metrics["failed_operations"] += 1
        
        # Update cache statistics
        if result.cache_hit is not None:
            if result.cache_hit:
                self.real_time_metrics["cache_hits"] += 1
            else:
                self.real_time_metrics["cache_misses"] += 1
        
        # Calculate rates
        total_ops = self.real_time_metrics["total_operations"]
        if total_ops > 0:
            self.real_time_metrics["success_rate"] = (
                self.real_time_metrics["successful_operations"] / total_ops
            )
            self.real_time_metrics["error_rate"] = (
                self.real_time_metrics["failed_operations"] / total_ops
            )
        
        cache_total = self.real_time_metrics["cache_hits"] + self.real_time_metrics["cache_misses"]
        if cache_total > 0:
            self.real_time_metrics["cache_hit_rate"] = (
                self.real_time_metrics["cache_hits"] / cache_total
            )
    
    def get_performance_metrics(self, time_window: Optional[TimeWindow] = None) -> PerformanceMetrics:
        """Get aggregated performance metrics for a time window."""
        with self._lock:
            # Filter data points by time window
            if time_window:
                cutoff_time = self._get_cutoff_time(time_window)
                filtered_data = [
                    dp for dp in self.data_points 
                    if dp.timestamp >= cutoff_time
                ]
            else:
                filtered_data = list(self.data_points)
            
            if not filtered_data:
                return PerformanceMetrics()
            
            # Calculate metrics
            execution_times = [
                dp.value for dp in filtered_data 
                if dp.metric == AnalyticsMetric.EXECUTION_TIME
            ]
            
            success_rates = [
                dp.value for dp in filtered_data 
                if dp.metric == AnalyticsMetric.SUCCESS_RATE
            ]
            
            cache_hit_rates = [
                dp.value for dp in filtered_data 
                if dp.metric == AnalyticsMetric.CACHE_HIT_RATE
            ]
            
            # Calculate statistics
            total_operations = len([dp for dp in filtered_data if dp.metric == AnalyticsMetric.EXECUTION_TIME])
            successful_operations = int(sum(success_rates))
            failed_operations = total_operations - successful_operations
            
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            median_execution_time = statistics.median(execution_times) if execution_times else 0.0
            min_execution_time = min(execution_times) if execution_times else 0.0
            max_execution_time = max(execution_times) if execution_times else 0.0
            std_dev_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            
            # Calculate percentiles
            p95_execution_time = np.percentile(execution_times, 95) if execution_times else 0.0
            p99_execution_time = np.percentile(execution_times, 99) if execution_times else 0.0
            
            # Calculate rates
            success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
            error_rate = failed_operations / total_operations if total_operations > 0 else 0.0
            cache_hit_rate = statistics.mean(cache_hit_rates) if cache_hit_rates else 0.0
            
            # Calculate throughput
            if time_window:
                window_seconds = self._get_window_seconds(time_window)
                throughput_ops_per_second = total_operations / window_seconds if window_seconds > 0 else 0.0
            else:
                throughput_ops_per_second = 0.0
            
            return PerformanceMetrics(
                total_operations=total_operations,
                successful_operations=successful_operations,
                failed_operations=failed_operations,
                average_execution_time=avg_execution_time,
                median_execution_time=median_execution_time,
                min_execution_time=min_execution_time,
                max_execution_time=max_execution_time,
                std_dev_execution_time=std_dev_execution_time,
                p95_execution_time=p95_execution_time,
                p99_execution_time=p99_execution_time,
                throughput_ops_per_second=throughput_ops_per_second,
                error_rate=error_rate,
                success_rate=success_rate,
                cache_hit_rate=cache_hit_rate,
                memory_usage_mb=self.system_metrics["memory_usage"],
                cpu_usage_percent=self.system_metrics["cpu_usage"]
            )
    
    def get_operation_analytics(self, operation_type: str, 
                               time_window: Optional[TimeWindow] = None) -> OperationAnalytics:
        """Get analytics for a specific operation type."""
        with self._lock:
            # Filter data points
            if time_window:
                cutoff_time = self._get_cutoff_time(time_window)
                filtered_data = [
                    dp for dp in self.data_points 
                    if dp.timestamp >= cutoff_time and dp.operation_type == operation_type
                ]
            else:
                filtered_data = [
                    dp for dp in self.data_points 
                    if dp.operation_type == operation_type
                ]
            
            if not filtered_data:
                return OperationAnalytics(operation_type=operation_type)
            
            # Calculate metrics
            execution_times = [
                dp.value for dp in filtered_data 
                if dp.metric == AnalyticsMetric.EXECUTION_TIME
            ]
            
            success_rates = [
                dp.value for dp in filtered_data 
                if dp.metric == AnalyticsMetric.SUCCESS_RATE
            ]
            
            # Method distribution
            method_distribution = defaultdict(int)
            for dp in filtered_data:
                if dp.method:
                    method_distribution[dp.method] += 1
            
            # Time distribution (bins)
            time_distribution = defaultdict(int)
            if execution_times:
                for time_val in execution_times:
                    if time_val < 0.001:
                        time_distribution["<1ms"] += 1
                    elif time_val < 0.01:
                        time_distribution["1-10ms"] += 1
                    elif time_val < 0.1:
                        time_distribution["10-100ms"] += 1
                    elif time_val < 1.0:
                        time_distribution["100ms-1s"] += 1
                    else:
                        time_distribution[">1s"] += 1
            
            # Calculate statistics
            total_count = len(execution_times)
            success_count = int(sum(success_rates))
            failure_count = total_count - success_count
            
            avg_time = statistics.mean(execution_times) if execution_times else 0.0
            median_time = statistics.median(execution_times) if execution_times else 0.0
            min_time = min(execution_times) if execution_times else 0.0
            max_time = max(execution_times) if execution_times else 0.0
            std_dev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            
            # Calculate percentiles
            p95_time = np.percentile(execution_times, 95) if execution_times else 0.0
            p99_time = np.percentile(execution_times, 99) if execution_times else 0.0
            
            # Calculate rates
            success_rate = success_count / total_count if total_count > 0 else 0.0
            error_rate = failure_count / total_count if total_count > 0 else 0.0
            
            # Calculate throughput
            if time_window:
                window_seconds = self._get_window_seconds(time_window)
                throughput_per_second = total_count / window_seconds if window_seconds > 0 else 0.0
            else:
                throughput_per_second = 0.0
            
            return OperationAnalytics(
                operation_type=operation_type,
                total_count=total_count,
                success_count=success_count,
                failure_count=failure_count,
                average_time=avg_time,
                median_time=median_time,
                min_time=min_time,
                max_time=max_time,
                std_dev_time=std_dev_time,
                p95_time=p95_time,
                p99_time=p99_time,
                error_rate=error_rate,
                success_rate=success_rate,
                method_distribution=dict(method_distribution),
                time_distribution=dict(time_distribution),
                throughput_per_second=throughput_per_second
            )
    
    def get_trend_analysis(self, metric: AnalyticsMetric, 
                          time_window: TimeWindow = TimeWindow.HOUR) -> List[Tuple[datetime, float]]:
        """Get trend analysis for a specific metric."""
        with self._lock:
            cutoff_time = self._get_cutoff_time(time_window)
            
            # Filter data points
            filtered_data = [
                dp for dp in self.data_points 
                if dp.timestamp >= cutoff_time and dp.metric == metric
            ]
            
            if not filtered_data:
                return []
            
            # Group by time intervals
            interval_seconds = self._get_window_seconds(time_window) / 10  # 10 intervals
            
            grouped_data = defaultdict(list)
            for dp in filtered_data:
                interval_start = dp.timestamp.replace(
                    second=dp.timestamp.second - (dp.timestamp.second % int(interval_seconds)),
                    microsecond=0
                )
                grouped_data[interval_start].append(dp.value)
            
            # Calculate averages for each interval
            trend_data = []
            for interval_start, values in sorted(grouped_data.items()):
                avg_value = statistics.mean(values)
                trend_data.append((interval_start, avg_value))
            
            return trend_data
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        with self._lock:
            metrics = dict(self.real_time_metrics)
            metrics.update(self.system_metrics)
            return metrics
    
    def add_alert(self, alert: Alert):
        """Add a new alert."""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Remove an alert."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert: {alert_id}")
    
    def add_alert_callback(self, callback: callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def add_analytics_callback(self, callback: callable):
        """Add an analytics callback function."""
        self.analytics_callbacks.append(callback)
    
    def export_analytics_data(self, format: str = "json") -> str:
        """Export analytics data in specified format."""
        with self._lock:
            if format.lower() == "json":
                data = {
                    "data_points": [
                        {
                            "timestamp": dp.timestamp.isoformat(),
                            "metric": dp.metric.value,
                            "value": dp.value,
                            "operation_type": dp.operation_type,
                            "method": dp.method,
                            "metadata": dp.metadata
                        }
                        for dp in self.data_points
                    ],
                    "performance_metrics": self.get_performance_metrics().__dict__,
                    "real_time_metrics": self.get_real_time_metrics(),
                    "alerts": [
                        {
                            "alert_id": alert.alert_id,
                            "name": alert.name,
                            "metric": alert.metric.value,
                            "threshold": alert.threshold,
                            "severity": alert.severity.value,
                            "enabled": alert.enabled,
                            "trigger_count": alert.trigger_count
                        }
                        for alert in self.alerts.values()
                    ]
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def _get_cutoff_time(self, time_window: TimeWindow) -> datetime:
        """Get cutoff time for a time window."""
        now = datetime.now()
        
        if time_window == TimeWindow.MINUTE:
            return now - timedelta(minutes=1)
        elif time_window == TimeWindow.HOUR:
            return now - timedelta(hours=1)
        elif time_window == TimeWindow.DAY:
            return now - timedelta(days=1)
        elif time_window == TimeWindow.WEEK:
            return now - timedelta(weeks=1)
        elif time_window == TimeWindow.MONTH:
            return now - timedelta(days=30)
        else:
            return now - timedelta(hours=1)  # Default to 1 hour
    
    def _get_window_seconds(self, time_window: TimeWindow) -> float:
        """Get time window duration in seconds."""
        if time_window == TimeWindow.MINUTE:
            return 60.0
        elif time_window == TimeWindow.HOUR:
            return 3600.0
        elif time_window == TimeWindow.DAY:
            return 86400.0
        elif time_window == TimeWindow.WEEK:
            return 604800.0
        elif time_window == TimeWindow.MONTH:
            return 2592000.0
        else:
            return 3600.0  # Default to 1 hour


class MathAnalyticsDashboard:
    """Dashboard for analytics visualization."""
    
    def __init__(self, analytics_engine: MathAnalyticsEngine):
        
    """__init__ function."""
self.analytics_engine = analytics_engine
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        return {
            "performance_metrics": self.analytics_engine.get_performance_metrics().__dict__,
            "real_time_metrics": self.analytics_engine.get_real_time_metrics(),
            "operation_analytics": {
                op_type: self.analytics_engine.get_operation_analytics(op_type).__dict__
                for op_type in ["add", "multiply", "divide", "power", "sqrt", "log"]
            },
            "trends": {
                "execution_time": self.analytics_engine.get_trend_analysis(AnalyticsMetric.EXECUTION_TIME),
                "success_rate": self.analytics_engine.get_trend_analysis(AnalyticsMetric.SUCCESS_RATE),
                "throughput": self.analytics_engine.get_trend_analysis(AnalyticsMetric.THROUGHPUT)
            },
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "enabled": alert.enabled,
                    "trigger_count": alert.trigger_count,
                    "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None
                }
                for alert in self.analytics_engine.alerts.values()
            ],
            "system_health": {
                "memory_usage_mb": self.analytics_engine.system_metrics["memory_usage"],
                "cpu_usage_percent": self.analytics_engine.system_metrics["cpu_usage"],
                "concurrent_operations": self.analytics_engine.system_metrics["concurrent_operations"]
            }
        }
    
    def generate_performance_report(self) -> str:
        """Generate a text-based performance report."""
        metrics = self.analytics_engine.get_performance_metrics()
        
        report = f"""
Math Platform Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Performance:
- Total Operations: {metrics.total_operations:,}
- Success Rate: {metrics.success_rate:.2%}
- Error Rate: {metrics.error_rate:.2%}
- Average Execution Time: {metrics.average_execution_time:.3f}s
- Median Execution Time: {metrics.median_execution_time:.3f}s
- 95th Percentile: {metrics.p95_execution_time:.3f}s
- 99th Percentile: {metrics.p99_execution_time:.3f}s
- Throughput: {metrics.throughput_ops_per_second:.2f} ops/sec
- Cache Hit Rate: {metrics.cache_hit_rate:.2%}

System Resources:
- Memory Usage: {metrics.memory_usage_mb:.1f} MB
- CPU Usage: {metrics.cpu_usage_percent:.1f}%

Operation Breakdown:
"""
        
        for op_type in ["add", "multiply", "divide", "power", "sqrt", "log"]:
            op_analytics = self.analytics_engine.get_operation_analytics(op_type)
            if op_analytics.total_count > 0:
                report += f"""
{op_type.upper()} Operations:
- Total: {op_analytics.total_count:,}
- Success Rate: {op_analytics.success_rate:.2%}
- Average Time: {op_analytics.average_time:.3f}s
- Throughput: {op_analytics.throughput_per_second:.2f} ops/sec
"""
        
        return report


async def main():
    """Main function for testing."""
    # Create analytics engine
    analytics_engine = MathAnalyticsEngine()
    await analytics_engine.initialize()
    
    try:
        # Create dashboard
        dashboard = MathAnalyticsDashboard(analytics_engine)
        
        # Generate dashboard data
        dashboard_data = dashboard.generate_dashboard_data()
        print("Dashboard data generated successfully")
        
        # Generate performance report
        report = dashboard.generate_performance_report()
        print(report)
        
    finally:
        await analytics_engine.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 