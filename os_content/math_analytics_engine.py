from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

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
from refactored_math_system import MathService, MathOperation, MathResult, OperationType, CalculationMethod
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Analytics Engine for OS Content
Advanced analytics, monitoring, and insights for mathematical operations.
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


class TimeWindow(Enum):
    """Time windows for analytics."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class AnalyticsDataPoint:
    """Single data point for analytics."""
    timestamp: datetime
    metric: AnalyticsMetric
    value: float
    operation_type: Optional[str] = None
    method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_execution_time: float
    median_execution_time: float
    min_execution_time: float
    max_execution_time: float
    cache_hits: int
    cache_misses: int
    throughput_ops_per_second: float
    error_rate: float
    success_rate: float
    cache_hit_rate: float


@dataclass
class OperationAnalytics:
    """Analytics for specific operation types."""
    operation_type: str
    total_count: int
    success_count: int
    failure_count: int
    average_time: float
    median_time: float
    min_time: float
    max_time: float
    error_rate: float
    success_rate: float
    method_distribution: Dict[str, int]
    time_distribution: Dict[str, int]


class MathAnalyticsEngine:
    """Advanced analytics engine for mathematical operations."""
    
    def __init__(self, math_service: MathService, max_data_points: int = 10000):
        
    """__init__ function."""
self.math_service = math_service
        self.max_data_points = max_data_points
        self.data_points: deque = deque(maxlen=max_data_points)
        self.operation_history: deque = deque(maxlen=max_data_points)
        self.real_time_metrics: Dict[str, Any] = defaultdict(float)
        self.alert_thresholds: Dict[str, float] = {}
        self.analytics_callbacks: List[callable] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.operation_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.method_counters = defaultdict(int)
        
        logger.info("MathAnalyticsEngine initialized")
    
    def record_operation(self, operation: MathOperation, result: MathResult):
        """Record a mathematical operation for analytics."""
        timestamp = datetime.now()
        
        # Record operation history
        self.operation_history.append({
            "timestamp": timestamp,
            "operation": operation,
            "result": result
        })
        
        # Update counters
        op_type = operation.operation_type.value
        method = operation.method.value
        
        self.operation_counters[op_type] += 1
        self.method_counters[method] += 1
        
        if not result.success:
            self.error_counters[op_type] += 1
        
        # Create analytics data points
        self._create_analytics_data_points(timestamp, operation, result)
        
        # Update real-time metrics
        self._update_real_time_metrics(operation, result)
        
        # Check alerts
        self._check_alerts(operation, result)
        
        # Trigger analytics callbacks
        self._trigger_analytics_callbacks(operation, result)
    
    def _create_analytics_data_points(self, timestamp: datetime, 
                                    operation: MathOperation, result: MathResult):
        """Create analytics data points from operation."""
        # Execution time data point
        execution_time_dp = AnalyticsDataPoint(
            timestamp=timestamp,
            metric=AnalyticsMetric.EXECUTION_TIME,
            value=result.execution_time,
            operation_type=operation.operation_type.value,
            method=operation.method.value,
            metadata={"success": result.success}
        )
        self.data_points.append(execution_time_dp)
        
        # Success rate data point
        success_dp = AnalyticsDataPoint(
            timestamp=timestamp,
            metric=AnalyticsMetric.SUCCESS_RATE,
            value=1.0 if result.success else 0.0,
            operation_type=operation.operation_type.value,
            method=operation.method.value
        )
        self.data_points.append(success_dp)
        
        # Cache hit rate data point (if available)
        if hasattr(result, 'cache_hit') and result.cache_hit is not None:
            cache_dp = AnalyticsDataPoint(
                timestamp=timestamp,
                metric=AnalyticsMetric.CACHE_HIT_RATE,
                value=1.0 if result.cache_hit else 0.0,
                operation_type=operation.operation_type.value,
                method=operation.method.value
            )
            self.data_points.append(cache_dp)
    
    def _update_real_time_metrics(self, operation: MathOperation, result: MathResult):
        """Update real-time metrics."""
        op_type = operation.operation_type.value
        
        # Update execution time metrics
        self.real_time_metrics[f"{op_type}_total_time"] += result.execution_time
        self.real_time_metrics[f"{op_type}_count"] += 1
        self.real_time_metrics[f"{op_type}_avg_time"] = (
            self.real_time_metrics[f"{op_type}_total_time"] / 
            self.real_time_metrics[f"{op_type}_count"]
        )
        
        # Update success metrics
        if result.success:
            self.real_time_metrics[f"{op_type}_success_count"] += 1
        else:
            self.real_time_metrics[f"{op_type}_error_count"] += 1
        
        # Update overall metrics
        self.real_time_metrics["total_operations"] += 1
        self.real_time_metrics["total_execution_time"] += result.execution_time
        self.real_time_metrics["average_execution_time"] = (
            self.real_time_metrics["total_execution_time"] / 
            self.real_time_metrics["total_operations"]
        )
    
    def _check_alerts(self, operation: MathOperation, result: MathResult):
        """Check if any alert thresholds are exceeded."""
        op_type = operation.operation_type.value
        
        # Check execution time threshold
        time_threshold = self.alert_thresholds.get(f"{op_type}_execution_time")
        if time_threshold and result.execution_time > time_threshold:
            logger.warning(f"Execution time threshold exceeded for {op_type}: {result.execution_time}s > {time_threshold}s")
        
        # Check error rate threshold
        error_threshold = self.alert_thresholds.get(f"{op_type}_error_rate")
        if error_threshold:
            current_error_rate = self.error_counters[op_type] / max(1, self.operation_counters[op_type])
            if current_error_rate > error_threshold:
                logger.warning(f"Error rate threshold exceeded for {op_type}: {current_error_rate:.2%} > {error_threshold:.2%}")
    
    def _trigger_analytics_callbacks(self, operation: MathOperation, result: MathResult):
        """Trigger registered analytics callbacks."""
        for callback in self.analytics_callbacks:
            try:
                callback(operation, result)
            except Exception as e:
                logger.error(f"Error in analytics callback: {e}")
    
    def get_performance_metrics(self, time_window: Optional[TimeWindow] = None) -> PerformanceMetrics:
        """Get aggregated performance metrics."""
        # Filter data points by time window if specified
        if time_window:
            cutoff_time = self._get_cutoff_time(time_window)
            recent_data = [dp for dp in self.data_points if dp.timestamp >= cutoff_time]
        else:
            recent_data = list(self.data_points)
        
        # Calculate metrics
        execution_times = [dp.value for dp in recent_data if dp.metric == AnalyticsMetric.EXECUTION_TIME]
        success_rates = [dp.value for dp in recent_data if dp.metric == AnalyticsMetric.SUCCESS_RATE]
        
        total_ops = len(execution_times)
        successful_ops = sum(success_rates)
        failed_ops = total_ops - successful_ops
        
        if execution_times:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
        else:
            avg_time = median_time = min_time = max_time = 0.0
        
        # Calculate rates
        error_rate = failed_ops / max(1, total_ops)
        success_rate = successful_ops / max(1, total_ops)
        
        # Calculate throughput
        if time_window:
            window_seconds = self._get_window_seconds(time_window)
            throughput = total_ops / max(1, window_seconds)
        else:
            uptime = (datetime.now() - self.start_time).total_seconds()
            throughput = total_ops / max(1, uptime)
        
        # Get cache metrics from math service
        service_stats = self.math_service.get_stats()
        cache_hits = service_stats.get("cache_hits", 0)
        cache_misses = total_ops - cache_hits
        cache_hit_rate = cache_hits / max(1, total_ops)
        
        return PerformanceMetrics(
            total_operations=total_ops,
            successful_operations=int(successful_ops),
            failed_operations=int(failed_ops),
            average_execution_time=avg_time,
            median_execution_time=median_time,
            min_execution_time=min_time,
            max_execution_time=max_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            throughput_ops_per_second=throughput,
            error_rate=error_rate,
            success_rate=success_rate,
            cache_hit_rate=cache_hit_rate
        )
    
    def get_operation_analytics(self, operation_type: str, 
                               time_window: Optional[TimeWindow] = None) -> OperationAnalytics:
        """Get detailed analytics for a specific operation type."""
        # Filter operations by type and time window
        if time_window:
            cutoff_time = self._get_cutoff_time(time_window)
            operations = [
                op for op in self.operation_history
                if op["operation"].operation_type.value == operation_type and
                op["timestamp"] >= cutoff_time
            ]
        else:
            operations = [
                op for op in self.operation_history
                if op["operation"].operation_type.value == operation_type
            ]
        
        if not operations:
            return OperationAnalytics(
                operation_type=operation_type,
                total_count=0,
                success_count=0,
                failure_count=0,
                average_time=0.0,
                median_time=0.0,
                min_time=0.0,
                max_time=0.0,
                error_rate=0.0,
                success_rate=0.0,
                method_distribution={},
                time_distribution={}
            )
        
        # Calculate metrics
        execution_times = [op["result"].execution_time for op in operations]
        success_count = sum(1 for op in operations if op["result"].success)
        failure_count = len(operations) - success_count
        
        # Method distribution
        method_dist = defaultdict(int)
        for op in operations:
            method_dist[op["operation"].method.value] += 1
        
        # Time distribution (buckets)
        time_dist = defaultdict(int)
        for time_val in execution_times:
            if time_val < 0.001:
                time_dist["<1ms"] += 1
            elif time_val < 0.01:
                time_dist["1-10ms"] += 1
            elif time_val < 0.1:
                time_dist["10-100ms"] += 1
            elif time_val < 1.0:
                time_dist["100ms-1s"] += 1
            else:
                time_dist[">1s"] += 1
        
        return OperationAnalytics(
            operation_type=operation_type,
            total_count=len(operations),
            success_count=success_count,
            failure_count=failure_count,
            average_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            min_time=min(execution_times),
            max_time=max(execution_times),
            error_rate=failure_count / len(operations),
            success_rate=success_count / len(operations),
            method_distribution=dict(method_dist),
            time_distribution=dict(time_dist)
        )
    
    def get_trend_analysis(self, metric: AnalyticsMetric, 
                          time_window: TimeWindow = TimeWindow.HOUR) -> List[Tuple[datetime, float]]:
        """Get trend analysis for a specific metric."""
        cutoff_time = self._get_cutoff_time(time_window)
        recent_data = [dp for dp in self.data_points 
                      if dp.metric == metric and dp.timestamp >= cutoff_time]
        
        # Group by time intervals
        interval_seconds = self._get_window_seconds(time_window) / 10  # 10 intervals
        grouped_data = defaultdict(list)
        
        for dp in recent_data:
            interval_start = dp.timestamp.replace(
                microsecond=0,
                second=dp.timestamp.second - (dp.timestamp.second % int(interval_seconds))
            )
            grouped_data[interval_start].append(dp.value)
        
        # Calculate averages for each interval
        trend_data = []
        for interval_start, values in sorted(grouped_data.items()):
            avg_value = statistics.mean(values)
            trend_data.append((interval_start, avg_value))
        
        return trend_data
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = threshold
        logger.info(f"Alert threshold set for {metric}: {threshold}")
    
    def add_analytics_callback(self, callback: callable):
        """Add a callback function for analytics events."""
        self.analytics_callbacks.append(callback)
        logger.info("Analytics callback added")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return dict(self.real_time_metrics)
    
    def export_analytics_data(self, format: str = "json") -> str:
        """Export analytics data in specified format."""
        data = {
            "performance_metrics": self.get_performance_metrics().__dict__,
            "real_time_metrics": self.get_real_time_metrics(),
            "operation_counters": dict(self.operation_counters),
            "error_counters": dict(self.error_counters),
            "method_counters": dict(self.method_counters),
            "data_points_count": len(self.data_points),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
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
            raise ValueError(f"Unsupported time window: {time_window}")
    
    def _get_window_seconds(self, time_window: TimeWindow) -> float:
        """Get window duration in seconds."""
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
            raise ValueError(f"Unsupported time window: {time_window}")


class MathAnalyticsDashboard:
    """Dashboard for displaying math analytics."""
    
    def __init__(self, analytics_engine: MathAnalyticsEngine):
        
    """__init__ function."""
self.analytics_engine = analytics_engine
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        # Get overall performance metrics
        performance_metrics = self.analytics_engine.get_performance_metrics()
        
        # Get analytics for each operation type
        operation_analytics = {}
        for op_type in ["add", "multiply", "divide", "power"]:
            operation_analytics[op_type] = self.analytics_engine.get_operation_analytics(op_type)
        
        # Get trend data
        execution_time_trend = self.analytics_engine.get_trend_analysis(
            AnalyticsMetric.EXECUTION_TIME, TimeWindow.HOUR
        )
        success_rate_trend = self.analytics_engine.get_trend_analysis(
            AnalyticsMetric.SUCCESS_RATE, TimeWindow.HOUR
        )
        
        # Get real-time metrics
        real_time_metrics = self.analytics_engine.get_real_time_metrics()
        
        return {
            "performance_metrics": performance_metrics.__dict__,
            "operation_analytics": {
                op_type: analytics.__dict__ 
                for op_type, analytics in operation_analytics.items()
            },
            "trends": {
                "execution_time": execution_time_trend,
                "success_rate": success_rate_trend
            },
            "real_time_metrics": real_time_metrics,
            "dashboard_timestamp": datetime.now().isoformat()
        }
    
    def generate_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        metrics = self.analytics_engine.get_performance_metrics()
        
        report = f"""
Math Operations Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Performance:
- Total Operations: {metrics.total_operations:,}
- Success Rate: {metrics.success_rate:.2%}
- Error Rate: {metrics.error_rate:.2%}
- Average Execution Time: {metrics.average_execution_time:.4f}s
- Throughput: {metrics.throughput_ops_per_second:.2f} ops/sec
- Cache Hit Rate: {metrics.cache_hit_rate:.2%}

Execution Time Statistics:
- Minimum: {metrics.min_execution_time:.4f}s
- Median: {metrics.median_execution_time:.4f}s
- Maximum: {metrics.max_execution_time:.4f}s

Cache Performance:
- Cache Hits: {metrics.cache_hits:,}
- Cache Misses: {metrics.cache_misses:,}
"""
        
        # Add operation-specific analytics
        for op_type in ["add", "multiply", "divide", "power"]:
            op_analytics = self.analytics_engine.get_operation_analytics(op_type)
            if op_analytics.total_count > 0:
                report += f"""
{op_type.upper()} Operations:
- Total: {op_analytics.total_count:,}
- Success Rate: {op_analytics.success_rate:.2%}
- Average Time: {op_analytics.average_time:.4f}s
- Method Distribution: {op_analytics.method_distribution}
"""
        
        return report


# Example usage
async def main():
    """Example usage of the math analytics engine."""
    # Create math service
    math_service = create_math_service()
    
    # Create analytics engine
    analytics_engine = MathAnalyticsEngine(math_service)
    
    # Set up analytics callbacks
    def analytics_callback(operation, result) -> Any:
        print(f"Analytics: {operation.operation_type.value} took {result.execution_time:.4f}s")
    
    analytics_engine.add_analytics_callback(analytics_callback)
    
    # Set alert thresholds
    analytics_engine.set_alert_threshold("add_execution_time", 0.1)
    analytics_engine.set_alert_threshold("add_error_rate", 0.05)
    
    # Create dashboard
    dashboard = MathAnalyticsDashboard(analytics_engine)
    
    # Simulate some operations
    operations = [
        (OperationType.ADD, [1, 2, 3], CalculationMethod.BASIC),
        (OperationType.MULTIPLY, [2, 3, 4], CalculationMethod.NUMPY),
        (OperationType.DIVIDE, [10, 2], CalculationMethod.BASIC),
        (OperationType.POWER, [2, 3], CalculationMethod.MATH),
    ]
    
    for op_type, operands, method in operations:
        operation = MathOperation(
            operation_type=op_type,
            operands=operands,
            method=method
        )
        
        result = await math_service.processor.process_operation(operation)
        analytics_engine.record_operation(operation, result)
        
        await asyncio.sleep(0.1)  # Simulate time between operations
    
    # Generate dashboard data
    dashboard_data = dashboard.generate_dashboard_data()
    print("Dashboard Data:", json.dumps(dashboard_data, indent=2, default=str))
    
    # Generate performance report
    report = dashboard.generate_performance_report()
    print(report)


match __name__:
    case "__main__":
    asyncio.run(main()) 