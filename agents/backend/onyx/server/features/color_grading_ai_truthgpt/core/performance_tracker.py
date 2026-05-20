"""
Performance Tracker for Color Grading AI
========================================

Unified performance tracking with metrics aggregation and analysis.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric."""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot."""
    timestamp: datetime
    metrics: Dict[str, float]
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceTracker:
    """
    Unified performance tracker.
    
    Features:
    - Metric collection
    - Time-series tracking
    - Aggregations
    - Percentiles
    - Alerts
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize performance tracker.
        
        Args:
            max_history: Maximum history entries
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._snapshots: List[PerformanceSnapshot] = []
        self._alerts: List[Dict[str, Any]] = []
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            tags: Optional tags
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        self._metrics[name].append(metric)
        logger.debug(f"Recorded metric: {name}={value}{unit}")
    
    def record_timing(
        self,
        operation: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record timing metric."""
        self.record_metric(f"{operation}_duration", duration, unit="s", tags=tags)
        self.record_metric(f"{operation}_count", 1, tags=tags)
    
    def time_operation(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation: Operation name
            tags: Optional tags
        """
        return TimingContext(self, operation, tags)
    
    def get_metric_stats(
        self,
        name: str,
        window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get metric statistics.
        
        Args:
            name: Metric name
            window_seconds: Optional time window
            
        Returns:
            Statistics dictionary
        """
        metrics = list(self._metrics.get(name, []))
        
        # Filter by time window
        if window_seconds:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        if not metrics:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
            }
        
        values = [m.value for m in metrics]
        sorted_values = sorted(values)
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] + c * (data[f + 1] - data[f])
            return data[f]
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "p50": percentile(sorted_values, 0.5),
            "p95": percentile(sorted_values, 0.95),
            "p99": percentile(sorted_values, 0.99),
        }
    
    def create_snapshot(self, tags: Optional[Dict[str, str]] = None) -> PerformanceSnapshot:
        """
        Create performance snapshot.
        
        Args:
            tags: Optional tags
            
        Returns:
            Performance snapshot
        """
        metrics_dict = {}
        for name, metric_list in self._metrics.items():
            if metric_list:
                latest = metric_list[-1]
                metrics_dict[name] = latest.value
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            metrics=metrics_dict,
            tags=tags or {}
        )
        
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_history:
            self._snapshots = self._snapshots[-self.max_history:]
        
        return snapshot
    
    def get_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PerformanceSnapshot]:
        """Get performance snapshots."""
        snapshots = self._snapshots
        
        if start_date:
            snapshots = [s for s in snapshots if s.timestamp >= start_date]
        if end_date:
            snapshots = [s for s in snapshots if s.timestamp <= end_date]
        
        return snapshots[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for name in self._metrics.keys():
            stats = self.get_metric_stats(name)
            summary[name] = stats
        
        return {
            "metrics": summary,
            "snapshots_count": len(self._snapshots),
            "timestamp": datetime.now().isoformat(),
        }


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: PerformanceTracker, operation: str, tags: Optional[Dict[str, str]]):
        self.tracker = tracker
        self.operation = operation
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.tracker.record_timing(self.operation, duration, self.tags)




