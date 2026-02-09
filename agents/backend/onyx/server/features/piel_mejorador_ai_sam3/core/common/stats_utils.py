"""
Statistics Utilities for Piel Mejorador AI SAM3
===============================================

Unified statistics and metrics utilities.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@dataclass
class Statistic:
    """Single statistic value."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticsSummary:
    """Summary of statistics."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    stdev: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    total: Optional[float] = None


class StatsUtils:
    """Unified statistics utilities."""
    
    @staticmethod
    def calculate_summary(values: List[float]) -> StatisticsSummary:
        """
        Calculate statistics summary from values.
        
        Args:
            values: List of numeric values
            
        Returns:
            StatisticsSummary
        """
        if not values:
            return StatisticsSummary(
                count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0
            )
        
        sorted_values = sorted(values)
        count = len(values)
        
        return StatisticsSummary(
            count=count,
            min=min(values),
            max=max(values),
            mean=mean(values),
            median=median(values),
            stdev=stdev(values) if count > 1 else None,
            p50=StatsUtils._percentile(sorted_values, 50),
            p95=StatsUtils._percentile(sorted_values, 95),
            p99=StatsUtils._percentile(sorted_values, 99),
            total=sum(values)
        )
    
    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """
        Calculate percentile.
        
        Args:
            sorted_data: Sorted list of values
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0
        
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    @staticmethod
    def calculate_rate(
        count: int,
        duration_seconds: float
    ) -> float:
        """
        Calculate rate (count per second).
        
        Args:
            count: Number of events
            duration_seconds: Duration in seconds
            
        Returns:
            Rate (events per second)
        """
        if duration_seconds <= 0:
            return 0.0
        return count / duration_seconds
    
    @staticmethod
    def calculate_success_rate(
        successful: int,
        total: int
    ) -> float:
        """
        Calculate success rate.
        
        Args:
            successful: Number of successful operations
            total: Total number of operations
            
        Returns:
            Success rate (0.0-1.0)
        """
        if total == 0:
            return 0.0
        return successful / total
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """
        Calculate percentage.
        
        Args:
            value: Value
            total: Total
            
        Returns:
            Percentage (0-100)
        """
        if total == 0:
            return 0.0
        return (value / total) * 100
    
    @staticmethod
    def aggregate_stats(
        stats_list: List[Dict[str, Any]],
        key: str = "value"
    ) -> StatisticsSummary:
        """
        Aggregate statistics from multiple dictionaries.
        
        Args:
            stats_list: List of stat dictionaries
            key: Key to extract value from
            
        Returns:
            Aggregated StatisticsSummary
        """
        values = [s.get(key, 0) for s in stats_list if isinstance(s.get(key), (int, float))]
        return StatsUtils.calculate_summary(values)
    
    @staticmethod
    def rolling_average(
        values: deque,
        window_size: int = 10
    ) -> float:
        """
        Calculate rolling average.
        
        Args:
            values: Deque of values
            window_size: Window size
            
        Returns:
            Rolling average
        """
        if not values:
            return 0.0
        
        window = list(values)[-window_size:]
        return mean(window) if window else 0.0
    
    @staticmethod
    def detect_anomaly(
        value: float,
        mean: float,
        stdev: float,
        threshold: float = 3.0
    ) -> bool:
        """
        Detect if value is an anomaly (using z-score).
        
        Args:
            value: Value to check
            mean: Mean value
            stdev: Standard deviation
            threshold: Z-score threshold (default 3.0 = 3 sigma)
            
        Returns:
            True if anomaly
        """
        if stdev == 0:
            return False
        
        z_score = abs((value - mean) / stdev)
        return z_score > threshold
    
    @staticmethod
    def normalize(
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """
        Normalize value to 0-1 range.
        
        Args:
            value: Value to normalize
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Normalized value (0.0-1.0)
        """
        if max_val == min_val:
            return 0.0
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))


class MetricsTracker:
    """Advanced metrics tracker."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum history to keep
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
    
    def record(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        stat = Statistic(
            name=name,
            value=value,
            metadata=metadata or {}
        )
        self._metrics[name].append(stat)
    
    def increment(self, name: str, amount: int = 1):
        """
        Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        self._counters[name] += amount
    
    def decrement(self, name: str, amount: int = 1):
        """
        Decrement a counter.
        
        Args:
            name: Counter name
            amount: Amount to decrement
        """
        self._counters[name] = max(0, self._counters[name] - amount)
    
    def set_gauge(self, name: str, value: float):
        """
        Set a gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        self._gauges[name] = value
    
    def get_summary(self, name: str) -> Optional[StatisticsSummary]:
        """
        Get statistics summary for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            StatisticsSummary or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        
        values = [stat.value for stat in self._metrics[name]]
        return StatsUtils.calculate_summary(values)
    
    def get_counter(self, name: str) -> int:
        """
        Get counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value
        """
        return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """
        Get gauge value.
        
        Args:
            name: Gauge name
            
        Returns:
            Gauge value or None
        """
        return self._gauges.get(name)
    
    def get_all_metrics(self) -> Dict[str, StatisticsSummary]:
        """
        Get all metric summaries.
        
        Returns:
            Dictionary mapping metric names to summaries
        """
        return {
            name: self.get_summary(name)
            for name in self._metrics.keys()
            if self._metrics[name]
        }
    
    def reset(self, name: Optional[str] = None):
        """
        Reset metrics.
        
        Args:
            name: Optional metric name (reset all if None)
        """
        if name:
            if name in self._metrics:
                self._metrics[name].clear()
            if name in self._counters:
                self._counters[name] = 0
            if name in self._gauges:
                del self._gauges[name]
        else:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()


# Convenience functions
def calculate_summary(values: List[float]) -> StatisticsSummary:
    """Calculate statistics summary."""
    return StatsUtils.calculate_summary(values)


def calculate_rate(count: int, duration_seconds: float) -> float:
    """Calculate rate."""
    return StatsUtils.calculate_rate(count, duration_seconds)


def calculate_success_rate(successful: int, total: int) -> float:
    """Calculate success rate."""
    return StatsUtils.calculate_success_rate(successful, total)


def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage."""
    return StatsUtils.calculate_percentage(value, total)




