"""
Metrics Aggregator for Piel Mejorador AI SAM3
==============================================

Advanced metrics aggregation and analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetric:
    """Aggregated metric data."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    median: float
    p95: float
    p99: float
    std_dev: float
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsAggregator:
    """
    Advanced metrics aggregation.
    
    Features:
    - Time-based aggregation
    - Statistical analysis
    - Percentile calculations
    - Trend analysis
    - Anomaly detection
    """
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize metrics aggregator.
        
        Args:
            window_seconds: Aggregation window in seconds
        """
        self.window_seconds = window_seconds
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, List[datetime]] = defaultdict(list)
        self._aggregations: Dict[str, AggregatedMetric] = {}
    
    def record(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self._metrics[name].append(value)
        self._timestamps[name].append(timestamp)
        
        # Clean old values outside window
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        self._clean_old_values(name, cutoff)
    
    def _clean_old_values(self, name: str, cutoff: datetime):
        """Remove values outside the time window."""
        if name not in self._timestamps:
            return
        
        timestamps = self._timestamps[name]
        values = self._metrics[name]
        
        # Find first index within window
        valid_start = 0
        for i, ts in enumerate(timestamps):
            if ts >= cutoff:
                valid_start = i
                break
        
        if valid_start > 0:
            self._timestamps[name] = timestamps[valid_start:]
            self._metrics[name] = values[valid_start:]
    
    def aggregate(self, name: str) -> Optional[AggregatedMetric]:
        """
        Aggregate metrics for a name.
        
        Args:
            name: Metric name
            
        Returns:
            AggregatedMetric or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        
        values = self._metrics[name]
        
        if not values:
            return None
        
        sorted_values = sorted(values)
        count = len(values)
        
        aggregation = AggregatedMetric(
            name=name,
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / count,
            median=statistics.median(values),
            p95=self._percentile(sorted_values, 95),
            p99=self._percentile(sorted_values, 99),
            std_dev=statistics.stdev(values) if count > 1 else 0.0,
        )
        
        self._aggregations[name] = aggregation
        return aggregation
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not sorted_values:
            return 0.0
        
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_all_aggregations(self) -> Dict[str, AggregatedMetric]:
        """Get all aggregations."""
        # Aggregate all metrics
        for name in list(self._metrics.keys()):
            self.aggregate(name)
        
        return self._aggregations.copy()
    
    def detect_anomalies(
        self,
        name: str,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics.
        
        Args:
            name: Metric name
            threshold_std: Standard deviations threshold
            
        Returns:
            List of anomalies
        """
        if name not in self._metrics or not self._metrics[name]:
            return []
        
        aggregation = self.aggregate(name)
        if not aggregation:
            return []
        
        anomalies = []
        values = self._metrics[name]
        timestamps = self._timestamps[name]
        
        upper_bound = aggregation.avg + (threshold_std * aggregation.std_dev)
        lower_bound = aggregation.avg - (threshold_std * aggregation.std_dev)
        
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            if value > upper_bound or value < lower_bound:
                anomalies.append({
                    "index": i,
                    "value": value,
                    "timestamp": timestamp.isoformat(),
                    "deviation": abs(value - aggregation.avg),
                    "type": "high" if value > upper_bound else "low",
                })
        
        return anomalies
    
    def get_trend(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get trend analysis for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Trend analysis or None
        """
        if name not in self._metrics or len(self._metrics[name]) < 2:
            return None
        
        values = self._metrics[name]
        
        # Simple linear trend
        n = len(values)
        x = list(range(n))
        
        # Calculate slope (simple linear regression)
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "first_value": values[0],
            "last_value": values[-1],
            "change": values[-1] - values[0],
            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
        }
    
    def reset(self, name: Optional[str] = None):
        """
        Reset metrics.
        
        Args:
            name: Optional metric name (all if None)
        """
        if name:
            self._metrics.pop(name, None)
            self._timestamps.pop(name, None)
            self._aggregations.pop(name, None)
        else:
            self._metrics.clear()
            self._timestamps.clear()
            self._aggregations.clear()




