"""
Advanced Metrics System for Flux2 Clothing Changer
===================================================

Advanced metrics collection and analysis.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Metric snapshot at a point in time."""
    timestamp: float
    metrics: Dict[str, float]
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class AdvancedMetrics:
    """Advanced metrics collection and analysis."""
    
    def __init__(
        self,
        history_size: int = 10000,
        aggregation_window: int = 60,  # seconds
    ):
        """
        Initialize advanced metrics system.
        
        Args:
            history_size: Maximum number of snapshots to keep
            aggregation_window: Window for metric aggregation in seconds
        """
        self.history_size = history_size
        self.aggregation_window = aggregation_window
        
        # Metric history
        self.snapshots: deque = deque(maxlen=history_size)
        
        # Aggregated metrics
        self.aggregated: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Metric definitions
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            metrics={name: value},
            tags=tags or {},
        )
        
        self.snapshots.append(snapshot)
        self._update_aggregated(name, value)
    
    def record_metrics(
        self,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            tags: Optional tags
        """
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            metrics=metrics,
            tags=tags or {},
        )
        
        self.snapshots.append(snapshot)
        
        for name, value in metrics.items():
            self._update_aggregated(name, value)
    
    def _update_aggregated(self, name: str, value: float) -> None:
        """Update aggregated metrics."""
        if name not in self.aggregated:
            self.aggregated[name] = {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "values": deque(maxlen=1000),
            }
        
        agg = self.aggregated[name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["values"].append(value)
    
    def get_metric_statistics(
        self,
        metric_name: str,
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            time_range: Time range to analyze
            
        Returns:
            Statistics dictionary
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        relevant_values = [
            s.metrics[metric_name]
            for s in self.snapshots
            if metric_name in s.metrics and s.timestamp >= cutoff_time
        ]
        
        if not relevant_values:
            return {}
        
        values_array = np.array(relevant_values)
        
        return {
            "count": len(relevant_values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p25": float(np.percentile(values_array, 25)),
            "p75": float(np.percentile(values_array, 75)),
            "p90": float(np.percentile(values_array, 90)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
        }
    
    def get_correlation(
        self,
        metric1: str,
        metric2: str,
        time_range: Optional[timedelta] = None,
    ) -> float:
        """
        Calculate correlation between two metrics.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            time_range: Time range to analyze
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        # Get paired values
        pairs = []
        for snapshot in self.snapshots:
            if snapshot.timestamp < cutoff_time:
                continue
            if metric1 in snapshot.metrics and metric2 in snapshot.metrics:
                pairs.append((
                    snapshot.metrics[metric1],
                    snapshot.metrics[metric2],
                ))
        
        if len(pairs) < 2:
            return 0.0
        
        values1, values2 = zip(*pairs)
        correlation = np.corrcoef(values1, values2)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def get_trend(
        self,
        metric_name: str,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate trend for a metric.
        
        Args:
            metric_name: Metric name
            window_size: Window size for trend calculation
            
        Returns:
            Trend information
        """
        values = [
            s.metrics[metric_name]
            for s in self.snapshots
            if metric_name in s.metrics
        ][-window_size:]
        
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": float(slope),
            "current_value": float(values[-1]),
            "previous_value": float(values[0]),
            "change": float(values[-1] - values[0]),
            "change_percent": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0,
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        # Get unique metric names
        metric_names = set()
        for snapshot in self.snapshots:
            metric_names.update(snapshot.metrics.keys())
        
        for metric_name in metric_names:
            if metric_name in self.aggregated:
                agg = self.aggregated[metric_name]
                summary[metric_name] = {
                    "count": agg["count"],
                    "mean": agg["sum"] / agg["count"] if agg["count"] > 0 else 0.0,
                    "min": agg["min"] if agg["min"] != float("inf") else 0.0,
                    "max": agg["max"] if agg["max"] != float("-inf") else 0.0,
                }
        
        return summary
    
    def export_metrics(self, file_path: str, format: str = "json") -> None:
        """
        Export metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format (json, csv)
        """
        if format == "json":
            import json
            data = {
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "metrics": s.metrics,
                        "tags": s.tags,
                    }
                    for s in self.snapshots
                ],
                "aggregated": {
                    name: {
                        "count": agg["count"],
                        "mean": agg["sum"] / agg["count"] if agg["count"] > 0 else 0.0,
                        "min": agg["min"],
                        "max": agg["max"],
                    }
                    for name, agg in self.aggregated.items()
                },
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            # Get all metric names
            metric_names = set()
            for snapshot in self.snapshots:
                metric_names.update(snapshot.metrics.keys())
            
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp"] + sorted(metric_names))
                
                for snapshot in self.snapshots:
                    row = [snapshot.timestamp]
                    for name in sorted(metric_names):
                        row.append(snapshot.metrics.get(name, ""))
                    writer.writerow(row)
        
        logger.info(f"Metrics exported to {file_path}")


