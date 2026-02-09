"""
Metrics Dashboard for Piel Mejorador AI SAM3
============================================

Advanced metrics dashboard system.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series metric data."""
    name: str
    points: List[MetricPoint]
    unit: str = ""
    description: str = ""


class MetricsDashboard:
    """
    Advanced metrics dashboard.
    
    Features:
    - Time series metrics
    - Aggregations
    - Dashboards
    - Alerts based on metrics
    """
    
    def __init__(self, max_points_per_series: int = 1000):
        """
        Initialize metrics dashboard.
        
        Args:
            max_points_per_series: Maximum points to keep per series
        """
        self.max_points = max_points_per_series
        self._series: Dict[str, deque] = {}
        self._aggregations: Dict[str, Dict[str, float]] = {}
        
        self._stats = {
            "total_metrics": 0,
            "series_count": 0,
        }
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels or {}
        )
        
        # Store in series
        if name not in self._series:
            self._series[name] = deque(maxlen=self.max_points)
            self._stats["series_count"] += 1
        
        self._series[name].append(point)
        self._stats["total_metrics"] += 1
        
        # Update aggregations
        self._update_aggregations(name, value)
    
    def _update_aggregations(self, name: str, value: float):
        """Update metric aggregations."""
        if name not in self._aggregations:
            self._aggregations[name] = {
                "min": value,
                "max": value,
                "sum": 0,
                "count": 0,
                "avg": value,
            }
        
        agg = self._aggregations[name]
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["sum"] += value
        agg["count"] += 1
        agg["avg"] = agg["sum"] / agg["count"]
    
    def get_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[MetricSeries]:
        """
        Get metric series.
        
        Args:
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            MetricSeries or None
        """
        if name not in self._series:
            return None
        
        points = list(self._series[name])
        
        # Filter by time range
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return MetricSeries(
            name=name,
            points=points,
            unit="",
            description=""
        )
    
    def get_aggregation(self, name: str) -> Optional[Dict[str, float]]:
        """Get metric aggregations."""
        return self._aggregations.get(name)
    
    def get_dashboard_data(
        self,
        metric_names: Optional[List[str]] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get dashboard data.
        
        Args:
            metric_names: Optional list of metric names (all if None)
            time_range_hours: Time range in hours
            
        Returns:
            Dashboard data dictionary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        metrics = metric_names or list(self._series.keys())
        
        dashboard = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "metrics": {},
        }
        
        for name in metrics:
            series = self.get_series(name, start_time, end_time)
            aggregation = self.get_aggregation(name)
            
            if series:
                dashboard["metrics"][name] = {
                    "points": [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "value": p.value,
                            "labels": p.labels,
                        }
                        for p in series.points
                    ],
                    "aggregation": aggregation,
                    "point_count": len(series.points),
                }
        
        return dashboard
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            **self._stats,
            "series": list(self._series.keys()),
        }




