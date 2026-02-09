"""
Business Metrics V2
===================

Advanced business metrics tracking and analysis.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type."""
    REVENUE = "revenue"
    USAGE = "usage"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USER = "user"


@dataclass
class BusinessMetric:
    """Business metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


class BusinessMetricsV2:
    """Advanced business metrics system."""
    
    def __init__(self):
        """Initialize business metrics."""
        self.metrics: List[BusinessMetric] = []
        self.goals: Dict[str, float] = {}
        self.kpis: Dict[str, str] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a business metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Metric type
            tags: Optional tags
            metadata: Optional metadata
        """
        metric = BusinessMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Keep only last 10000 metrics
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
        
        logger.debug(f"Recorded business metric: {name} = {value}")
    
    def set_goal(self, metric_name: str, target_value: float) -> None:
        """
        Set goal for a metric.
        
        Args:
            metric_name: Metric name
            target_value: Target value
        """
        self.goals[metric_name] = target_value
        logger.info(f"Set goal for {metric_name}: {target_value}")
    
    def register_kpi(self, kpi_name: str, metric_name: str) -> None:
        """
        Register a KPI.
        
        Args:
            kpi_name: KPI name
            metric_name: Associated metric name
        """
        self.kpis[kpi_name] = metric_name
        logger.info(f"Registered KPI: {kpi_name} -> {metric_name}")
    
    def get_metric_summary(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get metric summary.
        
        Args:
            metric_name: Metric name
            time_range: Optional time range in seconds
            
        Returns:
            Metric summary
        """
        metrics = [m for m in self.metrics if m.name == metric_name]
        
        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {
                "name": metric_name,
                "count": 0,
            }
        
        values = [m.value for m in metrics]
        
        summary = {
            "name": metric_name,
            "count": len(metrics),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
        }
        
        # Add goal comparison if exists
        if metric_name in self.goals:
            summary["goal"] = self.goals[metric_name]
            summary["goal_achievement"] = (summary["avg"] / self.goals[metric_name] * 100) if self.goals[metric_name] > 0 else 0
        
        return summary
    
    def get_kpi_status(self) -> Dict[str, Any]:
        """
        Get KPI status.
        
        Returns:
            KPI status dictionary
        """
        kpi_status = {}
        
        for kpi_name, metric_name in self.kpis.items():
            summary = self.get_metric_summary(metric_name)
            kpi_status[kpi_name] = {
                "metric": metric_name,
                "current_value": summary.get("avg", 0),
                "goal": summary.get("goal"),
                "achievement": summary.get("goal_achievement", 0),
            }
        
        return kpi_status
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        # Group metrics by type
        metrics_by_type = {}
        for metric in self.metrics[-1000:]:  # Last 1000 metrics
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric.value)
        
        # Calculate statistics by type
        type_stats = {}
        for metric_type, values in metrics_by_type.items():
            type_stats[metric_type] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        
        return {
            "total_metrics": len(self.metrics),
            "metrics_by_type": type_stats,
            "goals": self.goals,
            "kpis": self.get_kpi_status(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get business metrics statistics."""
        return {
            "total_metrics": len(self.metrics),
            "goals": len(self.goals),
            "kpis": len(self.kpis),
        }

