"""
Business Intelligence for Flux2 Clothing Changer
================================================

Advanced business intelligence and analytics.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BusinessMetric:
    """Business metric."""
    metric_name: str
    value: float
    timestamp: float
    category: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BusinessReport:
    """Business report."""
    report_id: str
    report_type: str
    period_start: float
    period_end: float
    metrics: Dict[str, float]
    insights: List[str]
    generated_at: float = time.time()


class BusinessIntelligence:
    """Advanced business intelligence system."""
    
    def __init__(self):
        """Initialize business intelligence."""
        self.metrics: Dict[str, List[BusinessMetric]] = defaultdict(list)
        self.reports: List[BusinessReport] = []
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record business metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            category: Metric category
            metadata: Optional metadata
        """
        metric = BusinessMetric(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            category=category,
            metadata=metadata or {},
        )
        
        self.metrics[metric_name].append(metric)
        logger.debug(f"Recorded metric: {metric_name} = {value}")
    
    def calculate_kpi(
        self,
        kpi_name: str,
        time_range: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate KPI.
        
        Args:
            kpi_name: KPI name
            time_range: Optional time range in seconds
            
        Returns:
            KPI value or None
        """
        if kpi_name not in self.metrics:
            return None
        
        metrics = self.metrics[kpi_name]
        
        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return None
        
        # Simple average (can be enhanced with different calculations)
        return sum(m.value for m in metrics) / len(metrics)
    
    def generate_report(
        self,
        report_type: str,
        period_start: float,
        period_end: float,
        metrics: List[str],
    ) -> BusinessReport:
        """
        Generate business report.
        
        Args:
            report_type: Report type
            period_start: Period start timestamp
            period_end: Period end timestamp
            metrics: List of metric names to include
            
        Returns:
            Generated report
        """
        report_id = f"report_{int(time.time() * 1000)}"
        report_metrics = {}
        insights = []
        
        for metric_name in metrics:
            metric_values = [
                m.value for m in self.metrics[metric_name]
                if period_start <= m.timestamp <= period_end
            ]
            
            if metric_values:
                report_metrics[metric_name] = {
                    "avg": sum(metric_values) / len(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "count": len(metric_values),
                }
        
        # Generate insights
        if "revenue" in report_metrics and "costs" in report_metrics:
            revenue = report_metrics["revenue"]["avg"]
            costs = report_metrics["costs"]["avg"]
            profit = revenue - costs
            profit_margin = (profit / revenue * 100) if revenue > 0 else 0
            insights.append(f"Profit margin: {profit_margin:.2f}%")
        
        report = BusinessReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            metrics=report_metrics,
            insights=insights,
        )
        
        self.reports.append(report)
        logger.info(f"Generated report: {report_id}")
        
        return report
    
    def get_trends(
        self,
        metric_name: str,
        time_range: float = 86400,  # 24 hours
    ) -> Dict[str, Any]:
        """
        Get metric trends.
        
        Args:
            metric_name: Metric name
            time_range: Time range in seconds
            
        Returns:
            Trend analysis
        """
        if metric_name not in self.metrics:
            return {}
        
        metrics = [
            m for m in self.metrics[metric_name]
            if m.timestamp >= time.time() - time_range
        ]
        
        if len(metrics) < 2:
            return {}
        
        values = [m.value for m in metrics]
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        trend = "increasing" if second_avg > first_avg else "decreasing" if second_avg < first_avg else "stable"
        change_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        return {
            "trend": trend,
            "change_percent": change_percent,
            "current_value": values[-1],
            "average": sum(values) / len(values),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get BI statistics."""
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "total_reports": len(self.reports),
        }


