"""
PDF Variantes - Advanced Analytics Dashboard
===========================================

Advanced analytics and reporting dashboard for PDF Variantes.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class TimeRange(str, Enum):
    """Time ranges."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


@dataclass
class DashboardMetric:
    """Dashboard metric."""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class DashboardWidget:
    """Dashboard widget."""
    widget_id: str
    title: str
    widget_type: str
    data: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 30  # seconds
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "widget_type": self.widget_type,
            "data": self.data,
            "position": self.position,
            "refresh_interval": self.refresh_interval,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DashboardReport:
    """Dashboard report."""
    report_id: str
    title: str
    description: str
    report_type: str
    data: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "description": self.description,
            "report_type": self.report_type,
            "data": self.data,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by
        }


class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard."""
    
    def __init__(self):
        self.metrics: Dict[str, List[DashboardMetric]] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        self.reports: Dict[str, DashboardReport] = {}
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized Advanced Analytics Dashboard")
    
    async def record_metric(
        self,
        metric_id: str,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.COUNTER,
        unit: str = "",
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric."""
        metric = DashboardMetric(
            metric_id=metric_id,
            name=name,
            description=description,
            metric_type=metric_type,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        if metric_id not in self.metrics:
            self.metrics[metric_id] = []
        
        self.metrics[metric_id].append(metric)
        
        # Keep only last 1000 metrics per metric_id
        if len(self.metrics[metric_id]) > 1000:
            self.metrics[metric_id] = self.metrics[metric_id][-1000:]
        
        logger.debug(f"Recorded metric: {metric_id} = {value}")
    
    async def get_metric_data(
        self,
        metric_id: str,
        time_range: TimeRange = TimeRange.LAST_DAY,
        custom_start: Optional[datetime] = None,
        custom_end: Optional[datetime] = None
    ) -> List[DashboardMetric]:
        """Get metric data for time range."""
        if metric_id not in self.metrics:
            return []
        
        metrics = self.metrics[metric_id]
        
        # Determine time range
        if time_range == TimeRange.CUSTOM and custom_start and custom_end:
            start_time = custom_start
            end_time = custom_end
        else:
            end_time = datetime.utcnow()
            start_time = self._get_start_time(end_time, time_range)
        
        # Filter metrics by time range
        filtered_metrics = [
            m for m in metrics
            if start_time <= m.timestamp <= end_time
        ]
        
        return filtered_metrics
    
    def _get_start_time(self, end_time: datetime, time_range: TimeRange) -> datetime:
        """Get start time for time range."""
        if time_range == TimeRange.LAST_HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == TimeRange.LAST_DAY:
            return end_time - timedelta(days=1)
        elif time_range == TimeRange.LAST_WEEK:
            return end_time - timedelta(weeks=1)
        elif time_range == TimeRange.LAST_MONTH:
            return end_time - timedelta(days=30)
        elif time_range == TimeRange.LAST_QUARTER:
            return end_time - timedelta(days=90)
        elif time_range == TimeRange.LAST_YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)
    
    async def create_widget(
        self,
        widget_id: str,
        title: str,
        widget_type: str,
        data_query: Dict[str, Any],
        position: Dict[str, int],
        refresh_interval: int = 30
    ) -> DashboardWidget:
        """Create dashboard widget."""
        # Execute data query
        data = await self._execute_data_query(data_query)
        
        widget = DashboardWidget(
            widget_id=widget_id,
            title=title,
            widget_type=widget_type,
            data=data,
            position=position,
            refresh_interval=refresh_interval
        )
        
        self.widgets[widget_id] = widget
        
        logger.info(f"Created widget: {widget_id}")
        return widget
    
    async def _execute_data_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data query for widget."""
        query_type = query.get("type", "metric")
        
        if query_type == "metric":
            return await self._execute_metric_query(query)
        elif query_type == "aggregation":
            return await self._execute_aggregation_query(query)
        elif query_type == "comparison":
            return await self._execute_comparison_query(query)
        else:
            return {"error": "Unknown query type"}
    
    async def _execute_metric_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute metric query."""
        metric_id = query.get("metric_id")
        time_range = TimeRange(query.get("time_range", "last_day"))
        
        if not metric_id:
            return {"error": "Missing metric_id"}
        
        metrics = await self.get_metric_data(metric_id, time_range)
        
        if not metrics:
            return {"data": [], "summary": {}}
        
        # Calculate summary statistics
        values = [m.value for m in metrics]
        summary = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0
        }
        
        return {
            "data": [m.to_dict() for m in metrics],
            "summary": summary
        }
    
    async def _execute_aggregation_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation query."""
        metric_ids = query.get("metric_ids", [])
        aggregation_type = query.get("aggregation", "sum")
        time_range = TimeRange(query.get("time_range", "last_day"))
        
        aggregated_data = {}
        
        for metric_id in metric_ids:
            metrics = await self.get_metric_data(metric_id, time_range)
            
            if not metrics:
                continue
            
            values = [m.value for m in metrics]
            
            if aggregation_type == "sum":
                aggregated_data[metric_id] = sum(values)
            elif aggregation_type == "avg":
                aggregated_data[metric_id] = sum(values) / len(values)
            elif aggregation_type == "max":
                aggregated_data[metric_id] = max(values)
            elif aggregation_type == "min":
                aggregated_data[metric_id] = min(values)
            elif aggregation_type == "count":
                aggregated_data[metric_id] = len(values)
        
        return {
            "aggregated_data": aggregated_data,
            "aggregation_type": aggregation_type,
            "time_range": time_range.value
        }
    
    async def _execute_comparison_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comparison query."""
        metric_id = query.get("metric_id")
        compare_periods = query.get("compare_periods", ["last_day", "last_week"])
        
        if not metric_id:
            return {"error": "Missing metric_id"}
        
        comparison_data = {}
        
        for period in compare_periods:
            time_range = TimeRange(period)
            metrics = await self.get_metric_data(metric_id, time_range)
            
            if metrics:
                values = [m.value for m in metrics]
                comparison_data[period] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values)
                }
        
        return {
            "comparison_data": comparison_data,
            "metric_id": metric_id
        }
    
    async def generate_report(
        self,
        report_id: str,
        title: str,
        description: str,
        report_type: str,
        data_queries: List[Dict[str, Any]],
        generated_by: str = "system"
    ) -> DashboardReport:
        """Generate dashboard report."""
        report_data = {}
        
        for query in data_queries:
            query_name = query.get("name", "unnamed")
            query_result = await self._execute_data_query(query)
            report_data[query_name] = query_result
        
        report = DashboardReport(
            report_id=report_id,
            title=title,
            description=description,
            report_type=report_type,
            data=report_data,
            generated_by=generated_by
        )
        
        self.reports[report_id] = report
        
        logger.info(f"Generated report: {report_id}")
        return report
    
    async def create_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        widget_ids: List[str],
        layout: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create dashboard."""
        dashboard = {
            "dashboard_id": dashboard_id,
            "title": title,
            "description": description,
            "widget_ids": widget_ids,
            "layout": layout,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.dashboards[dashboard_id] = dashboard
        
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data."""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        widgets_data = {}
        
        for widget_id in dashboard["widget_ids"]:
            if widget_id in self.widgets:
                widget = self.widgets[widget_id]
                widgets_data[widget_id] = widget.to_dict()
        
        return {
            "dashboard": dashboard,
            "widgets": widgets_data
        }
    
    async def get_usage_statistics(self, time_range: TimeRange = TimeRange.LAST_DAY) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = {
            "total_metrics": len(self.metrics),
            "total_widgets": len(self.widgets),
            "total_reports": len(self.reports),
            "total_dashboards": len(self.dashboards),
            "time_range": time_range.value
        }
        
        # Get metric counts by type
        metric_type_counts = {}
        for metric_id, metrics_list in self.metrics.items():
            if metrics_list:
                metric_type = metrics_list[0].metric_type
                metric_type_counts[metric_type.value] = metric_type_counts.get(metric_type.value, 0) + 1
        
        stats["metric_types"] = metric_type_counts
        
        # Get recent activity
        recent_metrics = []
        for metric_id, metrics_list in self.metrics.items():
            if metrics_list:
                recent_metrics.extend(metrics_list[-10:])  # Last 10 metrics per type
        
        recent_metrics.sort(key=lambda x: x.timestamp, reverse=True)
        stats["recent_activity"] = [m.to_dict() for m in recent_metrics[:50]]
        
        return stats
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        performance_metrics = {}
        
        # Calculate metrics processing performance
        total_metrics = sum(len(metrics) for metrics in self.metrics.values())
        performance_metrics["total_metrics_processed"] = total_metrics
        
        # Calculate widget refresh performance
        total_widgets = len(self.widgets)
        performance_metrics["total_widgets"] = total_widgets
        
        # Calculate report generation performance
        total_reports = len(self.reports)
        performance_metrics["total_reports_generated"] = total_reports
        
        # Calculate dashboard performance
        total_dashboards = len(self.dashboards)
        performance_metrics["total_dashboards"] = total_dashboards
        
        return performance_metrics
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Cleanup old data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Cleanup old metrics
        for metric_id in list(self.metrics.keys()):
            self.metrics[metric_id] = [
                m for m in self.metrics[metric_id]
                if m.timestamp >= cutoff_date
            ]
            
            if not self.metrics[metric_id]:
                del self.metrics[metric_id]
        
        # Cleanup old reports
        old_reports = [
            report_id for report_id, report in self.reports.items()
            if report.generated_at < cutoff_date
        ]
        
        for report_id in old_reports:
            del self.reports[report_id]
        
        logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        return {
            "metrics": {
                "total_metric_types": len(self.metrics),
                "total_metric_points": sum(len(metrics) for metrics in self.metrics.values())
            },
            "widgets": {
                "total_widgets": len(self.widgets),
                "widget_types": list(set(w.widget_type for w in self.widgets.values()))
            },
            "reports": {
                "total_reports": len(self.reports),
                "report_types": list(set(r.report_type for r in self.reports.values()))
            },
            "dashboards": {
                "total_dashboards": len(self.dashboards)
            }
        }


# Global instance
advanced_analytics_dashboard = AdvancedAnalyticsDashboard()
