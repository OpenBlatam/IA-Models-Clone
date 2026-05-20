"""
Analytics Dashboard Service for Color Grading AI
================================================

Real-time analytics dashboard with metrics aggregation.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Dashboard metric."""
    name: str
    value: float
    unit: str = ""
    trend: Optional[str] = None  # up, down, stable
    change_percent: Optional[float] = None


@dataclass
class DashboardWidget:
    """Dashboard widget."""
    widget_id: str
    widget_type: str  # metric, chart, table, etc.
    title: str
    data: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height


class AnalyticsDashboard:
    """
    Analytics dashboard service.
    
    Features:
    - Real-time metrics
    - Custom widgets
    - Time-series data
    - Aggregations
    - Filters
    """
    
    def __init__(self):
        """Initialize analytics dashboard."""
        self._widgets: Dict[str, DashboardWidget] = {}
        self._metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = 1000
    
    def create_widget(
        self,
        widget_type: str,
        title: str,
        widget_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create dashboard widget.
        
        Args:
            widget_type: Widget type
            title: Widget title
            widget_id: Optional widget ID
            data: Optional widget data
            
        Returns:
            Widget ID
        """
        import uuid
        wid = widget_id or str(uuid.uuid4())
        
        widget = DashboardWidget(
            widget_id=wid,
            widget_type=widget_type,
            title=title,
            data=data or {}
        )
        
        self._widgets[wid] = widget
        logger.info(f"Created widget: {wid} ({title})")
        
        return wid
    
    def update_widget(self, widget_id: str, data: Dict[str, Any]):
        """Update widget data."""
        widget = self._widgets.get(widget_id)
        if not widget:
            raise ValueError(f"Widget not found: {widget_id}")
        
        widget.data.update(data)
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record metric for dashboard.
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Optional timestamp
        """
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = []
        
        entry = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "value": value,
        }
        
        self._metrics_history[metric_name].append(entry)
        
        # Keep only last N entries
        if len(self._metrics_history[metric_name]) > self._max_history:
            self._metrics_history[metric_name] = self._metrics_history[metric_name][-self._max_history:]
    
    def get_metric_timeseries(
        self,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Get metric time-series data.
        
        Args:
            metric_name: Metric name
            start_date: Start date
            end_date: End date
            interval: Time interval (1m, 5m, 1h, 1d)
            
        Returns:
            Time-series data
        """
        history = self._metrics_history.get(metric_name, [])
        
        if not history:
            return []
        
        # Filter by date range
        if start_date or end_date:
            filtered = []
            for entry in history:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if start_date and entry_time < start_date:
                    continue
                if end_date and entry_time > end_date:
                    continue
                filtered.append(entry)
            history = filtered
        
        # Aggregate by interval (simplified)
        # In production, use proper time-series aggregation
        return history
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        widgets_data = []
        
        for widget in self._widgets.values():
            widgets_data.append({
                "widget_id": widget.widget_id,
                "type": widget.widget_type,
                "title": widget.title,
                "data": widget.data,
                "position": widget.position,
            })
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics()
        
        return {
            "widgets": widgets_data,
            "summary_metrics": summary_metrics,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _calculate_summary_metrics(self) -> List[DashboardMetric]:
        """Calculate summary metrics."""
        metrics = []
        
        # Total requests
        total_requests = sum(
            len(history) for history in self._metrics_history.values()
        )
        metrics.append(DashboardMetric(
            name="Total Requests",
            value=total_requests,
            unit="requests"
        ))
        
        # Active metrics
        active_metrics = len(self._metrics_history)
        metrics.append(DashboardMetric(
            name="Active Metrics",
            value=active_metrics,
            unit="metrics"
        ))
        
        # Widgets count
        metrics.append(DashboardMetric(
            name="Dashboard Widgets",
            value=len(self._widgets),
            unit="widgets"
        ))
        
        return metrics
    
    def get_aggregated_metrics(
        self,
        metric_names: List[str],
        aggregation: str = "sum",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get aggregated metrics.
        
        Args:
            metric_names: List of metric names
            aggregation: Aggregation type (sum, avg, min, max, count)
            start_date: Start date
            end_date: End date
            
        Returns:
            Aggregated values
        """
        results = {}
        
        for metric_name in metric_names:
            history = self._metrics_history.get(metric_name, [])
            
            if not history:
                results[metric_name] = 0.0
                continue
            
            # Filter by date
            if start_date or end_date:
                filtered = []
                for entry in history:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                    filtered.append(entry)
                history = filtered
            
            if not history:
                results[metric_name] = 0.0
                continue
            
            values = [entry["value"] for entry in history]
            
            if aggregation == "sum":
                results[metric_name] = sum(values)
            elif aggregation == "avg":
                results[metric_name] = sum(values) / len(values)
            elif aggregation == "min":
                results[metric_name] = min(values)
            elif aggregation == "max":
                results[metric_name] = max(values)
            elif aggregation == "count":
                results[metric_name] = len(values)
            else:
                results[metric_name] = sum(values)
        
        return results
    
    def export_dashboard(self, format: str = "json") -> str:
        """
        Export dashboard configuration.
        
        Args:
            format: Export format (json, yaml)
            
        Returns:
            Exported data
        """
        dashboard_data = self.get_dashboard_data()
        
        if format == "json":
            import json
            return json.dumps(dashboard_data, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump(dashboard_data)
        else:
            raise ValueError(f"Unsupported format: {format}")




