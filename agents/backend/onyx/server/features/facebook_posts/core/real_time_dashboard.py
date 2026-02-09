"""
Real-time Analytics Dashboard for Facebook Posts
Following functional programming principles and real-time data visualization
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


# Pure functions for real-time analytics

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    GAUGE = "gauge"
    HEATMAP = "heatmap"


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass(frozen=True)
class DataPoint:
    """Immutable data point - pure data structure"""
    timestamp: datetime
    value: float
    label: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "label": self.label,
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class ChartConfig:
    """Immutable chart configuration - pure data structure"""
    chart_type: ChartType
    title: str
    x_axis_label: str
    y_axis_label: str
    data_points: List[DataPoint]
    color: str
    show_legend: bool
    animation_duration: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "color": self.color,
            "show_legend": self.show_legend,
            "animation_duration": self.animation_duration
        }


@dataclass(frozen=True)
class DashboardWidget:
    """Immutable dashboard widget - pure data structure"""
    id: str
    title: str
    chart_config: ChartConfig
    refresh_interval: int
    position: Dict[str, int]
    size: Dict[str, int]
    is_visible: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "id": self.id,
            "title": self.title,
            "chart_config": self.chart_config.to_dict(),
            "refresh_interval": self.refresh_interval,
            "position": self.position,
            "size": self.size,
            "is_visible": self.is_visible
        }


def create_data_point(
    value: float,
    label: str,
    metadata: Optional[Dict[str, Any]] = None
) -> DataPoint:
    """Create data point - pure function"""
    return DataPoint(
        timestamp=datetime.utcnow(),
        value=value,
        label=label,
        metadata=metadata or {}
    )


def create_chart_config(
    chart_type: ChartType,
    title: str,
    data_points: List[DataPoint],
    x_axis_label: str = "Time",
    y_axis_label: str = "Value",
    color: str = "#3498db",
    show_legend: bool = True,
    animation_duration: int = 1000
) -> ChartConfig:
    """Create chart configuration - pure function"""
    return ChartConfig(
        chart_type=chart_type,
        title=title,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        data_points=data_points,
        color=color,
        show_legend=show_legend,
        animation_duration=animation_duration
    )


def create_dashboard_widget(
    widget_id: str,
    title: str,
    chart_config: ChartConfig,
    position: Dict[str, int],
    size: Dict[str, int],
    refresh_interval: int = 30,
    is_visible: bool = True
) -> DashboardWidget:
    """Create dashboard widget - pure function"""
    return DashboardWidget(
        id=widget_id,
        title=title,
        chart_config=chart_config,
        refresh_interval=refresh_interval,
        position=position,
        size=size,
        is_visible=is_visible
    )


def calculate_data_statistics(data_points: List[DataPoint]) -> Dict[str, Any]:
    """Calculate data statistics - pure function"""
    if not data_points:
        return {"count": 0, "average": 0, "min": 0, "max": 0, "sum": 0}
    
    values = [dp.value for dp in data_points]
    
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
        "last_value": values[-1] if values else 0,
        "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing"
    }


def filter_data_by_time_range(
    data_points: List[DataPoint],
    start_time: datetime,
    end_time: datetime
) -> List[DataPoint]:
    """Filter data by time range - pure function"""
    return [
        dp for dp in data_points
        if start_time <= dp.timestamp <= end_time
    ]


def aggregate_data_by_interval(
    data_points: List[DataPoint],
    interval_minutes: int
) -> List[DataPoint]:
    """Aggregate data by time interval - pure function"""
    if not data_points:
        return []
    
    # Group data points by time intervals
    intervals = defaultdict(list)
    
    for dp in data_points:
        # Round timestamp to interval
        rounded_time = dp.timestamp.replace(
            minute=(dp.timestamp.minute // interval_minutes) * interval_minutes,
            second=0,
            microsecond=0
        )
        intervals[rounded_time].append(dp)
    
    # Aggregate each interval
    aggregated_points = []
    for interval_time, points in intervals.items():
        if points:
            avg_value = sum(p.value for p in points) / len(points)
            aggregated_point = DataPoint(
                timestamp=interval_time,
                value=avg_value,
                label=f"Aggregated {interval_minutes}min",
                metadata={"original_count": len(points)}
            )
            aggregated_points.append(aggregated_point)
    
    return sorted(aggregated_points, key=lambda dp: dp.timestamp)


# Real-time Dashboard System Class

class RealTimeDashboard:
    """Real-time Analytics Dashboard following functional principles"""
    
    def __init__(self, max_data_points: int = 10000):
        self.max_data_points = max_data_points
        
        # Data storage
        self.metrics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.widgets: Dict[str, DashboardWidget] = {}
        self.subscribers: Set[Callable] = set()
        
        # Real-time updates
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        self.update_interval = 1  # seconds
        
        # Statistics
        self.stats = {
            "total_data_points": 0,
            "active_widgets": 0,
            "subscribers_count": 0,
            "last_update": None
        }
    
    async def start(self) -> None:
        """Start real-time dashboard"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Real-time dashboard started")
    
    async def stop(self) -> None:
        """Stop real-time dashboard"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time dashboard stopped")
    
    async def _update_loop(self) -> None:
        """Main update loop"""
        while self.is_running:
            try:
                await self._update_all_widgets()
                await self._notify_subscribers()
                
                self.stats["last_update"] = datetime.utcnow()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _update_all_widgets(self) -> None:
        """Update all widgets with fresh data"""
        for widget_id, widget in self.widgets.items():
            if not widget.is_visible:
                continue
            
            try:
                await self._update_widget_data(widget_id)
            except Exception as e:
                logger.error(f"Error updating widget {widget_id}", error=str(e))
    
    async def _update_widget_data(self, widget_id: str) -> None:
        """Update widget data"""
        widget = self.widgets.get(widget_id)
        if not widget:
            return
        
        # Get fresh data for the widget
        metric_name = widget.chart_config.title.lower().replace(" ", "_")
        data_points = list(self.metrics_data.get(metric_name, []))
        
        if not data_points:
            return
        
        # Filter recent data (last hour by default)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_data = [
            dp for dp in data_points
            if dp.timestamp > cutoff_time
        ]
        
        # Update widget with new data
        updated_chart_config = ChartConfig(
            chart_type=widget.chart_config.chart_type,
            title=widget.chart_config.title,
            x_axis_label=widget.chart_config.x_axis_label,
            y_axis_label=widget.chart_config.y_axis_label,
            data_points=recent_data,
            color=widget.chart_config.color,
            show_legend=widget.chart_config.show_legend,
            animation_duration=widget.chart_config.animation_duration
        )
        
        updated_widget = DashboardWidget(
            id=widget.id,
            title=widget.title,
            chart_config=updated_chart_config,
            refresh_interval=widget.refresh_interval,
            position=widget.position,
            size=widget.size,
            is_visible=widget.is_visible
        )
        
        self.widgets[widget_id] = updated_widget
    
    async def _notify_subscribers(self) -> None:
        """Notify all subscribers of updates"""
        if not self.subscribers:
            return
        
        dashboard_data = self.get_dashboard_data()
        
        for subscriber in list(self.subscribers):
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(dashboard_data)
                else:
                    subscriber(dashboard_data)
            except Exception as e:
                logger.error("Error notifying subscriber", error=str(e))
    
    def add_data_point(self, metric_name: str, value: float, label: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add data point to metric"""
        data_point = create_data_point(value, label, metadata)
        self.metrics_data[metric_name].append(data_point)
        self.stats["total_data_points"] += 1
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add widget to dashboard"""
        self.widgets[widget.id] = widget
        self.stats["active_widgets"] = len(self.widgets)
        logger.info(f"Widget {widget.id} added to dashboard")
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            self.stats["active_widgets"] = len(self.widgets)
            logger.info(f"Widget {widget_id} removed from dashboard")
            return True
        return False
    
    def update_widget(self, widget_id: str, updates: Dict[str, Any]) -> bool:
        """Update widget configuration"""
        if widget_id not in self.widgets:
            return False
        
        widget = self.widgets[widget_id]
        
        # Create updated widget
        updated_widget = DashboardWidget(
            id=widget.id,
            title=updates.get("title", widget.title),
            chart_config=widget.chart_config,
            refresh_interval=updates.get("refresh_interval", widget.refresh_interval),
            position=updates.get("position", widget.position),
            size=updates.get("size", widget.size),
            is_visible=updates.get("is_visible", widget.is_visible)
        )
        
        self.widgets[widget_id] = updated_widget
        return True
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to dashboard updates"""
        self.subscribers.add(callback)
        self.stats["subscribers_count"] = len(self.subscribers)
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from dashboard updates"""
        self.subscribers.discard(callback)
        self.stats["subscribers_count"] = len(self.subscribers)
    
    def get_metric_data(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[DataPoint]:
        """Get metric data with optional filtering"""
        data_points = list(self.metrics_data.get(metric_name, []))
        
        # Filter by time range
        if start_time and end_time:
            data_points = filter_data_by_time_range(data_points, start_time, end_time)
        
        # Limit results
        if limit:
            data_points = data_points[-limit:]
        
        return data_points
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, Any]:
        """Get metric statistics"""
        data_points = list(self.metrics_data.get(metric_name, []))
        return calculate_data_statistics(data_points)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            "widgets": [widget.to_dict() for widget in self.widgets.values()],
            "statistics": self.stats.copy(),
            "metrics": {
                name: {
                    "data_points": len(data),
                    "last_value": data[-1].value if data else 0,
                    "statistics": calculate_data_statistics(list(data))
                }
                for name, data in self.metrics_data.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_widget_data(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get specific widget data"""
        widget = self.widgets.get(widget_id)
        if not widget:
            return None
        
        return widget.to_dict()
    
    def create_performance_widget(self, widget_id: str) -> DashboardWidget:
        """Create performance monitoring widget"""
        # Get performance data
        performance_data = list(self.metrics_data.get("performance", []))
        
        chart_config = create_chart_config(
            chart_type=ChartType.LINE,
            title="System Performance",
            data_points=performance_data,
            x_axis_label="Time",
            y_axis_label="Performance Score",
            color="#e74c3c"
        )
        
        return create_dashboard_widget(
            widget_id=widget_id,
            title="Performance Monitor",
            chart_config=chart_config,
            position={"x": 0, "y": 0},
            size={"width": 400, "height": 300}
        )
    
    def create_engagement_widget(self, widget_id: str) -> DashboardWidget:
        """Create engagement monitoring widget"""
        # Get engagement data
        engagement_data = list(self.metrics_data.get("engagement", []))
        
        chart_config = create_chart_config(
            chart_type=ChartType.AREA,
            title="Post Engagement",
            data_points=engagement_data,
            x_axis_label="Time",
            y_axis_label="Engagement Rate",
            color="#2ecc71"
        )
        
        return create_dashboard_widget(
            widget_id=widget_id,
            title="Engagement Monitor",
            chart_config=chart_config,
            position={"x": 400, "y": 0},
            size={"width": 400, "height": 300}
        )
    
    def create_ai_metrics_widget(self, widget_id: str) -> DashboardWidget:
        """Create AI metrics widget"""
        # Get AI metrics data
        ai_data = list(self.metrics_data.get("ai_metrics", []))
        
        chart_config = create_chart_config(
            chart_type=ChartType.BAR,
            title="AI Model Performance",
            data_points=ai_data,
            x_axis_label="Model",
            y_axis_label="Accuracy",
            color="#9b59b6"
        )
        
        return create_dashboard_widget(
            widget_id=widget_id,
            title="AI Metrics",
            chart_config=chart_config,
            position={"x": 0, "y": 300},
            size={"width": 400, "height": 300}
        )
    
    def create_system_health_widget(self, widget_id: str) -> DashboardWidget:
        """Create system health widget"""
        # Get system health data
        health_data = list(self.metrics_data.get("system_health", []))
        
        chart_config = create_chart_config(
            chart_type=ChartType.GAUGE,
            title="System Health",
            data_points=health_data,
            x_axis_label="Time",
            y_axis_label="Health Score",
            color="#f39c12"
        )
        
        return create_dashboard_widget(
            widget_id=widget_id,
            title="System Health",
            chart_config=chart_config,
            position={"x": 400, "y": 300},
            size={"width": 400, "height": 300}
        )
    
    def setup_default_widgets(self) -> None:
        """Setup default dashboard widgets"""
        # Performance widget
        perf_widget = self.create_performance_widget("performance_widget")
        self.add_widget(perf_widget)
        
        # Engagement widget
        engagement_widget = self.create_engagement_widget("engagement_widget")
        self.add_widget(engagement_widget)
        
        # AI metrics widget
        ai_widget = self.create_ai_metrics_widget("ai_metrics_widget")
        self.add_widget(ai_widget)
        
        # System health widget
        health_widget = self.create_system_health_widget("system_health_widget")
        self.add_widget(health_widget)
        
        logger.info("Default dashboard widgets setup completed")


# Factory functions

def create_real_time_dashboard(max_data_points: int = 10000) -> RealTimeDashboard:
    """Create real-time dashboard - pure function"""
    return RealTimeDashboard(max_data_points)


async def get_real_time_dashboard() -> RealTimeDashboard:
    """Get real-time dashboard instance with default setup"""
    dashboard = create_real_time_dashboard()
    dashboard.setup_default_widgets()
    await dashboard.start()
    return dashboard


# WebSocket support for real-time updates

class DashboardWebSocketManager:
    """WebSocket manager for real-time dashboard updates"""
    
    def __init__(self, dashboard: RealTimeDashboard):
        self.dashboard = dashboard
        self.connections: Set[Any] = set()
    
    def add_connection(self, websocket: Any) -> None:
        """Add WebSocket connection"""
        self.connections.add(websocket)
        
        # Subscribe to dashboard updates
        self.dashboard.subscribe(self._notify_connection)
    
    def remove_connection(self, websocket: Any) -> None:
        """Remove WebSocket connection"""
        self.connections.discard(websocket)
    
    async def _notify_connection(self, dashboard_data: Dict[str, Any]) -> None:
        """Notify WebSocket connection of updates"""
        for connection in list(self.connections):
            try:
                await connection.send_text(json.dumps(dashboard_data))
            except Exception as e:
                logger.error("Error sending WebSocket update", error=str(e))
                self.connections.discard(connection)

