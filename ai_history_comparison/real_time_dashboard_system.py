"""
Real-time Dashboard System
==========================

Advanced real-time dashboard system for AI model analysis with live
monitoring, streaming data, and interactive dashboards.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import websockets
import aiohttp
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DashboardWidgetType(str, Enum):
    """Dashboard widget types"""
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    GAUGE = "gauge"
    INDICATOR = "indicator"
    HEATMAP = "heatmap"
    TABLE = "table"
    ALERT_PANEL = "alert_panel"
    STATUS_GRID = "status_grid"
    PROGRESS_BAR = "progress_bar"
    COUNTER = "counter"
    SPARKLINE = "sparkline"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    REAL_TIME_LOG = "real_time_log"


class DataStreamType(str, Enum):
    """Data stream types"""
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_METRICS = "system_metrics"
    ALERTS = "alerts"
    LOGS = "logs"
    PREDICTIONS = "predictions"
    COMPARISONS = "comparisons"
    BENCHMARKS = "benchmarks"
    CUSTOM = "custom"


class DashboardLayout(str, Enum):
    """Dashboard layouts"""
    GRID = "grid"
    FLEXIBLE = "flexible"
    RESPONSIVE = "responsive"
    CUSTOM = "custom"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: DashboardWidgetType
    title: str
    description: str
    data_source: str
    refresh_interval: int = 5
    size: Dict[str, int] = None
    position: Dict[str, int] = None
    styling: Dict[str, Any] = None
    filters: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.size is None:
            self.size = {"width": 4, "height": 3}
        if self.position is None:
            self.position = {"x": 0, "y": 0}
        if self.styling is None:
            self.styling = {}


@dataclass
class RealTimeDashboard:
    """Real-time dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    layout: DashboardLayout
    widgets: List[DashboardWidget]
    refresh_interval: int = 5
    auto_refresh: bool = True
    max_data_points: int = 1000
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class DataStream:
    """Data stream configuration"""
    stream_id: str
    stream_type: DataStreamType
    name: str
    description: str
    data_source: str
    update_frequency: int = 1
    buffer_size: int = 1000
    filters: Dict[str, Any] = None
    transformations: List[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.filters is None:
            self.filters = {}
        if self.transformations is None:
            self.transformations = []


@dataclass
class StreamData:
    """Stream data point"""
    stream_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealTimeDashboardSystem:
    """Advanced real-time dashboard system for AI model analysis"""
    
    def __init__(self, max_dashboards: int = 100, max_streams: int = 50):
        self.max_dashboards = max_dashboards
        self.max_streams = max_streams
        
        self.dashboards: Dict[str, RealTimeDashboard] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        self.data_streams: Dict[str, DataStream] = {}
        self.stream_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # WebSocket connections
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.data_queue = queue.Queue()
        self.running = False
        
        # Dashboard settings
        self.default_refresh_interval = 5
        self.max_data_points = 1000
        
        # Start background tasks
        self._start_background_tasks()
    
    async def create_dashboard(self, 
                             name: str,
                             description: str,
                             layout: DashboardLayout = DashboardLayout.GRID,
                             refresh_interval: int = 5) -> RealTimeDashboard:
        """Create real-time dashboard"""
        try:
            dashboard_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()
            
            dashboard = RealTimeDashboard(
                dashboard_id=dashboard_id,
                name=name,
                description=description,
                layout=layout,
                widgets=[],
                refresh_interval=refresh_interval
            )
            
            self.dashboards[dashboard_id] = dashboard
            
            logger.info(f"Created dashboard: {name}")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise e
    
    async def create_widget(self, 
                          widget_type: DashboardWidgetType,
                          title: str,
                          description: str,
                          data_source: str,
                          refresh_interval: int = 5,
                          size: Dict[str, int] = None,
                          position: Dict[str, int] = None,
                          styling: Dict[str, Any] = None) -> DashboardWidget:
        """Create dashboard widget"""
        try:
            widget_id = hashlib.md5(f"{widget_type}_{title}_{datetime.now()}".encode()).hexdigest()
            
            if size is None:
                size = {"width": 4, "height": 3}
            if position is None:
                position = {"x": 0, "y": 0}
            if styling is None:
                styling = {}
            
            widget = DashboardWidget(
                widget_id=widget_id,
                widget_type=widget_type,
                title=title,
                description=description,
                data_source=data_source,
                refresh_interval=refresh_interval,
                size=size,
                position=position,
                styling=styling
            )
            
            self.widgets[widget_id] = widget
            
            logger.info(f"Created widget: {title}")
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating widget: {str(e)}")
            raise e
    
    async def add_widget_to_dashboard(self, 
                                    dashboard_id: str,
                                    widget_id: str) -> bool:
        """Add widget to dashboard"""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            if widget_id not in self.widgets:
                raise ValueError(f"Widget {widget_id} not found")
            
            dashboard = self.dashboards[dashboard_id]
            widget = self.widgets[widget_id]
            
            # Check if widget already exists in dashboard
            if any(w.widget_id == widget_id for w in dashboard.widgets):
                return False
            
            dashboard.widgets.append(widget)
            
            logger.info(f"Added widget {widget_id} to dashboard {dashboard_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding widget to dashboard: {str(e)}")
            return False
    
    async def create_data_stream(self, 
                               stream_type: DataStreamType,
                               name: str,
                               description: str,
                               data_source: str,
                               update_frequency: int = 1,
                               buffer_size: int = 1000) -> DataStream:
        """Create data stream"""
        try:
            stream_id = hashlib.md5(f"{stream_type}_{name}_{datetime.now()}".encode()).hexdigest()
            
            stream = DataStream(
                stream_id=stream_id,
                stream_type=stream_type,
                name=name,
                description=description,
                data_source=data_source,
                update_frequency=update_frequency,
                buffer_size=buffer_size
            )
            
            self.data_streams[stream_id] = stream
            
            logger.info(f"Created data stream: {name}")
            
            return stream
            
        except Exception as e:
            logger.error(f"Error creating data stream: {str(e)}")
            raise e
    
    async def start_data_stream(self, stream_id: str) -> bool:
        """Start data stream"""
        try:
            if stream_id not in self.data_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream = self.data_streams[stream_id]
            
            # Start streaming task
            asyncio.create_task(self._stream_data(stream))
            
            logger.info(f"Started data stream: {stream_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting data stream: {str(e)}")
            return False
    
    async def get_dashboard_data(self, 
                               dashboard_id: str,
                               widget_id: str = None) -> Dict[str, Any]:
        """Get dashboard data"""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            dashboard = self.dashboards[dashboard_id]
            
            if widget_id:
                # Get specific widget data
                widget = next((w for w in dashboard.widgets if w.widget_id == widget_id), None)
                if not widget:
                    raise ValueError(f"Widget {widget_id} not found in dashboard")
                
                data = await self._get_widget_data(widget)
                return {
                    "widget_id": widget_id,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Get all dashboard data
                dashboard_data = {
                    "dashboard_id": dashboard_id,
                    "dashboard_name": dashboard.name,
                    "widgets": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                for widget in dashboard.widgets:
                    widget_data = await self._get_widget_data(widget)
                    dashboard_data["widgets"].append({
                        "widget_id": widget.widget_id,
                        "widget_type": widget.widget_type.value,
                        "title": widget.title,
                        "data": widget_data
                    })
                
                return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {"error": str(e)}
    
    async def get_stream_data(self, 
                            stream_id: str,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get stream data"""
        try:
            if stream_id not in self.data_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream_data = list(self.stream_data[stream_id])
            
            # Return last N data points
            return stream_data[-limit:] if limit > 0 else stream_data
            
        except Exception as e:
            logger.error(f"Error getting stream data: {str(e)}")
            return []
    
    async def create_performance_dashboard(self, 
                                         model_names: List[str] = None) -> RealTimeDashboard:
        """Create performance dashboard"""
        try:
            if model_names is None:
                model_names = ["gpt-4", "claude-3", "gemini-pro"]
            
            # Create dashboard
            dashboard = await self.create_dashboard(
                name="Performance Dashboard",
                description="Real-time performance monitoring dashboard",
                layout=DashboardLayout.GRID,
                refresh_interval=5
            )
            
            # Create performance metric cards
            for model_name in model_names:
                metric_card = await self.create_widget(
                    widget_type=DashboardWidgetType.METRIC_CARD,
                    title=f"{model_name} Performance",
                    description=f"Real-time performance metrics for {model_name}",
                    data_source=f"performance_{model_name}",
                    refresh_interval=5,
                    size={"width": 4, "height": 2}
                )
                await self.add_widget_to_dashboard(dashboard.dashboard_id, metric_card.widget_id)
            
            # Create performance trend chart
            trend_chart = await self.create_widget(
                widget_type=DashboardWidgetType.LINE_CHART,
                title="Performance Trends",
                description="Real-time performance trends for all models",
                data_source="performance_trends",
                refresh_interval=5,
                size={"width": 12, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, trend_chart.widget_id)
            
            # Create performance gauge
            performance_gauge = await self.create_widget(
                widget_type=DashboardWidgetType.GAUGE,
                title="Overall Performance",
                description="Overall system performance gauge",
                data_source="overall_performance",
                refresh_interval=5,
                size={"width": 4, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, performance_gauge.widget_id)
            
            # Create alert panel
            alert_panel = await self.create_widget(
                widget_type=DashboardWidgetType.ALERT_PANEL,
                title="Active Alerts",
                description="Real-time alerts and notifications",
                data_source="alerts",
                refresh_interval=1,
                size={"width": 8, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, alert_panel.widget_id)
            
            logger.info(f"Created performance dashboard with {len(dashboard.widgets)} widgets")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")
            raise e
    
    async def create_analytics_dashboard(self) -> RealTimeDashboard:
        """Create analytics dashboard"""
        try:
            # Create dashboard
            dashboard = await self.create_dashboard(
                name="Analytics Dashboard",
                description="Real-time analytics and insights dashboard",
                layout=DashboardLayout.GRID,
                refresh_interval=10
            )
            
            # Create analytics overview cards
            overview_card = await self.create_widget(
                widget_type=DashboardWidgetType.METRIC_CARD,
                title="Analytics Overview",
                description="Key analytics metrics overview",
                data_source="analytics_overview",
                refresh_interval=10,
                size={"width": 6, "height": 2}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, overview_card.widget_id)
            
            # Create insights counter
            insights_counter = await self.create_widget(
                widget_type=DashboardWidgetType.COUNTER,
                title="AI Insights Generated",
                description="Total AI insights generated today",
                data_source="insights_counter",
                refresh_interval=30,
                size={"width": 6, "height": 2}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, insights_counter.widget_id)
            
            # Create analytics heatmap
            analytics_heatmap = await self.create_widget(
                widget_type=DashboardWidgetType.HEATMAP,
                title="Analytics Heatmap",
                description="Analytics activity heatmap",
                data_source="analytics_heatmap",
                refresh_interval=60,
                size={"width": 8, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, analytics_heatmap.widget_id)
            
            # Create prediction accuracy chart
            prediction_chart = await self.create_widget(
                widget_type=DashboardWidgetType.LINE_CHART,
                title="Prediction Accuracy",
                description="Real-time prediction accuracy trends",
                data_source="prediction_accuracy",
                refresh_interval=15,
                size={"width": 4, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, prediction_chart.widget_id)
            
            logger.info(f"Created analytics dashboard with {len(dashboard.widgets)} widgets")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating analytics dashboard: {str(e)}")
            raise e
    
    async def create_system_monitoring_dashboard(self) -> RealTimeDashboard:
        """Create system monitoring dashboard"""
        try:
            # Create dashboard
            dashboard = await self.create_dashboard(
                name="System Monitoring Dashboard",
                description="Real-time system monitoring dashboard",
                layout=DashboardLayout.GRID,
                refresh_interval=2
            )
            
            # Create system status grid
            status_grid = await self.create_widget(
                widget_type=DashboardWidgetType.STATUS_GRID,
                title="System Status",
                description="Real-time system status overview",
                data_source="system_status",
                refresh_interval=2,
                size={"width": 12, "height": 3}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, status_grid.widget_id)
            
            # Create CPU usage gauge
            cpu_gauge = await self.create_widget(
                widget_type=DashboardWidgetType.GAUGE,
                title="CPU Usage",
                description="Real-time CPU usage",
                data_source="cpu_usage",
                refresh_interval=2,
                size={"width": 4, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, cpu_gauge.widget_id)
            
            # Create memory usage gauge
            memory_gauge = await self.create_widget(
                widget_type=DashboardWidgetType.GAUGE,
                title="Memory Usage",
                description="Real-time memory usage",
                data_source="memory_usage",
                refresh_interval=2,
                size={"width": 4, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, memory_gauge.widget_id)
            
            # Create disk usage gauge
            disk_gauge = await self.create_widget(
                widget_type=DashboardWidgetType.GAUGE,
                title="Disk Usage",
                description="Real-time disk usage",
                data_source="disk_usage",
                refresh_interval=5,
                size={"width": 4, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, disk_gauge.widget_id)
            
            # Create real-time log
            realtime_log = await self.create_widget(
                widget_type=DashboardWidgetType.REAL_TIME_LOG,
                title="System Logs",
                description="Real-time system logs",
                data_source="system_logs",
                refresh_interval=1,
                size={"width": 12, "height": 4}
            )
            await self.add_widget_to_dashboard(dashboard.dashboard_id, realtime_log.widget_id)
            
            logger.info(f"Created system monitoring dashboard with {len(dashboard.widgets)} widgets")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating system monitoring dashboard: {str(e)}")
            raise e
    
    async def get_dashboard_analytics(self, 
                                    time_range_hours: int = 24) -> Dict[str, Any]:
        """Get dashboard analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            analytics = {
                "total_dashboards": len(self.dashboards),
                "total_widgets": len(self.widgets),
                "total_streams": len(self.data_streams),
                "active_connections": len(self.websocket_connections),
                "dashboard_usage": {},
                "widget_usage": {},
                "stream_performance": {},
                "data_volume": 0
            }
            
            # Calculate data volume
            for stream_id, data in self.stream_data.items():
                analytics["data_volume"] += len(data)
            
            # Dashboard usage (simulated)
            for dashboard_id, dashboard in self.dashboards.items():
                analytics["dashboard_usage"][dashboard_id] = {
                    "name": dashboard.name,
                    "widget_count": len(dashboard.widgets),
                    "last_accessed": datetime.now().isoformat()
                }
            
            # Widget usage (simulated)
            for widget_id, widget in self.widgets.items():
                analytics["widget_usage"][widget_id] = {
                    "type": widget.widget_type.value,
                    "title": widget.title,
                    "refresh_interval": widget.refresh_interval
                }
            
            # Stream performance (simulated)
            for stream_id, stream in self.data_streams.items():
                data_count = len(self.stream_data[stream_id])
                analytics["stream_performance"][stream_id] = {
                    "name": stream.name,
                    "type": stream.stream_type.value,
                    "data_points": data_count,
                    "update_frequency": stream.update_frequency
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting dashboard analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get widget data based on type and data source"""
        try:
            widget_type = widget.widget_type
            data_source = widget.data_source
            
            if widget_type == DashboardWidgetType.METRIC_CARD:
                return await self._get_metric_card_data(data_source)
            elif widget_type == DashboardWidgetType.LINE_CHART:
                return await self._get_line_chart_data(data_source)
            elif widget_type == DashboardWidgetType.GAUGE:
                return await self._get_gauge_data(data_source)
            elif widget_type == DashboardWidgetType.INDICATOR:
                return await self._get_indicator_data(data_source)
            elif widget_type == DashboardWidgetType.HEATMAP:
                return await self._get_heatmap_data(data_source)
            elif widget_type == DashboardWidgetType.TABLE:
                return await self._get_table_data(data_source)
            elif widget_type == DashboardWidgetType.ALERT_PANEL:
                return await self._get_alert_panel_data(data_source)
            elif widget_type == DashboardWidgetType.STATUS_GRID:
                return await self._get_status_grid_data(data_source)
            elif widget_type == DashboardWidgetType.COUNTER:
                return await self._get_counter_data(data_source)
            elif widget_type == DashboardWidgetType.REAL_TIME_LOG:
                return await self._get_realtime_log_data(data_source)
            else:
                return {"error": f"Unsupported widget type: {widget_type}"}
                
        except Exception as e:
            logger.error(f"Error getting widget data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_metric_card_data(self, data_source: str) -> Dict[str, Any]:
        """Get metric card data"""
        try:
            # Generate sample metric data
            if "performance" in data_source:
                return {
                    "value": 0.85 + np.random.normal(0, 0.05),
                    "unit": "%",
                    "trend": "up",
                    "trend_value": 2.5,
                    "label": "Performance Score",
                    "color": "green"
                }
            else:
                return {
                    "value": np.random.randint(100, 1000),
                    "unit": "",
                    "trend": "stable",
                    "trend_value": 0,
                    "label": "Metric",
                    "color": "blue"
                }
        except Exception as e:
            logger.error(f"Error getting metric card data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_line_chart_data(self, data_source: str) -> Dict[str, Any]:
        """Get line chart data"""
        try:
            # Generate sample time series data
            timestamps = []
            values = []
            
            for i in range(20):
                timestamp = datetime.now() - timedelta(minutes=i*5)
                timestamps.append(timestamp.isoformat())
                values.append(0.7 + np.random.normal(0, 0.1))
            
            return {
                "timestamps": timestamps[::-1],
                "values": values[::-1],
                "title": "Performance Trend",
                "x_label": "Time",
                "y_label": "Performance"
            }
        except Exception as e:
            logger.error(f"Error getting line chart data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_gauge_data(self, data_source: str) -> Dict[str, Any]:
        """Get gauge data"""
        try:
            if "cpu" in data_source:
                return {
                    "value": np.random.uniform(20, 80),
                    "min": 0,
                    "max": 100,
                    "unit": "%",
                    "label": "CPU Usage",
                    "color": "orange"
                }
            elif "memory" in data_source:
                return {
                    "value": np.random.uniform(30, 70),
                    "min": 0,
                    "max": 100,
                    "unit": "%",
                    "label": "Memory Usage",
                    "color": "blue"
                }
            else:
                return {
                    "value": np.random.uniform(0, 100),
                    "min": 0,
                    "max": 100,
                    "unit": "%",
                    "label": "Gauge",
                    "color": "green"
                }
        except Exception as e:
            logger.error(f"Error getting gauge data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_indicator_data(self, data_source: str) -> Dict[str, Any]:
        """Get indicator data"""
        try:
            statuses = ["online", "offline", "warning", "error"]
            return {
                "status": np.random.choice(statuses),
                "label": "System Status",
                "value": np.random.uniform(0, 100),
                "unit": "%"
            }
        except Exception as e:
            logger.error(f"Error getting indicator data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_heatmap_data(self, data_source: str) -> Dict[str, Any]:
        """Get heatmap data"""
        try:
            # Generate sample heatmap data
            hours = list(range(24))
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            data = []
            for day in days:
                for hour in hours:
                    data.append({
                        "day": day,
                        "hour": hour,
                        "value": np.random.uniform(0, 100)
                    })
            
            return {
                "data": data,
                "x_axis": "Hour",
                "y_axis": "Day",
                "title": "Activity Heatmap"
            }
        except Exception as e:
            logger.error(f"Error getting heatmap data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_table_data(self, data_source: str) -> Dict[str, Any]:
        """Get table data"""
        try:
            # Generate sample table data
            columns = ["Model", "Performance", "Status", "Last Update"]
            rows = []
            
            models = ["gpt-4", "claude-3", "gemini-pro"]
            for model in models:
                rows.append([
                    model,
                    f"{0.7 + np.random.normal(0, 0.1):.3f}",
                    np.random.choice(["Active", "Warning", "Error"]),
                    datetime.now().strftime("%H:%M:%S")
                ])
            
            return {
                "columns": columns,
                "rows": rows,
                "title": "Model Status"
            }
        except Exception as e:
            logger.error(f"Error getting table data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_alert_panel_data(self, data_source: str) -> Dict[str, Any]:
        """Get alert panel data"""
        try:
            # Generate sample alert data
            alerts = []
            alert_types = ["warning", "error", "info", "success"]
            
            for i in range(np.random.randint(0, 5)):
                alerts.append({
                    "id": f"alert_{i}",
                    "type": np.random.choice(alert_types),
                    "message": f"Alert message {i+1}",
                    "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                    "severity": np.random.choice(["low", "medium", "high", "critical"])
                })
            
            return {
                "alerts": alerts,
                "total_count": len(alerts),
                "critical_count": len([a for a in alerts if a["severity"] == "critical"])
            }
        except Exception as e:
            logger.error(f"Error getting alert panel data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_status_grid_data(self, data_source: str) -> Dict[str, Any]:
        """Get status grid data"""
        try:
            # Generate sample status data
            services = ["API", "Database", "Cache", "Queue", "Storage", "Monitoring"]
            statuses = []
            
            for service in services:
                statuses.append({
                    "service": service,
                    "status": np.random.choice(["online", "offline", "warning"]),
                    "uptime": f"{np.random.randint(95, 100)}%",
                    "response_time": f"{np.random.uniform(10, 500):.1f}ms"
                })
            
            return {
                "services": statuses,
                "overall_status": "online" if all(s["status"] == "online" for s in statuses) else "warning"
            }
        except Exception as e:
            logger.error(f"Error getting status grid data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_counter_data(self, data_source: str) -> Dict[str, Any]:
        """Get counter data"""
        try:
            return {
                "value": np.random.randint(1000, 10000),
                "label": "Total Count",
                "increment": np.random.randint(1, 10),
                "trend": "up"
            }
        except Exception as e:
            logger.error(f"Error getting counter data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_realtime_log_data(self, data_source: str) -> Dict[str, Any]:
        """Get real-time log data"""
        try:
            # Generate sample log data
            logs = []
            log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
            
            for i in range(10):
                logs.append({
                    "timestamp": (datetime.now() - timedelta(seconds=i*30)).strftime("%H:%M:%S"),
                    "level": np.random.choice(log_levels),
                    "message": f"Log message {i+1}",
                    "source": np.random.choice(["API", "Database", "System"])
                })
            
            return {
                "logs": logs,
                "total_count": len(logs),
                "error_count": len([l for l in logs if l["level"] == "ERROR"])
            }
        except Exception as e:
            logger.error(f"Error getting real-time log data: {str(e)}")
            return {"error": str(e)}
    
    async def _stream_data(self, stream: DataStream) -> None:
        """Stream data for a data stream"""
        try:
            while self.running:
                # Generate sample data based on stream type
                data = await self._generate_stream_data(stream)
                
                # Create stream data point
                stream_data_point = StreamData(
                    stream_id=stream.stream_id,
                    timestamp=datetime.now(),
                    data=data
                )
                
                # Add to stream buffer
                self.stream_data[stream.stream_id].append(stream_data_point)
                
                # Notify WebSocket connections
                await self._notify_websocket_connections(stream.stream_id, data)
                
                # Wait for next update
                await asyncio.sleep(stream.update_frequency)
                
        except Exception as e:
            logger.error(f"Error streaming data for {stream.stream_id}: {str(e)}")
    
    async def _generate_stream_data(self, stream: DataStream) -> Dict[str, Any]:
        """Generate sample stream data"""
        try:
            stream_type = stream.stream_type
            
            if stream_type == DataStreamType.PERFORMANCE_METRICS:
                return {
                    "model_name": "gpt-4",
                    "performance_score": 0.7 + np.random.normal(0, 0.1),
                    "response_time": 1.0 + np.random.normal(0, 0.3),
                    "accuracy": 0.85 + np.random.normal(0, 0.05)
                }
            elif stream_type == DataStreamType.SYSTEM_METRICS:
                return {
                    "cpu_usage": np.random.uniform(20, 80),
                    "memory_usage": np.random.uniform(30, 70),
                    "disk_usage": np.random.uniform(40, 90),
                    "network_io": np.random.uniform(0, 1000)
                }
            elif stream_type == DataStreamType.ALERTS:
                return {
                    "alert_id": f"alert_{int(time.time())}",
                    "type": np.random.choice(["warning", "error", "info"]),
                    "message": f"Alert message {int(time.time())}",
                    "severity": np.random.choice(["low", "medium", "high"])
                }
            else:
                return {
                    "value": np.random.uniform(0, 100),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error generating stream data: {str(e)}")
            return {}
    
    async def _notify_websocket_connections(self, stream_id: str, data: Dict[str, Any]) -> None:
        """Notify WebSocket connections about new data"""
        try:
            message = {
                "type": "stream_update",
                "stream_id": stream_id,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected WebSocket clients
            disconnected = []
            for connection_id, websocket in self.websocket_connections.items():
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket {connection_id}: {str(e)}")
                    disconnected.append(connection_id)
            
            # Remove disconnected connections
            for connection_id in disconnected:
                del self.websocket_connections[connection_id]
                
        except Exception as e:
            logger.error(f"Error notifying WebSocket connections: {str(e)}")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        try:
            self.running = True
            
            # Start data streaming task
            streaming_thread = threading.Thread(target=self._run_streaming_tasks, daemon=True)
            streaming_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {str(e)}")
    
    def _run_streaming_tasks(self) -> None:
        """Run streaming tasks in background thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start all data streams
            for stream in self.data_streams.values():
                loop.create_task(self._stream_data(stream))
            
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"Error in streaming tasks: {str(e)}")


# Global dashboard system instance
_dashboard_system: Optional[RealTimeDashboardSystem] = None


def get_real_time_dashboard_system(max_dashboards: int = 100, max_streams: int = 50) -> RealTimeDashboardSystem:
    """Get or create global real-time dashboard system instance"""
    global _dashboard_system
    if _dashboard_system is None:
        _dashboard_system = RealTimeDashboardSystem(max_dashboards, max_streams)
    return _dashboard_system


# Example usage
async def main():
    """Example usage of the real-time dashboard system"""
    system = get_real_time_dashboard_system()
    
    # Create performance dashboard
    perf_dashboard = await system.create_performance_dashboard(
        model_names=["gpt-4", "claude-3", "gemini-pro"]
    )
    print(f"Created performance dashboard: {perf_dashboard.dashboard_id}")
    
    # Create analytics dashboard
    analytics_dashboard = await system.create_analytics_dashboard()
    print(f"Created analytics dashboard: {analytics_dashboard.dashboard_id}")
    
    # Create system monitoring dashboard
    system_dashboard = await system.create_system_monitoring_dashboard()
    print(f"Created system monitoring dashboard: {system_dashboard.dashboard_id}")
    
    # Create data stream
    performance_stream = await system.create_data_stream(
        stream_type=DataStreamType.PERFORMANCE_METRICS,
        name="Performance Metrics Stream",
        description="Real-time performance metrics stream",
        data_source="performance_data",
        update_frequency=5
    )
    print(f"Created performance stream: {performance_stream.stream_id}")
    
    # Start data stream
    await system.start_data_stream(performance_stream.stream_id)
    print("Started performance stream")
    
    # Get dashboard data
    dashboard_data = await system.get_dashboard_data(perf_dashboard.dashboard_id)
    print(f"Dashboard data: {len(dashboard_data.get('widgets', []))} widgets")
    
    # Get stream data
    stream_data = await system.get_stream_data(performance_stream.stream_id, limit=10)
    print(f"Stream data: {len(stream_data)} data points")
    
    # Get dashboard analytics
    analytics = await system.get_dashboard_analytics()
    print(f"Dashboard analytics: {analytics.get('total_dashboards', 0)} dashboards")


if __name__ == "__main__":
    asyncio.run(main())

























