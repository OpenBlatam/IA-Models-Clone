"""
BUL Real-Time Analytics Dashboard
=================================

Real-time analytics dashboard for monitoring document generation, system performance, and business insights.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class TimeWindow(str, Enum):
    """Time windows for analytics"""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"

@dataclass
class MetricData:
    """Metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class TimeSeriesData:
    """Time series data"""
    metric_name: str
    time_window: TimeWindow
    data_points: List[Tuple[datetime, float]]
    aggregation: str  # sum, avg, min, max, count
    metadata: Dict[str, Any]

@dataclass
class DashboardWidget:
    """Dashboard widget"""
    widget_id: str
    widget_type: str
    title: str
    description: str
    metric_name: str
    time_window: TimeWindow
    refresh_interval: int  # seconds
    position: Tuple[int, int]  # x, y
    size: Tuple[int, int]  # width, height
    config: Dict[str, Any]
    is_active: bool
    created_at: datetime
    last_updated: Optional[datetime] = None

@dataclass
class AlertRule:
    """Alert rule"""
    rule_id: str
    rule_name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    time_window: int  # seconds
    severity: str  # low, medium, high, critical
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    triggered_at: datetime
    is_acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

class RealTimeAnalyticsDashboard:
    """Real-Time Analytics Dashboard"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Data storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.time_series_data: Dict[str, TimeSeriesData] = {}
        self.dashboard_widgets: Dict[str, DashboardWidget] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        self.connection_lock = threading.Lock()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Initialize dashboard
        self._initialize_dashboard()
    
    def _initialize_dashboard(self):
        """Initialize the analytics dashboard"""
        try:
            # Create default widgets
            self._create_default_widgets()
            
            # Create default alert rules
            self._create_default_alert_rules()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("Real-Time Analytics Dashboard initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Real-Time Analytics Dashboard: {e}")
    
    def _create_default_widgets(self):
        """Create default dashboard widgets"""
        try:
            # Document Generation Rate Widget
            doc_gen_widget = DashboardWidget(
                widget_id="doc_generation_rate",
                widget_type="line_chart",
                title="Document Generation Rate",
                description="Number of documents generated per minute",
                metric_name="documents_generated",
                time_window=TimeWindow.LAST_HOUR,
                refresh_interval=30,
                position=(0, 0),
                size=(6, 4),
                config={"y_axis_label": "Documents/min", "color": "#3498db"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # System Performance Widget
            perf_widget = DashboardWidget(
                widget_id="system_performance",
                widget_type="gauge",
                title="System Performance",
                description="Overall system performance score",
                metric_name="system_performance",
                time_window=TimeWindow.LAST_HOUR,
                refresh_interval=60,
                position=(6, 0),
                size=(3, 4),
                config={"min_value": 0, "max_value": 100, "color": "#2ecc71"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # API Response Time Widget
            api_widget = DashboardWidget(
                widget_id="api_response_time",
                widget_type="histogram",
                title="API Response Time",
                description="Distribution of API response times",
                metric_name="api_response_time",
                time_window=TimeWindow.LAST_HOUR,
                refresh_interval=30,
                position=(9, 0),
                size=(3, 4),
                config={"buckets": [0.1, 0.5, 1.0, 2.0, 5.0], "color": "#e74c3c"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # Business Area Distribution Widget
            business_widget = DashboardWidget(
                widget_id="business_area_distribution",
                widget_type="pie_chart",
                title="Business Area Distribution",
                description="Distribution of documents by business area",
                metric_name="business_area_distribution",
                time_window=TimeWindow.LAST_DAY,
                refresh_interval=300,
                position=(0, 4),
                size=(4, 4),
                config={"show_percentages": True, "colors": ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]},
                is_active=True,
                created_at=datetime.now()
            )
            
            # Error Rate Widget
            error_widget = DashboardWidget(
                widget_id="error_rate",
                widget_type="bar_chart",
                title="Error Rate",
                description="Error rate by error type",
                metric_name="error_rate",
                time_window=TimeWindow.LAST_DAY,
                refresh_interval=60,
                position=(4, 4),
                size=(4, 4),
                config={"y_axis_label": "Errors/hour", "color": "#e74c3c"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # User Activity Widget
            user_widget = DashboardWidget(
                widget_id="user_activity",
                widget_type="heatmap",
                title="User Activity",
                description="User activity throughout the day",
                metric_name="user_activity",
                time_window=TimeWindow.LAST_DAY,
                refresh_interval=300,
                position=(8, 4),
                size=(4, 4),
                config={"time_format": "hour", "color_scheme": "blues"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # Quality Score Widget
            quality_widget = DashboardWidget(
                widget_id="document_quality",
                widget_type="line_chart",
                title="Document Quality Score",
                description="Average document quality score over time",
                metric_name="document_quality_score",
                time_window=TimeWindow.LAST_DAY,
                refresh_interval=60,
                position=(0, 8),
                size=(6, 4),
                config={"y_axis_label": "Quality Score", "color": "#9b59b6"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # Cache Hit Rate Widget
            cache_widget = DashboardWidget(
                widget_id="cache_hit_rate",
                widget_type="gauge",
                title="Cache Hit Rate",
                description="Percentage of cache hits",
                metric_name="cache_hit_rate",
                time_window=TimeWindow.LAST_HOUR,
                refresh_interval=30,
                position=(6, 8),
                size=(3, 4),
                config={"min_value": 0, "max_value": 100, "color": "#f39c12"},
                is_active=True,
                created_at=datetime.now()
            )
            
            # Active Alerts Widget
            alerts_widget = DashboardWidget(
                widget_id="active_alerts",
                widget_type="table",
                title="Active Alerts",
                description="Currently active system alerts",
                metric_name="active_alerts",
                time_window=TimeWindow.LAST_HOUR,
                refresh_interval=10,
                position=(9, 8),
                size=(3, 4),
                config={"max_rows": 10, "show_severity": True},
                is_active=True,
                created_at=datetime.now()
            )
            
            self.dashboard_widgets.update({
                doc_gen_widget.widget_id: doc_gen_widget,
                perf_widget.widget_id: perf_widget,
                api_widget.widget_id: api_widget,
                business_widget.widget_id: business_widget,
                error_widget.widget_id: error_widget,
                user_widget.widget_id: user_widget,
                quality_widget.widget_id: quality_widget,
                cache_widget.widget_id: cache_widget,
                alerts_widget.widget_id: alerts_widget
            })
            
            self.logger.info(f"Created {len(self.dashboard_widgets)} default dashboard widgets")
        
        except Exception as e:
            self.logger.error(f"Error creating default widgets: {e}")
    
    def _create_default_alert_rules(self):
        """Create default alert rules"""
        try:
            # High Error Rate Alert
            error_alert = AlertRule(
                rule_id="high_error_rate",
                rule_name="High Error Rate",
                metric_name="error_rate",
                condition=">",
                threshold=5.0,  # 5 errors per minute
                time_window=300,  # 5 minutes
                severity="high",
                is_active=True,
                created_at=datetime.now()
            )
            
            # Low System Performance Alert
            perf_alert = AlertRule(
                rule_id="low_system_performance",
                rule_name="Low System Performance",
                metric_name="system_performance",
                condition="<",
                threshold=70.0,  # Below 70%
                time_window=600,  # 10 minutes
                severity="medium",
                is_active=True,
                created_at=datetime.now()
            )
            
            # High API Response Time Alert
            api_alert = AlertRule(
                rule_id="high_api_response_time",
                rule_name="High API Response Time",
                metric_name="api_response_time",
                condition=">",
                threshold=5.0,  # Above 5 seconds
                time_window=300,  # 5 minutes
                severity="high",
                is_active=True,
                created_at=datetime.now()
            )
            
            # Low Cache Hit Rate Alert
            cache_alert = AlertRule(
                rule_id="low_cache_hit_rate",
                rule_name="Low Cache Hit Rate",
                metric_name="cache_hit_rate",
                condition="<",
                threshold=60.0,  # Below 60%
                time_window=900,  # 15 minutes
                severity="medium",
                is_active=True,
                created_at=datetime.now()
            )
            
            # Low Document Quality Alert
            quality_alert = AlertRule(
                rule_id="low_document_quality",
                rule_name="Low Document Quality",
                metric_name="document_quality_score",
                condition="<",
                threshold=0.6,  # Below 60%
                time_window=1800,  # 30 minutes
                severity="low",
                is_active=True,
                created_at=datetime.now()
            )
            
            self.alert_rules.update({
                error_alert.rule_id: error_alert,
                perf_alert.rule_id: perf_alert,
                api_alert.rule_id: api_alert,
                cache_alert.rule_id: cache_alert,
                quality_alert.rule_id: quality_alert
            })
            
            self.logger.info(f"Created {len(self.alert_rules)} default alert rules")
        
        except Exception as e:
            self.logger.error(f"Error creating default alert rules: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            self.is_running = True
            
            # Start metric collection task
            task1 = asyncio.create_task(self._metric_collection_task())
            self.background_tasks.append(task1)
            
            # Start alert monitoring task
            task2 = asyncio.create_task(self._alert_monitoring_task())
            self.background_tasks.append(task2)
            
            # Start data aggregation task
            task3 = asyncio.create_task(self._data_aggregation_task())
            self.background_tasks.append(task3)
            
            # Start WebSocket broadcasting task
            task4 = asyncio.create_task(self._websocket_broadcast_task())
            self.background_tasks.append(task4)
            
            self.logger.info("Started background tasks for analytics dashboard")
        
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
    
    async def _metric_collection_task(self):
        """Background task for collecting metrics"""
        while self.is_running:
            try:
                # Simulate metric collection
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Error in metric collection task: {e}")
                await asyncio.sleep(10)
    
    async def _alert_monitoring_task(self):
        """Background task for monitoring alerts"""
        while self.is_running:
            try:
                await self._check_alert_rules()
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                self.logger.error(f"Error in alert monitoring task: {e}")
                await asyncio.sleep(30)
    
    async def _data_aggregation_task(self):
        """Background task for data aggregation"""
        while self.is_running:
            try:
                await self._aggregate_time_series_data()
                await asyncio.sleep(60)  # Aggregate every minute
            
            except Exception as e:
                self.logger.error(f"Error in data aggregation task: {e}")
                await asyncio.sleep(60)
    
    async def _websocket_broadcast_task(self):
        """Background task for WebSocket broadcasting"""
        while self.is_running:
            try:
                if self.active_connections:
                    await self._broadcast_dashboard_update()
                await asyncio.sleep(5)  # Broadcast every 5 seconds
            
            except Exception as e:
                self.logger.error(f"Error in WebSocket broadcast task: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            current_time = datetime.now()
            
            # Simulate document generation rate
            doc_rate = np.random.poisson(2)  # Average 2 documents per minute
            self._add_metric("documents_generated", MetricType.COUNTER, doc_rate, current_time)
            
            # Simulate system performance
            perf_score = np.random.normal(85, 10)  # Average 85% with std 10
            perf_score = max(0, min(100, perf_score))
            self._add_metric("system_performance", MetricType.GAUGE, perf_score, current_time)
            
            # Simulate API response time
            response_time = np.random.exponential(1.5)  # Average 1.5 seconds
            self._add_metric("api_response_time", MetricType.HISTOGRAM, response_time, current_time)
            
            # Simulate error rate
            error_rate = np.random.poisson(0.5)  # Average 0.5 errors per minute
            self._add_metric("error_rate", MetricType.RATE, error_rate, current_time)
            
            # Simulate cache hit rate
            cache_hit_rate = np.random.normal(75, 5)  # Average 75% with std 5
            cache_hit_rate = max(0, min(100, cache_hit_rate))
            self._add_metric("cache_hit_rate", MetricType.GAUGE, cache_hit_rate, current_time)
            
            # Simulate document quality score
            quality_score = np.random.normal(0.75, 0.1)  # Average 0.75 with std 0.1
            quality_score = max(0, min(1, quality_score))
            self._add_metric("document_quality_score", MetricType.GAUGE, quality_score, current_time)
            
            # Simulate user activity
            user_activity = np.random.poisson(3)  # Average 3 active users
            self._add_metric("user_activity", MetricType.GAUGE, user_activity, current_time)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _add_metric(self, metric_name: str, metric_type: MetricType, value: float, timestamp: datetime, labels: Dict[str, str] = None):
        """Add a metric data point"""
        try:
            metric_data = MetricData(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                labels=labels or {},
                metadata={}
            )
            
            self.metrics[metric_name].append(metric_data)
        
        except Exception as e:
            self.logger.error(f"Error adding metric {metric_name}: {e}")
    
    async def _check_alert_rules(self):
        """Check alert rules and trigger alerts if needed"""
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.is_active:
                    continue
                
                # Get recent metric values
                recent_values = self._get_recent_metric_values(rule.metric_name, rule.time_window)
                
                if not recent_values:
                    continue
                
                # Calculate aggregated value
                if rule.metric_name in ["error_rate", "api_response_time"]:
                    # For rate metrics, use average
                    current_value = np.mean(recent_values)
                else:
                    # For other metrics, use latest value
                    current_value = recent_values[-1] if recent_values else 0
                
                # Check condition
                should_trigger = False
                if rule.condition == ">":
                    should_trigger = current_value > rule.threshold
                elif rule.condition == "<":
                    should_trigger = current_value < rule.threshold
                elif rule.condition == ">=":
                    should_trigger = current_value >= rule.threshold
                elif rule.condition == "<=":
                    should_trigger = current_value <= rule.threshold
                elif rule.condition == "==":
                    should_trigger = current_value == rule.threshold
                elif rule.condition == "!=":
                    should_trigger = current_value != rule.threshold
                
                if should_trigger:
                    # Check if alert already exists
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if alert.rule_id == rule_id and not alert.is_acknowledged:
                            existing_alert = alert
                            break
                    
                    if not existing_alert:
                        # Create new alert
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            rule_id=rule_id,
                            metric_name=rule.metric_name,
                            current_value=current_value,
                            threshold=rule.threshold,
                            severity=rule.severity,
                            message=f"{rule.rule_name}: {rule.metric_name} is {current_value:.2f} (threshold: {rule.threshold})",
                            triggered_at=datetime.now()
                        )
                        
                        self.active_alerts[alert.alert_id] = alert
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
                        
                        self.logger.warning(f"Alert triggered: {alert.message}")
        
        except Exception as e:
            self.logger.error(f"Error checking alert rules: {e}")
    
    def _get_recent_metric_values(self, metric_name: str, time_window_seconds: int) -> List[float]:
        """Get recent metric values within time window"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
            recent_values = []
            
            for metric_data in self.metrics[metric_name]:
                if metric_data.timestamp >= cutoff_time:
                    recent_values.append(metric_data.value)
            
            return recent_values
        
        except Exception as e:
            self.logger.error(f"Error getting recent metric values: {e}")
            return []
    
    async def _aggregate_time_series_data(self):
        """Aggregate time series data for widgets"""
        try:
            for widget_id, widget in self.dashboard_widgets.items():
                if not widget.is_active:
                    continue
                
                # Get time window
                time_window_seconds = self._get_time_window_seconds(widget.time_window)
                cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
                
                # Get metric data
                metric_data = []
                for data_point in self.metrics[widget.metric_name]:
                    if data_point.timestamp >= cutoff_time:
                        metric_data.append((data_point.timestamp, data_point.value))
                
                # Aggregate data
                if metric_data:
                    # Group by time intervals (e.g., 1 minute)
                    interval_seconds = 60
                    aggregated_data = self._aggregate_data_by_interval(metric_data, interval_seconds)
                    
                    # Create time series data
                    time_series = TimeSeriesData(
                        metric_name=widget.metric_name,
                        time_window=widget.time_window,
                        data_points=aggregated_data,
                        aggregation="avg",
                        metadata={"widget_id": widget_id}
                    )
                    
                    self.time_series_data[widget_id] = time_series
        
        except Exception as e:
            self.logger.error(f"Error aggregating time series data: {e}")
    
    def _get_time_window_seconds(self, time_window: TimeWindow) -> int:
        """Convert time window to seconds"""
        time_windows = {
            TimeWindow.LAST_HOUR: 3600,
            TimeWindow.LAST_DAY: 86400,
            TimeWindow.LAST_WEEK: 604800,
            TimeWindow.LAST_MONTH: 2592000
        }
        return time_windows.get(time_window, 3600)
    
    def _aggregate_data_by_interval(self, data_points: List[Tuple[datetime, float]], interval_seconds: int) -> List[Tuple[datetime, float]]:
        """Aggregate data points by time intervals"""
        try:
            if not data_points:
                return []
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x[0])
            
            # Group by intervals
            intervals = {}
            for timestamp, value in data_points:
                # Round timestamp to interval
                interval_start = timestamp.replace(second=0, microsecond=0)
                interval_start = interval_start.replace(minute=(interval_start.minute // (interval_seconds // 60)) * (interval_seconds // 60))
                
                if interval_start not in intervals:
                    intervals[interval_start] = []
                intervals[interval_start].append(value)
            
            # Calculate averages for each interval
            aggregated = []
            for interval_start, values in intervals.items():
                avg_value = np.mean(values)
                aggregated.append((interval_start, avg_value))
            
            return aggregated
        
        except Exception as e:
            self.logger.error(f"Error aggregating data by interval: {e}")
            return []
    
    async def _broadcast_dashboard_update(self):
        """Broadcast dashboard update to WebSocket connections"""
        try:
            if not self.active_connections:
                return
            
            # Prepare update data
            update_data = {
                "timestamp": datetime.now().isoformat(),
                "widgets": {},
                "alerts": [asdict(alert) for alert in self.active_alerts.values() if not alert.is_acknowledged]
            }
            
            # Add widget data
            for widget_id, widget in self.dashboard_widgets.items():
                if widget.is_active and widget_id in self.time_series_data:
                    time_series = self.time_series_data[widget_id]
                    update_data["widgets"][widget_id] = {
                        "title": widget.title,
                        "data": time_series.data_points,
                        "config": widget.config
                    }
            
            # Broadcast to all connections
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(update_data))
                except Exception as e:
                    self.logger.warning(f"Error broadcasting to WebSocket: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.active_connections.remove(connection)
        
        except Exception as e:
            self.logger.error(f"Error broadcasting dashboard update: {e}")
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection for real-time updates"""
        try:
            await websocket.accept()
            with self.connection_lock:
                self.active_connections.append(websocket)
            self.logger.info(f"WebSocket connection added. Total connections: {len(self.active_connections)}")
        
        except Exception as e:
            self.logger.error(f"Error adding WebSocket connection: {e}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        try:
            with self.connection_lock:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket connection removed. Total connections: {len(self.active_connections)}")
        
        except Exception as e:
            self.logger.error(f"Error removing WebSocket connection: {e}")
    
    async def get_dashboard_data(self, widget_ids: List[str] = None) -> Dict[str, Any]:
        """Get dashboard data for specified widgets"""
        try:
            if widget_ids is None:
                widget_ids = list(self.dashboard_widgets.keys())
            
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "widgets": {},
                "alerts": [asdict(alert) for alert in self.active_alerts.values() if not alert.is_acknowledged]
            }
            
            for widget_id in widget_ids:
                if widget_id in self.dashboard_widgets:
                    widget = self.dashboard_widgets[widget_id]
                    if widget.is_active and widget_id in self.time_series_data:
                        time_series = self.time_series_data[widget_id]
                        dashboard_data["widgets"][widget_id] = {
                            "title": widget.title,
                            "description": widget.description,
                            "widget_type": widget.widget_type,
                            "data": time_series.data_points,
                            "config": widget.config,
                            "last_updated": widget.last_updated.isoformat() if widget.last_updated else None
                        }
            
            return dashboard_data
        
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.is_acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get analytics dashboard system status"""
        try:
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            active_widgets = len([w for w in self.dashboard_widgets.values() if w.is_active])
            active_alerts = len([a for a in self.active_alerts.values() if not a.is_acknowledged])
            active_rules = len([r for r in self.alert_rules.values() if r.is_active])
            
            return {
                'total_metrics': total_metrics,
                'active_widgets': active_widgets,
                'active_alerts': active_alerts,
                'active_rules': active_rules,
                'websocket_connections': len(self.active_connections),
                'background_tasks': len(self.background_tasks),
                'is_running': self.is_running,
                'system_health': 'active' if self.is_running else 'inactive'
            }
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}

# Global analytics dashboard
_analytics_dashboard: Optional[RealTimeAnalyticsDashboard] = None

def get_analytics_dashboard() -> RealTimeAnalyticsDashboard:
    """Get the global analytics dashboard"""
    global _analytics_dashboard
    if _analytics_dashboard is None:
        _analytics_dashboard = RealTimeAnalyticsDashboard()
    return _analytics_dashboard

# Analytics dashboard router
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])

@analytics_router.get("/dashboard")
async def get_dashboard_data_endpoint(widget_ids: str = None):
    """Get dashboard data"""
    try:
        dashboard = get_analytics_dashboard()
        widget_id_list = widget_ids.split(',') if widget_ids else None
        data = await dashboard.get_dashboard_data(widget_id_list)
        return {"data": data, "success": True}
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")

@analytics_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    dashboard = get_analytics_dashboard()
    await dashboard.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await dashboard.remove_websocket_connection(websocket)

@analytics_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert_endpoint(alert_id: str, acknowledged_by: str = "system"):
    """Acknowledge an alert"""
    try:
        dashboard = get_analytics_dashboard()
        success = await dashboard.acknowledge_alert(alert_id, acknowledged_by)
        return {"success": success}
    
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@analytics_router.get("/status")
async def get_analytics_status_endpoint():
    """Get analytics dashboard status"""
    try:
        dashboard = get_analytics_dashboard()
        status = await dashboard.get_system_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting analytics status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics status")

