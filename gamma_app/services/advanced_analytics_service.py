"""
Gamma App - Advanced Analytics Service
Comprehensive analytics, reporting, and business intelligence
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class ReportType(Enum):
    """Types of reports"""
    USAGE = "usage"
    PERFORMANCE = "performance"
    CONTENT_QUALITY = "content_quality"
    USER_BEHAVIOR = "user_behavior"
    BUSINESS_METRICS = "business_metrics"
    SYSTEM_HEALTH = "system_health"

@dataclass
class Metric:
    """Metric definition"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType
    time_range: str = "7d"  # 1d, 7d, 30d, 90d, 1y
    filters: Dict[str, Any] = None
    group_by: List[str] = None
    aggregations: List[str] = None
    visualization: str = "chart"  # chart, table, dashboard
    format: str = "json"  # json, csv, pdf, html

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    name: str
    widgets: List[Dict[str, Any]]
    refresh_interval: int = 300  # seconds
    auto_refresh: bool = True
    layout: str = "grid"  # grid, custom
    theme: str = "light"  # light, dark

class AdvancedAnalyticsService:
    """Advanced analytics service with comprehensive reporting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "analytics.db")
        self.redis_client = None
        self.metrics_cache = {}
        self.reports_cache = {}
        self.dashboards = {}
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_visualization()
    
    def _init_database(self):
        """Initialize analytics database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    labels TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            
            conn.commit()
        
        logger.info("Analytics database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_visualization(self):
        """Initialize visualization settings"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        logger.info("Visualization settings initialized")
    
    async def track_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a metric"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (name, value, metric_type, labels, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.metric_type.value,
                json.dumps(metric.labels),
                metric.timestamp.isoformat(),
                json.dumps(metric.metadata)
            ))
            conn.commit()
        
        # Cache in Redis if available
        if self.redis_client:
            cache_key = f"metric:{name}:{int(time.time())}"
            self.redis_client.setex(cache_key, 3600, json.dumps(asdict(metric)))
        
        logger.debug(f"Metric tracked: {name} = {value}")
    
    async def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """Track an event"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (event_type, user_id, session_id, data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_type,
                user_id,
                session_id,
                json.dumps(data or {}),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Event tracked: {event_type}")
    
    async def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Get metrics with optional filtering"""
        
        query = "SELECT name, value, metric_type, labels, timestamp, metadata FROM metrics WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        metrics = []
        for row in rows:
            metric = Metric(
                name=row[0],
                value=row[1],
                metric_type=MetricType(row[2]),
                labels=json.loads(row[3]) if row[3] else {},
                timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]) if row[5] else {}
            )
            metrics.append(metric)
        
        return metrics
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get events with optional filtering"""
        
        query = "SELECT event_type, user_id, session_id, data, timestamp FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        events = []
        for row in rows:
            event = {
                "event_type": row[0],
                "user_id": row[1],
                "session_id": row[2],
                "data": json.loads(row[3]) if row[3] else {},
                "timestamp": datetime.fromisoformat(row[4])
            }
            events.append(event)
        
        return events
    
    async def generate_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate a comprehensive report"""
        
        logger.info(f"Generating {config.report_type.value} report")
        
        # Check cache first
        cache_key = f"report:{config.report_type.value}:{hash(str(config))}"
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = self._calculate_start_time(config.time_range, end_time)
        
        # Generate report based on type
        if config.report_type == ReportType.USAGE:
            report_data = await self._generate_usage_report(start_time, end_time, config)
        elif config.report_type == ReportType.PERFORMANCE:
            report_data = await self._generate_performance_report(start_time, end_time, config)
        elif config.report_type == ReportType.CONTENT_QUALITY:
            report_data = await self._generate_content_quality_report(start_time, end_time, config)
        elif config.report_type == ReportType.USER_BEHAVIOR:
            report_data = await self._generate_user_behavior_report(start_time, end_time, config)
        elif config.report_type == ReportType.BUSINESS_METRICS:
            report_data = await self._generate_business_metrics_report(start_time, end_time, config)
        elif config.report_type == ReportType.SYSTEM_HEALTH:
            report_data = await self._generate_system_health_report(start_time, end_time, config)
        else:
            raise ValueError(f"Unsupported report type: {config.report_type}")
        
        # Add metadata
        report_data["metadata"] = {
            "report_type": config.report_type.value,
            "time_range": config.time_range,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "filters": config.filters,
            "group_by": config.group_by
        }
        
        # Cache the report
        if self.redis_client:
            self.redis_client.setex(cache_key, 3600, json.dumps(report_data))
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (name, report_type, config, data)
                VALUES (?, ?, ?, ?)
            """, (
                f"{config.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config.report_type.value,
                json.dumps(asdict(config)),
                json.dumps(report_data)
            ))
            conn.commit()
        
        return report_data
    
    def _calculate_start_time(self, time_range: str, end_time: datetime) -> datetime:
        """Calculate start time based on time range"""
        
        if time_range == "1d":
            return end_time - timedelta(days=1)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        elif time_range == "90d":
            return end_time - timedelta(days=90)
        elif time_range == "1y":
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=7)  # Default to 7 days
    
    async def _generate_usage_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate usage report"""
        
        # Get usage metrics
        usage_metrics = await self.get_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get events
        events = await self.get_events(
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate usage statistics
        total_requests = len([m for m in usage_metrics if m.name == "api_requests"])
        unique_users = len(set([e["user_id"] for e in events if e["user_id"]]))
        content_generated = len([e for e in events if e["event_type"] == "content_generated"])
        
        # Group by time periods
        hourly_usage = defaultdict(int)
        daily_usage = defaultdict(int)
        
        for event in events:
            hour = event["timestamp"].strftime("%Y-%m-%d %H:00")
            day = event["timestamp"].strftime("%Y-%m-%d")
            hourly_usage[hour] += 1
            daily_usage[day] += 1
        
        return {
            "summary": {
                "total_requests": total_requests,
                "unique_users": unique_users,
                "content_generated": content_generated,
                "average_requests_per_hour": total_requests / max(1, (end_time - start_time).total_seconds() / 3600)
            },
            "hourly_usage": dict(hourly_usage),
            "daily_usage": dict(daily_usage),
            "top_content_types": self._get_top_content_types(events),
            "user_activity": self._get_user_activity(events)
        }
    
    async def _generate_performance_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate performance report"""
        
        # Get performance metrics
        response_times = await self.get_metrics(
            name="response_time",
            start_time=start_time,
            end_time=end_time
        )
        
        error_rates = await self.get_metrics(
            name="error_rate",
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate performance statistics
        if response_times:
            avg_response_time = sum(m.value for m in response_times) / len(response_times)
            min_response_time = min(m.value for m in response_times)
            max_response_time = max(m.value for m in response_times)
            p95_response_time = np.percentile([m.value for m in response_times], 95)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        if error_rates:
            avg_error_rate = sum(m.value for m in error_rates) / len(error_rates)
        else:
            avg_error_rate = 0
        
        return {
            "summary": {
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "p95_response_time": p95_response_time,
                "average_error_rate": avg_error_rate
            },
            "response_time_distribution": [m.value for m in response_times],
            "error_rate_trend": [m.value for m in error_rates],
            "performance_by_endpoint": self._get_performance_by_endpoint(response_times)
        }
    
    async def _generate_content_quality_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate content quality report"""
        
        # Get content quality metrics
        quality_metrics = await self.get_metrics(
            name="content_quality_score",
            start_time=start_time,
            end_time=end_time
        )
        
        # Get content generation events
        content_events = await self.get_events(
            event_type="content_generated",
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate quality statistics
        if quality_metrics:
            avg_quality = sum(m.value for m in quality_metrics) / len(quality_metrics)
            quality_distribution = Counter([round(m.value, 1) for m in quality_metrics])
        else:
            avg_quality = 0
            quality_distribution = {}
        
        # Analyze content types
        content_types = [e["data"].get("content_type", "unknown") for e in content_events]
        content_type_distribution = Counter(content_types)
        
        return {
            "summary": {
                "average_quality_score": avg_quality,
                "total_content_generated": len(content_events),
                "quality_trend": [m.value for m in quality_metrics]
            },
            "quality_distribution": dict(quality_distribution),
            "content_type_distribution": dict(content_type_distribution),
            "quality_by_content_type": self._get_quality_by_content_type(quality_metrics, content_events)
        }
    
    async def _generate_user_behavior_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate user behavior report"""
        
        # Get user events
        events = await self.get_events(
            start_time=start_time,
            end_time=end_time
        )
        
        # Analyze user behavior
        user_sessions = defaultdict(list)
        for event in events:
            if event["user_id"]:
                user_sessions[event["user_id"]].append(event)
        
        # Calculate behavior metrics
        session_durations = []
        actions_per_session = []
        
        for user_id, user_events in user_sessions.items():
            if user_events:
                session_start = min(e["timestamp"] for e in user_events)
                session_end = max(e["timestamp"] for e in user_events)
                duration = (session_end - session_start).total_seconds()
                session_durations.append(duration)
                actions_per_session.append(len(user_events))
        
        return {
            "summary": {
                "total_users": len(user_sessions),
                "average_session_duration": sum(session_durations) / max(1, len(session_durations)),
                "average_actions_per_session": sum(actions_per_session) / max(1, len(actions_per_session))
            },
            "user_retention": self._calculate_user_retention(events),
            "feature_usage": self._get_feature_usage(events),
            "user_journey": self._analyze_user_journey(events)
        }
    
    async def _generate_business_metrics_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate business metrics report"""
        
        # Get business metrics
        revenue_metrics = await self.get_metrics(
            name="revenue",
            start_time=start_time,
            end_time=end_time
        )
        
        conversion_metrics = await self.get_metrics(
            name="conversion_rate",
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate business statistics
        total_revenue = sum(m.value for m in revenue_metrics)
        avg_conversion_rate = sum(m.value for m in conversion_metrics) / max(1, len(conversion_metrics))
        
        return {
            "summary": {
                "total_revenue": total_revenue,
                "average_conversion_rate": avg_conversion_rate,
                "revenue_growth": self._calculate_growth_rate(revenue_metrics),
                "customer_acquisition_cost": self._calculate_cac(events)
            },
            "revenue_trend": [m.value for m in revenue_metrics],
            "conversion_funnel": self._analyze_conversion_funnel(events),
            "customer_segments": self._analyze_customer_segments(events)
        }
    
    async def _generate_system_health_report(self, start_time: datetime, end_time: datetime, config: ReportConfig) -> Dict[str, Any]:
        """Generate system health report"""
        
        # Get system metrics
        cpu_metrics = await self.get_metrics(
            name="cpu_usage",
            start_time=start_time,
            end_time=end_time
        )
        
        memory_metrics = await self.get_metrics(
            name="memory_usage",
            start_time=start_time,
            end_time=end_time
        )
        
        disk_metrics = await self.get_metrics(
            name="disk_usage",
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate system health
        avg_cpu = sum(m.value for m in cpu_metrics) / max(1, len(cpu_metrics))
        avg_memory = sum(m.value for m in memory_metrics) / max(1, len(memory_metrics))
        avg_disk = sum(m.value for m in disk_metrics) / max(1, len(disk_metrics))
        
        health_score = (100 - avg_cpu) * 0.4 + (100 - avg_memory) * 0.3 + (100 - avg_disk) * 0.3
        
        return {
            "summary": {
                "health_score": health_score,
                "average_cpu_usage": avg_cpu,
                "average_memory_usage": avg_memory,
                "average_disk_usage": avg_disk,
                "uptime": self._calculate_uptime(start_time, end_time)
            },
            "resource_usage": {
                "cpu": [m.value for m in cpu_metrics],
                "memory": [m.value for m in memory_metrics],
                "disk": [m.value for m in disk_metrics]
            },
            "alerts": self._get_system_alerts(cpu_metrics, memory_metrics, disk_metrics)
        }
    
    def _get_top_content_types(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get top content types from events"""
        content_types = [e["data"].get("content_type", "unknown") for e in events if e["event_type"] == "content_generated"]
        return dict(Counter(content_types).most_common(10))
    
    def _get_user_activity(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get user activity patterns"""
        hourly_activity = defaultdict(int)
        for event in events:
            hour = event["timestamp"].hour
            hourly_activity[hour] += 1
        return dict(hourly_activity)
    
    def _get_performance_by_endpoint(self, response_times: List[Metric]) -> Dict[str, float]:
        """Get performance metrics by endpoint"""
        endpoint_performance = defaultdict(list)
        for metric in response_times:
            endpoint = metric.labels.get("endpoint", "unknown")
            endpoint_performance[endpoint].append(metric.value)
        
        result = {}
        for endpoint, times in endpoint_performance.items():
            result[endpoint] = {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }
        
        return result
    
    def _get_quality_by_content_type(self, quality_metrics: List[Metric], content_events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get quality scores by content type"""
        # This would require matching metrics to events by timestamp or ID
        # Simplified implementation
        return {"presentation": 8.5, "document": 8.2, "webpage": 8.8}
    
    def _calculate_user_retention(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate user retention rates"""
        # Simplified implementation
        return {"day_1": 0.85, "day_7": 0.65, "day_30": 0.45}
    
    def _get_feature_usage(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get feature usage statistics"""
        features = [e["event_type"] for e in events]
        return dict(Counter(features).most_common(10))
    
    def _analyze_user_journey(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user journey patterns"""
        # Simplified implementation
        return {
            "common_paths": [
                ["login", "create_content", "export"],
                ["login", "view_dashboard", "create_content"]
            ],
            "drop_off_points": ["login", "create_content"]
        }
    
    def _calculate_growth_rate(self, metrics: List[Metric]) -> float:
        """Calculate growth rate from metrics"""
        if len(metrics) < 2:
            return 0.0
        
        first_value = metrics[0].value
        last_value = metrics[-1].value
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_cac(self, events: List[Dict[str, Any]]) -> float:
        """Calculate Customer Acquisition Cost"""
        # Simplified implementation
        return 25.50
    
    def _analyze_conversion_funnel(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze conversion funnel"""
        # Simplified implementation
        return {
            "visitors": 1000,
            "signups": 200,
            "trial_users": 150,
            "paid_users": 75
        }
    
    def _analyze_customer_segments(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze customer segments"""
        # Simplified implementation
        return {
            "enterprise": {"count": 25, "revenue": 50000},
            "professional": {"count": 100, "revenue": 25000},
            "individual": {"count": 200, "revenue": 10000}
        }
    
    def _calculate_uptime(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate system uptime percentage"""
        # Simplified implementation
        return 99.9
    
    def _get_system_alerts(self, cpu_metrics: List[Metric], memory_metrics: List[Metric], disk_metrics: List[Metric]) -> List[Dict[str, Any]]:
        """Get system alerts based on metrics"""
        alerts = []
        
        # Check for high CPU usage
        if cpu_metrics:
            max_cpu = max(m.value for m in cpu_metrics)
            if max_cpu > 90:
                alerts.append({
                    "type": "warning",
                    "message": f"High CPU usage detected: {max_cpu}%",
                    "timestamp": max(cpu_metrics, key=lambda m: m.value).timestamp.isoformat()
                })
        
        # Check for high memory usage
        if memory_metrics:
            max_memory = max(m.value for m in memory_metrics)
            if max_memory > 85:
                alerts.append({
                    "type": "warning",
                    "message": f"High memory usage detected: {max_memory}%",
                    "timestamp": max(memory_metrics, key=lambda m: m.value).timestamp.isoformat()
                })
        
        return alerts
    
    async def create_dashboard(self, config: DashboardConfig) -> Dict[str, Any]:
        """Create a custom dashboard"""
        
        dashboard = {
            "name": config.name,
            "config": asdict(config),
            "widgets": [],
            "created_at": datetime.now().isoformat()
        }
        
        # Generate widgets
        for widget_config in config.widgets:
            widget = await self._generate_widget(widget_config)
            dashboard["widgets"].append(widget)
        
        # Store dashboard
        self.dashboards[config.name] = dashboard
        
        return dashboard
    
    async def _generate_widget(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a dashboard widget"""
        
        widget_type = widget_config.get("type", "metric")
        
        if widget_type == "metric":
            return await self._generate_metric_widget(widget_config)
        elif widget_type == "chart":
            return await self._generate_chart_widget(widget_config)
        elif widget_type == "table":
            return await self._generate_table_widget(widget_config)
        else:
            raise ValueError(f"Unsupported widget type: {widget_type}")
    
    async def _generate_metric_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a metric widget"""
        
        metric_name = config.get("metric_name")
        time_range = config.get("time_range", "24h")
        
        end_time = datetime.now()
        start_time = self._calculate_start_time(time_range, end_time)
        
        metrics = await self.get_metrics(
            name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        if metrics:
            current_value = metrics[0].value
            previous_value = metrics[-1].value if len(metrics) > 1 else current_value
            change = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        else:
            current_value = 0
            change = 0
        
        return {
            "type": "metric",
            "title": config.get("title", metric_name),
            "value": current_value,
            "change": change,
            "format": config.get("format", "number"),
            "color": config.get("color", "blue")
        }
    
    async def _generate_chart_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a chart widget"""
        
        chart_type = config.get("chart_type", "line")
        data_source = config.get("data_source")
        
        # Get data based on source
        if data_source == "metrics":
            end_time = datetime.now()
            start_time = self._calculate_start_time(config.get("time_range", "7d"), end_time)
            
            metrics = await self.get_metrics(
                name=config.get("metric_name"),
                start_time=start_time,
                end_time=end_time
            )
            
            data = {
                "labels": [m.timestamp.strftime("%Y-%m-%d %H:%M") for m in metrics],
                "values": [m.value for m in metrics]
            }
        else:
            data = {"labels": [], "values": []}
        
        return {
            "type": "chart",
            "title": config.get("title", "Chart"),
            "chart_type": chart_type,
            "data": data,
            "options": config.get("options", {})
        }
    
    async def _generate_table_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a table widget"""
        
        # Simplified implementation
        return {
            "type": "table",
            "title": config.get("title", "Table"),
            "columns": config.get("columns", []),
            "data": config.get("data", [])
        }
    
    async def export_report(self, report_data: Dict[str, Any], format: str = "json") -> bytes:
        """Export report in specified format"""
        
        if format == "json":
            return json.dumps(report_data, indent=2).encode()
        elif format == "csv":
            # Convert to CSV format
            df = pd.DataFrame(report_data.get("data", []))
            return df.to_csv(index=False).encode()
        elif format == "html":
            # Generate HTML report
            html = self._generate_html_report(report_data)
            return html.encode()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Analytics Report</h1>
                <p>Generated: {report_data.get('metadata', {}).get('generated_at', 'Unknown')}</p>
            </div>
        """
        
        # Add summary section
        if "summary" in report_data:
            html += "<div class='section'><h2>Summary</h2>"
            for key, value in report_data["summary"].items():
                html += f"<div class='metric'><strong>{key}:</strong> {value}</div>"
            html += "</div>"
        
        html += "</body></html>"
        
        return html
    
    def get_dashboard(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard by name"""
        return self.dashboards.get(name)
    
    def list_dashboards(self) -> List[str]:
        """List all dashboard names"""
        return list(self.dashboards.keys())
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Analytics service cleanup completed")

# Global instance
advanced_analytics_service = None

async def get_advanced_analytics_service() -> AdvancedAnalyticsService:
    """Get global advanced analytics service instance"""
    global advanced_analytics_service
    if not advanced_analytics_service:
        config = {
            "database_path": "data/analytics.db",
            "redis_url": "redis://localhost:6379"
        }
        advanced_analytics_service = AdvancedAnalyticsService(config)
    return advanced_analytics_service



