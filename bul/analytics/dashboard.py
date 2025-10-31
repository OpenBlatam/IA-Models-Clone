"""
Analytics Dashboard System for BUL
Provides comprehensive analytics, insights, and reporting capabilities
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json
from datetime import datetime, timedelta
import statistics
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class TimeRange(str, Enum):
    """Time range options"""
    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    LAST_QUARTER = "90d"
    LAST_YEAR = "365d"
    CUSTOM = "custom"


class ChartType(str, Enum):
    """Chart types"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    TABLE = "table"
    METRIC = "metric"


@dataclass
class MetricDataPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class ChartData:
    """Chart data structure"""
    chart_type: ChartType
    title: str
    description: str
    data: List[Dict[str, Any]]
    x_axis: str
    y_axis: str
    labels: List[str]
    colors: List[str]
    metadata: Dict[str, Any]


class DashboardWidget(BaseModel):
    """Dashboard widget definition"""
    id: str = Field(..., description="Widget ID")
    title: str = Field(..., description="Widget title")
    description: str = Field(..., description="Widget description")
    chart_type: ChartType = Field(..., description="Chart type")
    metric_name: str = Field(..., description="Metric name")
    time_range: TimeRange = Field(..., description="Time range")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters")
    position: Dict[str, int] = Field(..., description="Widget position")
    size: Dict[str, int] = Field(..., description="Widget size")
    refresh_interval: int = Field(default=300, description="Refresh interval in seconds")
    is_visible: bool = Field(default=True, description="Is widget visible")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Dashboard(BaseModel):
    """Dashboard definition"""
    id: str = Field(..., description="Dashboard ID")
    name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    widgets: List[DashboardWidget] = Field(..., description="Dashboard widgets")
    layout: Dict[str, Any] = Field(default_factory=dict, description="Dashboard layout")
    is_public: bool = Field(default=False, description="Is dashboard public")
    created_by: str = Field(..., description="Creator user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list, description="Dashboard tags")


class AnalyticsQuery(BaseModel):
    """Analytics query definition"""
    metric_name: str = Field(..., description="Metric name")
    time_range: TimeRange = Field(..., description="Time range")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    aggregation: str = Field(default="sum", description="Aggregation function")
    group_by: List[str] = Field(default_factory=list, description="Group by fields")
    limit: int = Field(default=1000, description="Result limit")


class AnalyticsInsight(BaseModel):
    """Analytics insight definition"""
    id: str = Field(..., description="Insight ID")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Insight description")
    type: str = Field(..., description="Insight type")
    severity: str = Field(..., description="Insight severity")
    metric_name: str = Field(..., description="Related metric")
    value: float = Field(..., description="Insight value")
    threshold: float = Field(..., description="Threshold value")
    recommendation: str = Field(..., description="Recommendation")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_resolved: bool = Field(default=False, description="Is insight resolved")


class AnalyticsDashboard:
    """Advanced Analytics Dashboard System"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        self.dashboards: Dict[str, Dashboard] = {}
        self.insights: List[AnalyticsInsight] = []
        self._initialize_default_dashboards()
        self._initialize_sample_metrics()
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboards"""
        default_dashboards = [
            self._create_overview_dashboard(),
            self._create_user_analytics_dashboard(),
            self._create_model_performance_dashboard(),
            self._create_system_health_dashboard(),
            self._create_business_metrics_dashboard(),
        ]
        
        for dashboard in default_dashboards:
            self.dashboards[dashboard.id] = dashboard
        
        logger.info(f"Initialized {len(default_dashboards)} default dashboards")
    
    def _create_overview_dashboard(self) -> Dashboard:
        """Create overview dashboard"""
        return Dashboard(
            id="overview",
            name="System Overview",
            description="High-level system metrics and KPIs",
            widgets=[
                DashboardWidget(
                    id="total_users",
                    title="Total Users",
                    description="Total number of registered users",
                    chart_type=ChartType.METRIC,
                    metric_name="users_total",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 0},
                    size={"width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="documents_generated",
                    title="Documents Generated",
                    description="Total documents generated this month",
                    chart_type=ChartType.METRIC,
                    metric_name="documents_generated_total",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 3, "y": 0},
                    size={"width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="api_requests",
                    title="API Requests",
                    description="Total API requests processed",
                    chart_type=ChartType.METRIC,
                    metric_name="api_requests_total",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 6, "y": 0},
                    size={"width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="revenue",
                    title="Revenue",
                    description="Total revenue generated",
                    chart_type=ChartType.METRIC,
                    metric_name="revenue_total",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 9, "y": 0},
                    size={"width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="usage_trend",
                    title="Usage Trend",
                    description="Daily usage trend over time",
                    chart_type=ChartType.LINE,
                    metric_name="daily_usage",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 2},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="document_types",
                    title="Document Types",
                    description="Distribution of document types",
                    chart_type=ChartType.PIE,
                    metric_name="document_types_distribution",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 6, "y": 2},
                    size={"width": 6, "height": 4}
                )
            ],
            created_by="system",
            tags=["overview", "kpi", "system"]
        )
    
    def _create_user_analytics_dashboard(self) -> Dashboard:
        """Create user analytics dashboard"""
        return Dashboard(
            id="user_analytics",
            name="User Analytics",
            description="User behavior and engagement metrics",
            widgets=[
                DashboardWidget(
                    id="active_users",
                    title="Active Users",
                    description="Daily and monthly active users",
                    chart_type=ChartType.LINE,
                    metric_name="active_users",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="user_retention",
                    title="User Retention",
                    description="User retention rates by cohort",
                    chart_type=ChartType.BAR,
                    metric_name="user_retention",
                    time_range=TimeRange.LAST_QUARTER,
                    position={"x": 6, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="user_segments",
                    title="User Segments",
                    description="User distribution by segments",
                    chart_type=ChartType.PIE,
                    metric_name="user_segments",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 4},
                    size={"width": 4, "height": 4}
                ),
                DashboardWidget(
                    id="feature_usage",
                    title="Feature Usage",
                    description="Most used features",
                    chart_type=ChartType.BAR,
                    metric_name="feature_usage",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 4, "y": 4},
                    size={"width": 8, "height": 4}
                )
            ],
            created_by="system",
            tags=["users", "analytics", "engagement"]
        )
    
    def _create_model_performance_dashboard(self) -> Dashboard:
        """Create model performance dashboard"""
        return Dashboard(
            id="model_performance",
            name="Model Performance",
            description="AI model performance and usage metrics",
            widgets=[
                DashboardWidget(
                    id="model_usage",
                    title="Model Usage",
                    description="Usage distribution across models",
                    chart_type=ChartType.PIE,
                    metric_name="model_usage_distribution",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="response_times",
                    title="Response Times",
                    description="Average response times by model",
                    chart_type=ChartType.BAR,
                    metric_name="model_response_times",
                    time_range=TimeRange.LAST_WEEK,
                    position={"x": 6, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="cost_analysis",
                    title="Cost Analysis",
                    description="Cost per model over time",
                    chart_type=ChartType.LINE,
                    metric_name="model_costs",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 0, "y": 4},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="quality_scores",
                    title="Quality Scores",
                    description="Quality scores by model",
                    chart_type=ChartType.BAR,
                    metric_name="model_quality_scores",
                    time_range=TimeRange.LAST_WEEK,
                    position={"x": 6, "y": 4},
                    size={"width": 6, "height": 4}
                )
            ],
            created_by="system",
            tags=["models", "performance", "ai"]
        )
    
    def _create_system_health_dashboard(self) -> Dashboard:
        """Create system health dashboard"""
        return Dashboard(
            id="system_health",
            name="System Health",
            description="System performance and health metrics",
            widgets=[
                DashboardWidget(
                    id="api_health",
                    title="API Health",
                    description="API response times and error rates",
                    chart_type=ChartType.LINE,
                    metric_name="api_health",
                    time_range=TimeRange.LAST_DAY,
                    position={"x": 0, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="error_rates",
                    title="Error Rates",
                    description="Error rates by endpoint",
                    chart_type=ChartType.BAR,
                    metric_name="error_rates",
                    time_range=TimeRange.LAST_DAY,
                    position={"x": 6, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="resource_usage",
                    title="Resource Usage",
                    description="CPU, memory, and disk usage",
                    chart_type=ChartType.AREA,
                    metric_name="resource_usage",
                    time_range=TimeRange.LAST_DAY,
                    position={"x": 0, "y": 4},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="database_performance",
                    title="Database Performance",
                    description="Database query performance",
                    chart_type=ChartType.LINE,
                    metric_name="database_performance",
                    time_range=TimeRange.LAST_DAY,
                    position={"x": 6, "y": 4},
                    size={"width": 6, "height": 4}
                )
            ],
            created_by="system",
            tags=["system", "health", "performance"]
        )
    
    def _create_business_metrics_dashboard(self) -> Dashboard:
        """Create business metrics dashboard"""
        return Dashboard(
            id="business_metrics",
            name="Business Metrics",
            description="Business KPIs and financial metrics",
            widgets=[
                DashboardWidget(
                    id="revenue_trend",
                    title="Revenue Trend",
                    description="Monthly revenue trend",
                    chart_type=ChartType.LINE,
                    metric_name="revenue_trend",
                    time_range=TimeRange.LAST_QUARTER,
                    position={"x": 0, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="subscription_metrics",
                    title="Subscription Metrics",
                    description="Subscription tiers and churn",
                    chart_type=ChartType.BAR,
                    metric_name="subscription_metrics",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 6, "y": 0},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="customer_lifetime_value",
                    title="Customer Lifetime Value",
                    description="CLV by customer segment",
                    chart_type=ChartType.BAR,
                    metric_name="customer_lifetime_value",
                    time_range=TimeRange.LAST_QUARTER,
                    position={"x": 0, "y": 4},
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="conversion_funnel",
                    title="Conversion Funnel",
                    description="User conversion funnel",
                    chart_type=ChartType.BAR,
                    metric_name="conversion_funnel",
                    time_range=TimeRange.LAST_MONTH,
                    position={"x": 6, "y": 4},
                    size={"width": 6, "height": 4}
                )
            ],
            created_by="system",
            tags=["business", "kpi", "financial"]
        )
    
    def _initialize_sample_metrics(self):
        """Initialize sample metrics for demonstration"""
        now = datetime.utcnow()
        
        # Generate sample data for the last 30 days
        for i in range(30):
            timestamp = now - timedelta(days=i)
            
            # Users metrics
            self.metrics["users_total"].append(MetricDataPoint(
                timestamp=timestamp,
                value=1000 + i * 10 + (i % 7) * 5,
                labels={"type": "total"},
                metadata={}
            ))
            
            # Documents generated
            self.metrics["documents_generated_total"].append(MetricDataPoint(
                timestamp=timestamp,
                value=500 + i * 20 + (i % 5) * 15,
                labels={"type": "total"},
                metadata={}
            ))
            
            # API requests
            self.metrics["api_requests_total"].append(MetricDataPoint(
                timestamp=timestamp,
                value=10000 + i * 100 + (i % 3) * 50,
                labels={"type": "total"},
                metadata={}
            ))
            
            # Revenue
            self.metrics["revenue_total"].append(MetricDataPoint(
                timestamp=timestamp,
                value=5000 + i * 200 + (i % 4) * 100,
                labels={"type": "total", "currency": "USD"},
                metadata={}
            ))
            
            # Daily usage
            self.metrics["daily_usage"].append(MetricDataPoint(
                timestamp=timestamp,
                value=100 + i * 5 + (i % 6) * 10,
                labels={"type": "daily"},
                metadata={}
            ))
        
        logger.info("Initialized sample metrics data")
    
    async def add_metric(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
        timestamp: datetime = None
    ):
        """Add a metric data point"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        self.metrics[metric_name].append(data_point)
        
        # Keep only last 1000 data points per metric
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
        
        # Generate insights
        await self._generate_insights(metric_name, data_point)
    
    async def query_metrics(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query metrics based on criteria"""
        metric_data = self.metrics.get(query.metric_name, [])
        
        # Filter by time range
        end_time = datetime.utcnow()
        if query.time_range == TimeRange.LAST_HOUR:
            start_time = end_time - timedelta(hours=1)
        elif query.time_range == TimeRange.LAST_DAY:
            start_time = end_time - timedelta(days=1)
        elif query.time_range == TimeRange.LAST_WEEK:
            start_time = end_time - timedelta(weeks=1)
        elif query.time_range == TimeRange.LAST_MONTH:
            start_time = end_time - timedelta(days=30)
        elif query.time_range == TimeRange.LAST_QUARTER:
            start_time = end_time - timedelta(days=90)
        elif query.time_range == TimeRange.LAST_YEAR:
            start_time = end_time - timedelta(days=365)
        else:
            start_time = end_time - timedelta(days=30)  # Default
        
        filtered_data = [
            dp for dp in metric_data
            if start_time <= dp.timestamp <= end_time
        ]
        
        # Apply filters
        if query.filters:
            filtered_data = [
                dp for dp in filtered_data
                if all(dp.labels.get(k) == v for k, v in query.filters.items())
            ]
        
        # Apply aggregation
        if query.aggregation == "sum":
            result = sum(dp.value for dp in filtered_data)
        elif query.aggregation == "avg":
            result = statistics.mean(dp.value for dp in filtered_data) if filtered_data else 0
        elif query.aggregation == "min":
            result = min(dp.value for dp in filtered_data) if filtered_data else 0
        elif query.aggregation == "max":
            result = max(dp.value for dp in filtered_data) if filtered_data else 0
        elif query.aggregation == "count":
            result = len(filtered_data)
        else:
            result = sum(dp.value for dp in filtered_data)
        
        # Group by fields
        if query.group_by:
            grouped_data = defaultdict(list)
            for dp in filtered_data:
                key = tuple(dp.labels.get(field, "") for field in query.group_by)
                grouped_data[key].append(dp.value)
            
            return [
                {
                    "group": dict(zip(query.group_by, key)),
                    "value": self._apply_aggregation(grouped_data[key], query.aggregation),
                    "count": len(grouped_data[key])
                }
                for key in grouped_data
            ]
        
        return [{"value": result, "count": len(filtered_data)}]
    
    def _apply_aggregation(self, values: List[float], aggregation: str) -> float:
        """Apply aggregation function to values"""
        if not values:
            return 0
        
        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            return sum(values)
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data with widget data"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        widget_data = {}
        
        for widget in dashboard.widgets:
            if not widget.is_visible:
                continue
            
            query = AnalyticsQuery(
                metric_name=widget.metric_name,
                time_range=widget.time_range,
                filters=widget.filters
            )
            
            try:
                data = await self.query_metrics(query)
                widget_data[widget.id] = {
                    "widget": widget.dict(),
                    "data": data,
                    "chart_data": self._format_chart_data(widget, data)
                }
            except Exception as e:
                logger.error(f"Error getting data for widget {widget.id}: {e}")
                widget_data[widget.id] = {
                    "widget": widget.dict(),
                    "data": [],
                    "error": str(e)
                }
        
        return {
            "dashboard": dashboard.dict(),
            "widgets": widget_data,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _format_chart_data(self, widget: DashboardWidget, data: List[Dict[str, Any]]) -> ChartData:
        """Format data for chart display"""
        if widget.chart_type == ChartType.METRIC:
            value = data[0]["value"] if data else 0
            return ChartData(
                chart_type=widget.chart_type,
                title=widget.title,
                description=widget.description,
                data=[{"value": value}],
                x_axis="",
                y_axis="Value",
                labels=[],
                colors=["#3B82F6"],
                metadata={}
            )
        
        elif widget.chart_type == ChartType.LINE:
            return ChartData(
                chart_type=widget.chart_type,
                title=widget.title,
                description=widget.description,
                data=data,
                x_axis="Time",
                y_axis="Value",
                labels=[str(i) for i in range(len(data))],
                colors=["#3B82F6"],
                metadata={}
            )
        
        elif widget.chart_type == ChartType.PIE:
            return ChartData(
                chart_type=widget.chart_type,
                title=widget.title,
                description=widget.description,
                data=data,
                x_axis="",
                y_axis="",
                labels=[item.get("group", {}).get("type", f"Item {i}") for i, item in enumerate(data)],
                colors=["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"],
                metadata={}
            )
        
        elif widget.chart_type == ChartType.BAR:
            return ChartData(
                chart_type=widget.chart_type,
                title=widget.title,
                description=widget.description,
                data=data,
                x_axis="Category",
                y_axis="Value",
                labels=[item.get("group", {}).get("type", f"Item {i}") for i, item in enumerate(data)],
                colors=["#3B82F6"],
                metadata={}
            )
        
        else:
            return ChartData(
                chart_type=widget.chart_type,
                title=widget.title,
                description=widget.description,
                data=data,
                x_axis="X",
                y_axis="Y",
                labels=[],
                colors=["#3B82F6"],
                metadata={}
            )
    
    async def _generate_insights(self, metric_name: str, data_point: MetricDataPoint):
        """Generate insights from metric data"""
        # Simple insight generation logic
        recent_data = self.metrics[metric_name][-10:]  # Last 10 data points
        
        if len(recent_data) < 5:
            return
        
        # Check for trends
        values = [dp.value for dp in recent_data]
        if len(values) >= 5:
            # Calculate trend
            trend = (values[-1] - values[0]) / len(values)
            
            if abs(trend) > statistics.stdev(values) * 0.5:  # Significant trend
                insight = AnalyticsInsight(
                    id=f"trend_{metric_name}_{datetime.utcnow().timestamp()}",
                    title=f"{metric_name.title()} Trend Alert",
                    description=f"Significant {'increase' if trend > 0 else 'decrease'} detected in {metric_name}",
                    type="trend",
                    severity="medium" if abs(trend) < statistics.stdev(values) else "high",
                    metric_name=metric_name,
                    value=values[-1],
                    threshold=statistics.mean(values),
                    recommendation=f"Monitor {metric_name} closely for continued {'growth' if trend > 0 else 'decline'}"
                )
                
                self.insights.append(insight)
        
        # Check for anomalies
        if len(values) >= 3:
            mean_val = statistics.mean(values[:-1])  # Mean excluding last point
            std_val = statistics.stdev(values[:-1])
            
            if std_val > 0 and abs(values[-1] - mean_val) > 2 * std_val:  # 2-sigma anomaly
                insight = AnalyticsInsight(
                    id=f"anomaly_{metric_name}_{datetime.utcnow().timestamp()}",
                    title=f"{metric_name.title()} Anomaly Detected",
                    description=f"Unusual value detected in {metric_name}: {values[-1]:.2f}",
                    type="anomaly",
                    severity="high",
                    metric_name=metric_name,
                    value=values[-1],
                    threshold=mean_val + 2 * std_val,
                    recommendation=f"Investigate the cause of the unusual {metric_name} value"
                )
                
                self.insights.append(insight)
    
    async def get_insights(
        self,
        metric_name: Optional[str] = None,
        severity: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        limit: int = 50
    ) -> List[AnalyticsInsight]:
        """Get analytics insights"""
        insights = self.insights
        
        if metric_name:
            insights = [i for i in insights if i.metric_name == metric_name]
        
        if severity:
            insights = [i for i in insights if i.severity == severity]
        
        if is_resolved is not None:
            insights = [i for i in insights if i.is_resolved == is_resolved]
        
        # Sort by creation time (newest first)
        insights.sort(key=lambda x: x.created_at, reverse=True)
        
        return insights[:limit]
    
    async def create_dashboard(self, dashboard_data: Dict[str, Any]) -> Dashboard:
        """Create a new dashboard"""
        dashboard = Dashboard(**dashboard_data)
        self.dashboards[dashboard.id] = dashboard
        
        logger.info(f"Created dashboard {dashboard.id}")
        return dashboard
    
    async def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> Optional[Dashboard]:
        """Update an existing dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        for key, value in updates.items():
            if hasattr(dashboard, key):
                setattr(dashboard, key, value)
        
        dashboard.updated_at = datetime.utcnow()
        logger.info(f"Updated dashboard {dashboard_id}")
        
        return dashboard
    
    async def list_dashboards(
        self,
        created_by: Optional[str] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dashboard]:
        """List dashboards with optional filtering"""
        dashboards = list(self.dashboards.values())
        
        if created_by:
            dashboards = [d for d in dashboards if d.created_by == created_by]
        
        if is_public is not None:
            dashboards = [d for d in dashboards if d.is_public == is_public]
        
        if tags:
            dashboards = [d for d in dashboards if any(tag in d.tags for tag in tags)]
        
        return dashboards
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        now = datetime.utcnow()
        last_month = now - timedelta(days=30)
        
        # Calculate key metrics
        total_users = len([dp for dp in self.metrics["users_total"] if dp.timestamp >= last_month])
        total_documents = sum(dp.value for dp in self.metrics["documents_generated_total"] if dp.timestamp >= last_month)
        total_requests = sum(dp.value for dp in self.metrics["api_requests_total"] if dp.timestamp >= last_month)
        total_revenue = sum(dp.value for dp in self.metrics["revenue_total"] if dp.timestamp >= last_month)
        
        # Get recent insights
        recent_insights = await self.get_insights(limit=10)
        
        # Calculate growth rates
        current_month_users = sum(dp.value for dp in self.metrics["users_total"][-30:])
        previous_month_users = sum(dp.value for dp in self.metrics["users_total"][-60:-30]) if len(self.metrics["users_total"]) >= 60 else current_month_users
        user_growth_rate = ((current_month_users - previous_month_users) / previous_month_users * 100) if previous_month_users > 0 else 0
        
        return {
            "summary": {
                "total_users": total_users,
                "total_documents": total_documents,
                "total_requests": total_requests,
                "total_revenue": total_revenue,
                "user_growth_rate": user_growth_rate
            },
            "insights": [
                {
                    "id": i.id,
                    "title": i.title,
                    "description": i.description,
                    "severity": i.severity,
                    "created_at": i.created_at.isoformat()
                }
                for i in recent_insights
            ],
            "dashboards": [
                {
                    "id": d.id,
                    "name": d.name,
                    "description": d.description,
                    "widget_count": len(d.widgets)
                }
                for d in self.dashboards.values()
            ],
            "generated_at": now.isoformat()
        }


# Global analytics dashboard instance
analytics_dashboard = AnalyticsDashboard()














