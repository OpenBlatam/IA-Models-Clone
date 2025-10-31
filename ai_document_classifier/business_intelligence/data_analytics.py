"""
Advanced Business Intelligence and Data Analytics System
=====================================================

Comprehensive business intelligence system with advanced analytics,
reporting, and data visualization for document classification insights.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import statistics
import math

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Analytics types"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    TREND = "trend"

class MetricType(Enum):
    """Metric types"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    CONVERSION_RATE = "conversion_rate"
    RETENTION_RATE = "retention_rate"
    CHURN_RATE = "churn_rate"

class VisualizationType(Enum):
    """Visualization types"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    DASHBOARD = "dashboard"
    TABLE = "table"
    KPI_CARD = "kpi_card"
    FUNNEL_CHART = "funnel_chart"
    SANKEY_DIAGRAM = "sankey_diagram"

@dataclass
class DataPoint:
    """Single data point"""
    timestamp: datetime
    value: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Business metric"""
    id: str
    name: str
    description: str
    metric_type: MetricType
    data_points: List[DataPoint]
    calculated_value: float
    unit: str
    calculated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KPI:
    """Key Performance Indicator"""
    id: str
    name: str
    description: str
    current_value: float
    target_value: float
    unit: str
    status: str  # above_target, on_target, below_target
    trend: str  # improving, stable, declining
    calculated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Report:
    """Analytics report"""
    id: str
    title: str
    description: str
    report_type: AnalyticsType
    metrics: List[Metric]
    kpis: List[KPI]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dashboard:
    """Analytics dashboard"""
    id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    filters: Dict[str, Any]
    refresh_interval: int  # seconds
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedDataAnalytics:
    """
    Advanced business intelligence and data analytics system
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data analytics system
        
        Args:
            data_dir: Directory for analytics data
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.metrics: Dict[str, Metric] = {}
        self.kpis: Dict[str, KPI] = {}
        self.reports: Dict[str, Report] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Raw data
        self.raw_data: List[Dict[str, Any]] = []
        
        # Analytics configuration
        self.analytics_config = self._initialize_analytics_config()
        
        # Initialize default KPIs
        self._initialize_default_kpis()
        
        # Data processing
        self.data_processor = DataProcessor()
        self.insight_generator = InsightGenerator()
    
    def _initialize_analytics_config(self) -> Dict[str, Any]:
        """Initialize analytics configuration"""
        return {
            "default_periods": {
                "hourly": 24,
                "daily": 30,
                "weekly": 12,
                "monthly": 12,
                "yearly": 5
            },
            "kpi_thresholds": {
                "document_classification_accuracy": 95.0,
                "response_time_ms": 1000.0,
                "user_satisfaction": 4.0,
                "system_uptime": 99.9,
                "error_rate": 1.0
            },
            "alert_thresholds": {
                "performance_degradation": 20.0,  # percentage
                "error_spike": 50.0,  # percentage increase
                "usage_drop": 30.0,  # percentage decrease
                "accuracy_drop": 5.0  # percentage points
            }
        }
    
    def _initialize_default_kpis(self):
        """Initialize default KPIs"""
        default_kpis = [
            {
                "id": "document_classification_accuracy",
                "name": "Document Classification Accuracy",
                "description": "Percentage of correctly classified documents",
                "target_value": 95.0,
                "unit": "%"
            },
            {
                "id": "response_time",
                "name": "Average Response Time",
                "description": "Average time to classify a document",
                "target_value": 1000.0,
                "unit": "ms"
            },
            {
                "id": "user_satisfaction",
                "name": "User Satisfaction Score",
                "description": "Average user satisfaction rating",
                "target_value": 4.0,
                "unit": "/5"
            },
            {
                "id": "system_uptime",
                "name": "System Uptime",
                "description": "Percentage of time system is available",
                "target_value": 99.9,
                "unit": "%"
            },
            {
                "id": "error_rate",
                "name": "Error Rate",
                "description": "Percentage of failed requests",
                "target_value": 1.0,
                "unit": "%"
            },
            {
                "id": "daily_active_users",
                "name": "Daily Active Users",
                "description": "Number of unique users per day",
                "target_value": 1000.0,
                "unit": "users"
            },
            {
                "id": "documents_processed",
                "name": "Documents Processed",
                "description": "Total documents processed per day",
                "target_value": 10000.0,
                "unit": "documents"
            },
            {
                "id": "revenue_per_user",
                "name": "Revenue Per User",
                "description": "Average revenue generated per user",
                "target_value": 50.0,
                "unit": "$"
            }
        ]
        
        for kpi_data in default_kpis:
            kpi = KPI(
                id=kpi_data["id"],
                name=kpi_data["name"],
                description=kpi_data["description"],
                current_value=0.0,
                target_value=kpi_data["target_value"],
                unit=kpi_data["unit"],
                status="unknown",
                trend="unknown",
                calculated_at=datetime.now()
            )
            self.kpis[kpi.id] = kpi
    
    async def add_data_point(self, metric_name: str, value: float, category: str = "default", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a data point to analytics
        
        Args:
            metric_name: Name of the metric
            value: Value of the data point
            category: Category of the data point
            metadata: Additional metadata
            
        Returns:
            Data point ID
        """
        if metadata is None:
            metadata = {}
        
        data_point = DataPoint(
            timestamp=datetime.now(),
            value=value,
            category=category,
            metadata=metadata
        )
        
        # Add to raw data
        self.raw_data.append({
            "metric_name": metric_name,
            "timestamp": data_point.timestamp,
            "value": value,
            "category": category,
            "metadata": metadata
        })
        
        # Update metric if exists
        if metric_name in self.metrics:
            self.metrics[metric_name].data_points.append(data_point)
        else:
            # Create new metric
            metric = Metric(
                id=str(uuid.uuid4()),
                name=metric_name,
                description=f"Metric for {metric_name}",
                metric_type=MetricType.AVERAGE,
                data_points=[data_point],
                calculated_value=value,
                unit="",
                calculated_at=datetime.now()
            )
            self.metrics[metric_name] = metric
        
        logger.info(f"Added data point for {metric_name}: {value}")
        
        return str(uuid.uuid4())
    
    async def calculate_metric(self, metric_name: str, metric_type: MetricType, period_hours: int = 24) -> Metric:
        """
        Calculate metric for a specific period
        
        Args:
            metric_name: Name of the metric
            metric_type: Type of calculation
            period_hours: Period in hours
            
        Returns:
            Calculated metric
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric not found: {metric_name}")
        
        metric = self.metrics[metric_name]
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        # Filter data points for the period
        period_data = [
            dp for dp in metric.data_points
            if dp.timestamp >= cutoff_time
        ]
        
        if not period_data:
            raise ValueError(f"No data available for {metric_name} in the specified period")
        
        values = [dp.value for dp in period_data]
        
        # Calculate based on metric type
        if metric_type == MetricType.COUNT:
            calculated_value = len(values)
        elif metric_type == MetricType.SUM:
            calculated_value = sum(values)
        elif metric_type == MetricType.AVERAGE:
            calculated_value = statistics.mean(values)
        elif metric_type == MetricType.MEDIAN:
            calculated_value = statistics.median(values)
        elif metric_type == MetricType.PERCENTILE:
            calculated_value = np.percentile(values, 95)
        elif metric_type == MetricType.GROWTH_RATE:
            if len(values) >= 2:
                growth = (values[-1] - values[0]) / values[0] * 100
                calculated_value = growth
            else:
                calculated_value = 0.0
        else:
            calculated_value = statistics.mean(values)
        
        # Update metric
        metric.calculated_value = calculated_value
        metric.metric_type = metric_type
        metric.calculated_at = datetime.now()
        
        logger.info(f"Calculated {metric_name}: {calculated_value:.2f}")
        
        return metric
    
    async def update_kpi(self, kpi_id: str, current_value: float) -> KPI:
        """
        Update KPI with current value
        
        Args:
            kpi_id: KPI identifier
            current_value: Current value
            
        Returns:
            Updated KPI
        """
        if kpi_id not in self.kpis:
            raise ValueError(f"KPI not found: {kpi_id}")
        
        kpi = self.kpis[kpi_id]
        kpi.current_value = current_value
        kpi.calculated_at = datetime.now()
        
        # Determine status
        if current_value >= kpi.target_value:
            kpi.status = "above_target"
        elif current_value >= kpi.target_value * 0.9:
            kpi.status = "on_target"
        else:
            kpi.status = "below_target"
        
        # Calculate trend (simplified)
        kpi.trend = "stable"  # In practice, you'd compare with historical data
        
        logger.info(f"Updated KPI {kpi_id}: {current_value} ({kpi.status})")
        
        return kpi
    
    async def generate_report(self, report_type: AnalyticsType, period_hours: int = 24, include_insights: bool = True) -> Report:
        """
        Generate analytics report
        
        Args:
            report_type: Type of analytics report
            period_hours: Period in hours
            include_insights: Whether to include insights
            
        Returns:
            Generated report
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)
        
        # Calculate all metrics
        calculated_metrics = []
        for metric_name in self.metrics:
            try:
                metric = await self.calculate_metric(metric_name, MetricType.AVERAGE, period_hours)
                calculated_metrics.append(metric)
            except ValueError:
                continue
        
        # Update KPIs
        updated_kpis = []
        for kpi in self.kpis.values():
            # Simulate KPI calculation (in practice, you'd calculate from actual data)
            simulated_value = np.random.normal(kpi.target_value, kpi.target_value * 0.1)
            updated_kpi = await self.update_kpi(kpi.id, simulated_value)
            updated_kpis.append(updated_kpi)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(calculated_metrics, period_hours)
        
        # Generate insights and recommendations
        insights = []
        recommendations = []
        
        if include_insights:
            insights = await self.insight_generator.generate_insights(calculated_metrics, updated_kpis)
            recommendations = await self.insight_generator.generate_recommendations(calculated_metrics, updated_kpis)
        
        # Create report
        report = Report(
            id=str(uuid.uuid4()),
            title=f"{report_type.value.title()} Analytics Report",
            description=f"Analytics report for {period_hours} hours",
            report_type=report_type,
            metrics=calculated_metrics,
            kpis=updated_kpis,
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(),
            period_start=start_time,
            period_end=end_time,
            metadata={
                "period_hours": period_hours,
                "metrics_count": len(calculated_metrics),
                "kpis_count": len(updated_kpis)
            }
        )
        
        self.reports[report.id] = report
        
        logger.info(f"Generated {report_type.value} report with {len(calculated_metrics)} metrics")
        
        return report
    
    def _generate_visualizations(self, metrics: List[Metric], period_hours: int) -> List[Dict[str, Any]]:
        """Generate visualizations for metrics"""
        visualizations = []
        
        # KPI Cards
        kpi_cards = {
            "type": VisualizationType.KPI_CARD.value,
            "title": "Key Performance Indicators",
            "data": [
                {
                    "name": kpi.name,
                    "value": kpi.current_value,
                    "target": kpi.target_value,
                    "unit": kpi.unit,
                    "status": kpi.status,
                    "trend": kpi.trend
                }
                for kpi in self.kpis.values()
            ]
        }
        visualizations.append(kpi_cards)
        
        # Line chart for time series data
        if metrics:
            line_chart = {
                "type": VisualizationType.LINE_CHART.value,
                "title": "Metrics Over Time",
                "data": {
                    "labels": [dp.timestamp.isoformat() for dp in metrics[0].data_points[-10:]],
                    "datasets": [
                        {
                            "label": metric.name,
                            "data": [dp.value for dp in metric.data_points[-10:]],
                            "borderColor": f"hsl({i * 60}, 70%, 50%)"
                        }
                        for i, metric in enumerate(metrics[:5])  # Limit to 5 metrics
                    ]
                }
            }
            visualizations.append(line_chart)
        
        # Bar chart for categorical data
        if metrics:
            categories = Counter(dp.category for metric in metrics for dp in metric.data_points)
            bar_chart = {
                "type": VisualizationType.BAR_CHART.value,
                "title": "Data by Category",
                "data": {
                    "labels": list(categories.keys()),
                    "datasets": [{
                        "label": "Count",
                        "data": list(categories.values()),
                        "backgroundColor": "rgba(54, 162, 235, 0.6)"
                    }]
                }
            }
            visualizations.append(bar_chart)
        
        return visualizations
    
    async def create_dashboard(self, name: str, description: str, widgets: List[Dict[str, Any]], layout: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create analytics dashboard
        
        Args:
            name: Dashboard name
            description: Dashboard description
            widgets: List of widgets
            layout: Dashboard layout
            
        Returns:
            Created dashboard
        """
        if layout is None:
            layout = {"columns": 3, "rows": 4}
        
        dashboard = Dashboard(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            widgets=widgets,
            layout=layout,
            filters={},
            refresh_interval=300,  # 5 minutes
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.dashboards[dashboard.id] = dashboard
        
        logger.info(f"Created dashboard: {name}")
        
        return dashboard
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        total_metrics = len(self.metrics)
        total_kpis = len(self.kpis)
        total_reports = len(self.reports)
        total_dashboards = len(self.dashboards)
        
        # Calculate KPI status distribution
        kpi_statuses = Counter(kpi.status for kpi in self.kpis.values())
        
        # Calculate recent activity
        recent_data_points = len([
            dp for metric in self.metrics.values()
            for dp in metric.data_points
            if dp.timestamp >= datetime.now() - timedelta(hours=24)
        ])
        
        return {
            "total_metrics": total_metrics,
            "total_kpis": total_kpis,
            "total_reports": total_reports,
            "total_dashboards": total_dashboards,
            "kpi_status_distribution": dict(kpi_statuses),
            "recent_data_points_24h": recent_data_points,
            "data_points_total": sum(len(metric.data_points) for metric in self.metrics.values()),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_metric(self, metric_name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self.metrics.get(metric_name)
    
    def get_kpi(self, kpi_id: str) -> Optional[KPI]:
        """Get KPI by ID"""
        return self.kpis.get(kpi_id)
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get report by ID"""
        return self.reports.get(report_id)
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)

class DataProcessor:
    """Data processing utilities"""
    
    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate data"""
        cleaned_data = []
        
        for record in data:
            # Remove null values
            cleaned_record = {k: v for k, v in record.items() if v is not None}
            
            # Validate required fields
            if all(key in cleaned_record for key in ["timestamp", "value"]):
                cleaned_data.append(cleaned_record)
        
        return cleaned_data
    
    def aggregate_data(self, data: List[Dict[str, Any]], group_by: str, aggregation: str) -> Dict[str, float]:
        """Aggregate data by field"""
        grouped_data = defaultdict(list)
        
        for record in data:
            if group_by in record:
                grouped_data[record[group_by]].append(record["value"])
        
        result = {}
        for key, values in grouped_data.items():
            if aggregation == "sum":
                result[key] = sum(values)
            elif aggregation == "average":
                result[key] = statistics.mean(values)
            elif aggregation == "count":
                result[key] = len(values)
            elif aggregation == "max":
                result[key] = max(values)
            elif aggregation == "min":
                result[key] = min(values)
        
        return result

class InsightGenerator:
    """Generate insights and recommendations"""
    
    async def generate_insights(self, metrics: List[Metric], kpis: List[KPI]) -> List[str]:
        """Generate insights from metrics and KPIs"""
        insights = []
        
        # KPI insights
        for kpi in kpis:
            if kpi.status == "below_target":
                insights.append(f"{kpi.name} is {((kpi.target_value - kpi.current_value) / kpi.target_value * 100):.1f}% below target")
            elif kpi.status == "above_target":
                insights.append(f"{kpi.name} is {((kpi.current_value - kpi.target_value) / kpi.target_value * 100):.1f}% above target")
        
        # Metric insights
        for metric in metrics:
            if metric.calculated_value > 0:
                insights.append(f"{metric.name} shows a value of {metric.calculated_value:.2f}")
        
        return insights
    
    async def generate_recommendations(self, metrics: List[Metric], kpis: List[KPI]) -> List[str]:
        """Generate recommendations from metrics and KPIs"""
        recommendations = []
        
        # KPI recommendations
        for kpi in kpis:
            if kpi.status == "below_target":
                recommendations.append(f"Focus on improving {kpi.name} to meet target of {kpi.target_value}")
            elif kpi.status == "above_target":
                recommendations.append(f"Maintain current performance for {kpi.name}")
        
        # General recommendations
        recommendations.extend([
            "Monitor key metrics regularly",
            "Set up automated alerts for critical KPIs",
            "Review and update targets quarterly",
            "Implement data-driven decision making"
        ])
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize analytics system
    analytics = AdvancedDataAnalytics()
    
    # Add sample data
    for i in range(100):
        await analytics.add_data_point("document_classifications", np.random.normal(100, 20), "daily")
        await analytics.add_data_point("response_time", np.random.normal(500, 100), "performance")
        await analytics.add_data_point("user_satisfaction", np.random.normal(4.2, 0.5), "feedback")
    
    # Calculate metrics
    accuracy_metric = await analytics.calculate_metric("document_classifications", MetricType.AVERAGE, 24)
    response_metric = await analytics.calculate_metric("response_time", MetricType.AVERAGE, 24)
    
    print("Calculated Metrics:")
    print(f"Document Classifications: {accuracy_metric.calculated_value:.2f}")
    print(f"Response Time: {response_metric.calculated_value:.2f}ms")
    
    # Update KPIs
    await analytics.update_kpi("document_classification_accuracy", 96.5)
    await analytics.update_kpi("response_time", 450.0)
    await analytics.update_kpi("user_satisfaction", 4.3)
    
    # Generate report
    report = await analytics.generate_report(AnalyticsType.DESCRIPTIVE, 24)
    
    print(f"\nGenerated Report:")
    print(f"Title: {report.title}")
    print(f"Metrics: {len(report.metrics)}")
    print(f"KPIs: {len(report.kpis)}")
    print(f"Insights: {len(report.insights)}")
    print(f"Recommendations: {len(report.recommendations)}")
    
    # Create dashboard
    widgets = [
        {"type": "kpi_card", "title": "Key Metrics"},
        {"type": "line_chart", "title": "Trends"},
        {"type": "bar_chart", "title": "Categories"}
    ]
    
    dashboard = await analytics.create_dashboard(
        "Main Dashboard",
        "Primary analytics dashboard",
        widgets
    )
    
    print(f"\nCreated Dashboard: {dashboard.name}")
    print(f"Widgets: {len(dashboard.widgets)}")
    
    # Get summary
    summary = await analytics.get_analytics_summary()
    print(f"\nAnalytics Summary:")
    print(f"Total Metrics: {summary['total_metrics']}")
    print(f"Total KPIs: {summary['total_kpis']}")
    print(f"Total Reports: {summary['total_reports']}")
    print(f"Recent Data Points (24h): {summary['recent_data_points_24h']}")
    
    print("\nAdvanced Data Analytics initialized successfully")

























