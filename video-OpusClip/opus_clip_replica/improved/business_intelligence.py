"""
Advanced Business Intelligence System for OpusClip Improved
========================================================

Comprehensive BI system with analytics, reporting, and business insights.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
import redis

from .schemas import get_settings
from .exceptions import BusinessIntelligenceError, create_bi_error

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Report types"""
    PERFORMANCE = "performance"
    FINANCIAL = "financial"
    USER_ANALYTICS = "user_analytics"
    CONTENT_ANALYTICS = "content_analytics"
    OPERATIONAL = "operational"
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    PERCENTILE = "percentile"


class DashboardType(str, Enum):
    """Dashboard types"""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    REAL_TIME = "real_time"
    CUSTOM = "custom"


@dataclass
class BusinessMetric:
    """Business metric definition"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    category: str
    unit: str
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    calculation_formula: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class ReportDefinition:
    """Report definition"""
    report_id: str
    name: str
    description: str
    report_type: ReportType
    metrics: List[str]  # Metric IDs
    filters: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression
    recipients: List[str] = None
    format: str = "pdf"  # pdf, excel, csv, json
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class DashboardDefinition:
    """Dashboard definition"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    filters: Dict[str, Any]
    refresh_interval: int = 300  # seconds
    created_at: datetime = None
    updated_at: datetime = None


class DataCollector:
    """Data collection and aggregation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.Redis(
            host=self.settings.redis.host,
            port=self.settings.redis.port,
            password=self.settings.redis.password,
            db=self.settings.redis.db
        )
    
    async def collect_user_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect user-related metrics"""
        try:
            start_time, end_time = time_range
            
            # Simulate data collection from database
            # In practice, this would query the actual database
            
            metrics = {
                "total_users": 1250,
                "active_users": 890,
                "new_users": 45,
                "churned_users": 12,
                "user_engagement_rate": 0.72,
                "average_session_duration": 18.5,  # minutes
                "user_retention_rate": 0.68,
                "user_satisfaction_score": 4.2,
                "support_tickets": 23,
                "user_feedback_score": 4.1
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"User metrics collection failed: {e}")
            raise create_bi_error("user_metrics_collection", "users", e)
    
    async def collect_content_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect content-related metrics"""
        try:
            start_time, end_time = time_range
            
            # Simulate data collection
            metrics = {
                "total_videos_processed": 1250,
                "total_clips_generated": 3750,
                "average_processing_time": 45.2,  # seconds
                "success_rate": 0.96,
                "error_rate": 0.04,
                "content_quality_score": 0.82,
                "viral_potential_score": 0.68,
                "platform_distribution": {
                    "youtube": 450,
                    "tiktok": 380,
                    "instagram": 320,
                    "linkedin": 200,
                    "twitter": 150
                },
                "content_categories": {
                    "entertainment": 520,
                    "education": 380,
                    "business": 200,
                    "lifestyle": 150
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Content metrics collection failed: {e}")
            raise create_bi_error("content_metrics_collection", "content", e)
    
    async def collect_financial_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect financial metrics"""
        try:
            start_time, end_time = time_range
            
            # Simulate financial data collection
            metrics = {
                "revenue": 125000.0,  # USD
                "costs": 85000.0,
                "profit": 40000.0,
                "profit_margin": 0.32,
                "revenue_growth": 0.15,  # 15% growth
                "customer_acquisition_cost": 25.0,
                "customer_lifetime_value": 180.0,
                "monthly_recurring_revenue": 45000.0,
                "churn_rate": 0.05,  # 5% monthly churn
                "average_revenue_per_user": 100.0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Financial metrics collection failed: {e}")
            raise create_bi_error("financial_metrics_collection", "financial", e)
    
    async def collect_operational_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect operational metrics"""
        try:
            start_time, end_time = time_range
            
            # Simulate operational data collection
            metrics = {
                "system_uptime": 0.999,  # 99.9%
                "average_response_time": 1.2,  # seconds
                "throughput": 150,  # requests per second
                "error_rate": 0.001,  # 0.1%
                "cpu_utilization": 0.65,
                "memory_utilization": 0.72,
                "disk_utilization": 0.45,
                "network_utilization": 0.38,
                "api_calls_per_minute": 2500,
                "cache_hit_rate": 0.85
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Operational metrics collection failed: {e}")
            raise create_bi_error("operational_metrics_collection", "operations", e)


class AnalyticsEngine:
    """Advanced analytics and insights generation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_collector = DataCollector()
    
    async def generate_performance_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance insights"""
        try:
            insights = {
                "overall_performance": "excellent",
                "key_insights": [],
                "recommendations": [],
                "trends": [],
                "alerts": []
            }
            
            # Analyze user metrics
            if "user_metrics" in metrics:
                user_metrics = metrics["user_metrics"]
                
                if user_metrics.get("user_engagement_rate", 0) > 0.7:
                    insights["key_insights"].append("High user engagement indicates strong product-market fit")
                else:
                    insights["recommendations"].append("Focus on improving user engagement through better UX")
                
                if user_metrics.get("user_retention_rate", 0) > 0.6:
                    insights["key_insights"].append("Strong user retention demonstrates product value")
                else:
                    insights["recommendations"].append("Implement retention strategies to reduce churn")
            
            # Analyze content metrics
            if "content_metrics" in metrics:
                content_metrics = metrics["content_metrics"]
                
                if content_metrics.get("success_rate", 0) > 0.95:
                    insights["key_insights"].append("High processing success rate ensures reliable service")
                else:
                    insights["alerts"].append("Processing error rate is above acceptable threshold")
                
                if content_metrics.get("viral_potential_score", 0) > 0.7:
                    insights["key_insights"].append("Content shows strong viral potential")
                else:
                    insights["recommendations"].append("Optimize content for better viral potential")
            
            # Analyze financial metrics
            if "financial_metrics" in metrics:
                financial_metrics = metrics["financial_metrics"]
                
                if financial_metrics.get("profit_margin", 0) > 0.3:
                    insights["key_insights"].append("Healthy profit margins indicate sustainable business model")
                else:
                    insights["recommendations"].append("Review cost structure to improve profitability")
                
                if financial_metrics.get("revenue_growth", 0) > 0.1:
                    insights["key_insights"].append("Strong revenue growth shows market expansion")
                else:
                    insights["recommendations"].append("Focus on revenue growth strategies")
            
            return insights
            
        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
            raise create_bi_error("performance_insights", "analytics", e)
    
    async def calculate_kpis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        try:
            kpis = {}
            
            # User KPIs
            if "user_metrics" in metrics:
                user_metrics = metrics["user_metrics"]
                kpis["user_kpis"] = {
                    "user_growth_rate": self._calculate_growth_rate(
                        user_metrics.get("new_users", 0),
                        user_metrics.get("total_users", 1)
                    ),
                    "engagement_score": user_metrics.get("user_engagement_rate", 0) * 100,
                    "retention_score": user_metrics.get("user_retention_rate", 0) * 100,
                    "satisfaction_score": user_metrics.get("user_satisfaction_score", 0)
                }
            
            # Content KPIs
            if "content_metrics" in metrics:
                content_metrics = metrics["content_metrics"]
                kpis["content_kpis"] = {
                    "processing_efficiency": content_metrics.get("success_rate", 0) * 100,
                    "quality_score": content_metrics.get("content_quality_score", 0) * 100,
                    "viral_potential": content_metrics.get("viral_potential_score", 0) * 100,
                    "throughput": content_metrics.get("total_videos_processed", 0)
                }
            
            # Financial KPIs
            if "financial_metrics" in metrics:
                financial_metrics = metrics["financial_metrics"]
                kpis["financial_kpis"] = {
                    "revenue_growth": financial_metrics.get("revenue_growth", 0) * 100,
                    "profit_margin": financial_metrics.get("profit_margin", 0) * 100,
                    "roi": self._calculate_roi(
                        financial_metrics.get("profit", 0),
                        financial_metrics.get("costs", 1)
                    ),
                    "customer_lifetime_value": financial_metrics.get("customer_lifetime_value", 0)
                }
            
            # Operational KPIs
            if "operational_metrics" in metrics:
                operational_metrics = metrics["operational_metrics"]
                kpis["operational_kpis"] = {
                    "system_reliability": operational_metrics.get("system_uptime", 0) * 100,
                    "performance_score": self._calculate_performance_score(operational_metrics),
                    "efficiency_score": operational_metrics.get("cache_hit_rate", 0) * 100,
                    "scalability_score": self._calculate_scalability_score(operational_metrics)
                }
            
            return kpis
            
        except Exception as e:
            logger.error(f"KPI calculation failed: {e}")
            raise create_bi_error("kpi_calculation", "analytics", e)
    
    def _calculate_growth_rate(self, new_value: float, total_value: float) -> float:
        """Calculate growth rate percentage"""
        if total_value == 0:
            return 0
        return (new_value / total_value) * 100
    
    def _calculate_roi(self, profit: float, costs: float) -> float:
        """Calculate return on investment"""
        if costs == 0:
            return 0
        return (profit / costs) * 100
    
    def _calculate_performance_score(self, operational_metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        uptime = operational_metrics.get("system_uptime", 0)
        response_time = operational_metrics.get("average_response_time", 1)
        error_rate = operational_metrics.get("error_rate", 0)
        
        # Normalize metrics to 0-100 scale
        uptime_score = uptime * 100
        response_score = max(0, 100 - (response_time * 10))  # Lower response time is better
        error_score = max(0, 100 - (error_rate * 10000))  # Lower error rate is better
        
        # Weighted average
        performance_score = (uptime_score * 0.4 + response_score * 0.3 + error_score * 0.3)
        
        return round(performance_score, 2)
    
    def _calculate_scalability_score(self, operational_metrics: Dict[str, Any]) -> float:
        """Calculate scalability score"""
        cpu_util = operational_metrics.get("cpu_utilization", 0)
        memory_util = operational_metrics.get("memory_utilization", 0)
        throughput = operational_metrics.get("throughput", 0)
        
        # Lower utilization with high throughput indicates good scalability
        utilization_score = max(0, 100 - ((cpu_util + memory_util) * 50))
        throughput_score = min(100, throughput * 2)  # Scale throughput to 0-100
        
        scalability_score = (utilization_score * 0.6 + throughput_score * 0.4)
        
        return round(scalability_score, 2)


class ReportGenerator:
    """Report generation and formatting"""
    
    def __init__(self):
        self.settings = get_settings()
        self.analytics_engine = AnalyticsEngine()
    
    async def generate_performance_report(self, time_range: Tuple[datetime, datetime], 
                                        format: str = "json") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Collect all metrics
            data_collector = DataCollector()
            
            user_metrics = await data_collector.collect_user_metrics(time_range)
            content_metrics = await data_collector.collect_content_metrics(time_range)
            financial_metrics = await data_collector.collect_financial_metrics(time_range)
            operational_metrics = await data_collector.collect_operational_metrics(time_range)
            
            # Combine metrics
            all_metrics = {
                "user_metrics": user_metrics,
                "content_metrics": content_metrics,
                "financial_metrics": financial_metrics,
                "operational_metrics": operational_metrics
            }
            
            # Generate insights and KPIs
            insights = await self.analytics_engine.generate_performance_insights(all_metrics)
            kpis = await self.analytics_engine.calculate_kpis(all_metrics)
            
            # Create report
            report = {
                "report_id": str(uuid4()),
                "report_type": "performance",
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "metrics": all_metrics,
                "insights": insights,
                "kpis": kpis,
                "summary": self._generate_report_summary(insights, kpis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            raise create_bi_error("performance_report", "reporting", e)
    
    def _generate_report_summary(self, insights: Dict[str, Any], kpis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary"""
        summary = {
            "overall_status": "excellent",
            "key_highlights": [],
            "areas_for_improvement": [],
            "next_actions": []
        }
        
        # Analyze insights
        if insights.get("overall_performance") == "excellent":
            summary["key_highlights"].append("Overall performance is excellent across all metrics")
        
        # Add key insights as highlights
        for insight in insights.get("key_insights", []):
            summary["key_highlights"].append(insight)
        
        # Add recommendations as improvement areas
        for recommendation in insights.get("recommendations", []):
            summary["areas_for_improvement"].append(recommendation)
        
        # Add alerts as next actions
        for alert in insights.get("alerts", []):
            summary["next_actions"].append(f"Address: {alert}")
        
        return summary
    
    async def generate_financial_report(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate financial report"""
        try:
            data_collector = DataCollector()
            financial_metrics = await data_collector.collect_financial_metrics(time_range)
            
            # Calculate additional financial metrics
            revenue = financial_metrics.get("revenue", 0)
            costs = financial_metrics.get("costs", 0)
            profit = financial_metrics.get("profit", 0)
            
            # Calculate ratios and trends
            financial_analysis = {
                "revenue_analysis": {
                    "total_revenue": revenue,
                    "revenue_growth": financial_metrics.get("revenue_growth", 0),
                    "monthly_recurring_revenue": financial_metrics.get("monthly_recurring_revenue", 0),
                    "average_revenue_per_user": financial_metrics.get("average_revenue_per_user", 0)
                },
                "cost_analysis": {
                    "total_costs": costs,
                    "customer_acquisition_cost": financial_metrics.get("customer_acquisition_cost", 0),
                    "cost_per_processing": costs / max(1, financial_metrics.get("total_videos_processed", 1))
                },
                "profitability_analysis": {
                    "gross_profit": profit,
                    "profit_margin": financial_metrics.get("profit_margin", 0),
                    "roi": (profit / costs * 100) if costs > 0 else 0
                },
                "customer_metrics": {
                    "customer_lifetime_value": financial_metrics.get("customer_lifetime_value", 0),
                    "churn_rate": financial_metrics.get("churn_rate", 0),
                    "ltv_cac_ratio": financial_metrics.get("customer_lifetime_value", 0) / 
                                   max(1, financial_metrics.get("customer_acquisition_cost", 1))
                }
            }
            
            report = {
                "report_id": str(uuid4()),
                "report_type": "financial",
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "financial_metrics": financial_metrics,
                "financial_analysis": financial_analysis,
                "recommendations": self._generate_financial_recommendations(financial_analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Financial report generation failed: {e}")
            raise create_bi_error("financial_report", "reporting", e)
    
    def _generate_financial_recommendations(self, financial_analysis: Dict[str, Any]) -> List[str]:
        """Generate financial recommendations"""
        recommendations = []
        
        # Revenue recommendations
        revenue_growth = financial_analysis["revenue_analysis"]["revenue_growth"]
        if revenue_growth < 0.1:  # Less than 10% growth
            recommendations.append("Focus on revenue growth strategies to achieve target growth rate")
        
        # Cost recommendations
        ltv_cac_ratio = financial_analysis["customer_metrics"]["ltv_cac_ratio"]
        if ltv_cac_ratio < 3:  # LTV should be at least 3x CAC
            recommendations.append("Optimize customer acquisition cost or increase customer lifetime value")
        
        # Profitability recommendations
        profit_margin = financial_analysis["profitability_analysis"]["profit_margin"]
        if profit_margin < 0.2:  # Less than 20% margin
            recommendations.append("Review cost structure to improve profit margins")
        
        # Churn recommendations
        churn_rate = financial_analysis["customer_metrics"]["churn_rate"]
        if churn_rate > 0.05:  # More than 5% monthly churn
            recommendations.append("Implement customer retention strategies to reduce churn")
        
        return recommendations


class DashboardManager:
    """Dashboard creation and management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.dashboards: Dict[str, DashboardDefinition] = {}
        self._create_default_dashboards()
    
    def _create_default_dashboards(self):
        """Create default dashboards"""
        try:
            # Executive Dashboard
            executive_dashboard = DashboardDefinition(
                dashboard_id="executive_dashboard",
                name="Executive Dashboard",
                description="High-level business metrics for executives",
                dashboard_type=DashboardType.EXECUTIVE,
                widgets=[
                    {
                        "widget_id": "revenue_kpi",
                        "type": "kpi",
                        "title": "Monthly Revenue",
                        "metric": "revenue",
                        "position": {"x": 0, "y": 0, "w": 3, "h": 2}
                    },
                    {
                        "widget_id": "user_growth",
                        "type": "line_chart",
                        "title": "User Growth",
                        "metric": "total_users",
                        "position": {"x": 3, "y": 0, "w": 6, "h": 2}
                    },
                    {
                        "widget_id": "content_metrics",
                        "type": "bar_chart",
                        "title": "Content Processing",
                        "metric": "total_videos_processed",
                        "position": {"x": 9, "y": 0, "w": 3, "h": 2}
                    }
                ],
                layout={"columns": 12, "rows": 4},
                filters={"time_range": "30d"},
                refresh_interval=300
            )
            
            self.dashboards["executive_dashboard"] = executive_dashboard
            
            # Operational Dashboard
            operational_dashboard = DashboardDefinition(
                dashboard_id="operational_dashboard",
                name="Operational Dashboard",
                description="Real-time operational metrics",
                dashboard_type=DashboardType.OPERATIONAL,
                widgets=[
                    {
                        "widget_id": "system_health",
                        "type": "gauge",
                        "title": "System Health",
                        "metric": "system_uptime",
                        "position": {"x": 0, "y": 0, "w": 4, "h": 2}
                    },
                    {
                        "widget_id": "processing_queue",
                        "type": "real_time_chart",
                        "title": "Processing Queue",
                        "metric": "queue_size",
                        "position": {"x": 4, "y": 0, "w": 4, "h": 2}
                    },
                    {
                        "widget_id": "error_rate",
                        "type": "sparkline",
                        "title": "Error Rate",
                        "metric": "error_rate",
                        "position": {"x": 8, "y": 0, "w": 4, "h": 2}
                    }
                ],
                layout={"columns": 12, "rows": 4},
                filters={"time_range": "1h"},
                refresh_interval=30
            )
            
            self.dashboards["operational_dashboard"] = operational_dashboard
            
            logger.info("Default dashboards created")
            
        except Exception as e:
            logger.error(f"Default dashboard creation failed: {e}")
    
    def create_dashboard(self, dashboard: DashboardDefinition) -> bool:
        """Create new dashboard"""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Created dashboard: {dashboard.dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            raise create_bi_error("dashboard_creation", dashboard.name, e)
    
    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardDefinition]:
        """Get dashboard definition"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[DashboardDefinition]:
        """List all dashboards"""
        return list(self.dashboards.values())
    
    async def render_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Render dashboard with data"""
        try:
            dashboard = self.get_dashboard(dashboard_id)
            if not dashboard:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            # Collect data for dashboard
            data_collector = DataCollector()
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)  # Default 30 days
            
            user_metrics = await data_collector.collect_user_metrics((start_time, end_time))
            content_metrics = await data_collector.collect_content_metrics((start_time, end_time))
            financial_metrics = await data_collector.collect_financial_metrics((start_time, end_time))
            operational_metrics = await data_collector.collect_operational_metrics((start_time, end_time))
            
            # Combine all metrics
            all_metrics = {
                **user_metrics,
                **content_metrics,
                **financial_metrics,
                **operational_metrics
            }
            
            # Render widgets
            rendered_widgets = []
            for widget in dashboard.widgets:
                widget_data = self._render_widget(widget, all_metrics)
                rendered_widgets.append(widget_data)
            
            return {
                "dashboard_id": dashboard_id,
                "dashboard_name": dashboard.name,
                "rendered_at": datetime.utcnow().isoformat(),
                "widgets": rendered_widgets,
                "layout": dashboard.layout
            }
            
        except Exception as e:
            logger.error(f"Dashboard rendering failed: {e}")
            raise create_bi_error("dashboard_rendering", dashboard_id, e)
    
    def _render_widget(self, widget: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Render individual widget"""
        widget_type = widget["type"]
        metric_name = widget["metric"]
        
        # Get metric value
        metric_value = metrics.get(metric_name, 0)
        
        # Render based on widget type
        if widget_type == "kpi":
            return {
                "widget_id": widget["widget_id"],
                "type": "kpi",
                "title": widget["title"],
                "value": metric_value,
                "formatted_value": self._format_metric_value(metric_value, metric_name),
                "trend": self._calculate_trend(metric_name, metric_value),
                "position": widget["position"]
            }
        
        elif widget_type == "line_chart":
            return {
                "widget_id": widget["widget_id"],
                "type": "line_chart",
                "title": widget["title"],
                "data": self._generate_chart_data(metric_name, "line"),
                "position": widget["position"]
            }
        
        elif widget_type == "bar_chart":
            return {
                "widget_id": widget["widget_id"],
                "type": "bar_chart",
                "title": widget["title"],
                "data": self._generate_chart_data(metric_name, "bar"),
                "position": widget["position"]
            }
        
        elif widget_type == "gauge":
            return {
                "widget_id": widget["widget_id"],
                "type": "gauge",
                "title": widget["title"],
                "value": metric_value,
                "min": 0,
                "max": 100,
                "position": widget["position"]
            }
        
        else:
            return {
                "widget_id": widget["widget_id"],
                "type": widget_type,
                "title": widget["title"],
                "data": {"error": "Unknown widget type"},
                "position": widget["position"]
            }
    
    def _format_metric_value(self, value: Any, metric_name: str) -> str:
        """Format metric value for display"""
        if isinstance(value, float):
            if "rate" in metric_name or "ratio" in metric_name:
                return f"{value:.1%}"
            elif "revenue" in metric_name or "cost" in metric_name:
                return f"${value:,.2f}"
            else:
                return f"{value:.2f}"
        else:
            return str(value)
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for metric"""
        # Simulate trend calculation
        # In practice, this would compare with historical data
        if current_value > 0:
            return "up"
        elif current_value < 0:
            return "down"
        else:
            return "stable"
    
    def _generate_chart_data(self, metric_name: str, chart_type: str) -> Dict[str, Any]:
        """Generate chart data"""
        # Simulate chart data generation
        # In practice, this would use real historical data
        
        if chart_type == "line":
            return {
                "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
                "datasets": [{
                    "label": metric_name,
                    "data": [100, 120, 110, 140],
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)"
                }]
            }
        
        elif chart_type == "bar":
            return {
                "labels": ["Jan", "Feb", "Mar", "Apr"],
                "datasets": [{
                    "label": metric_name,
                    "data": [50, 60, 70, 80],
                    "backgroundColor": "rgba(54, 162, 235, 0.6)"
                }]
            }
        
        return {"error": "Unknown chart type"}


class BusinessIntelligenceManager:
    """Main business intelligence manager"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_collector = DataCollector()
        self.analytics_engine = AnalyticsEngine()
        self.report_generator = ReportGenerator()
        self.dashboard_manager = DashboardManager()
        
        self.business_metrics: Dict[str, BusinessMetric] = {}
        self.reports: Dict[str, ReportDefinition] = {}
        
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default business metrics"""
        try:
            # User metrics
            user_engagement = BusinessMetric(
                metric_id="user_engagement_rate",
                name="User Engagement Rate",
                description="Percentage of active users engaging with the platform",
                metric_type=MetricType.GAUGE,
                category="user",
                unit="percentage",
                target_value=0.75,
                threshold_warning=0.65,
                threshold_critical=0.55
            )
            self.business_metrics["user_engagement_rate"] = user_engagement
            
            # Content metrics
            processing_success = BusinessMetric(
                metric_id="processing_success_rate",
                name="Processing Success Rate",
                description="Percentage of successful video processing operations",
                metric_type=MetricType.GAUGE,
                category="content",
                unit="percentage",
                target_value=0.95,
                threshold_warning=0.90,
                threshold_critical=0.85
            )
            self.business_metrics["processing_success_rate"] = processing_success
            
            # Financial metrics
            revenue_growth = BusinessMetric(
                metric_id="revenue_growth",
                name="Revenue Growth",
                description="Month-over-month revenue growth rate",
                metric_type=MetricType.GAUGE,
                category="financial",
                unit="percentage",
                target_value=0.15,
                threshold_warning=0.10,
                threshold_critical=0.05
            )
            self.business_metrics["revenue_growth"] = revenue_growth
            
            # Operational metrics
            system_uptime = BusinessMetric(
                metric_id="system_uptime",
                name="System Uptime",
                description="System availability percentage",
                metric_type=MetricType.GAUGE,
                category="operational",
                unit="percentage",
                target_value=0.999,
                threshold_warning=0.995,
                threshold_critical=0.99
            )
            self.business_metrics["system_uptime"] = system_uptime
            
            logger.info("Default business metrics initialized")
            
        except Exception as e:
            logger.error(f"Default metrics initialization failed: {e}")
    
    async def generate_comprehensive_report(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate comprehensive business intelligence report"""
        try:
            # Generate all types of reports
            performance_report = await self.report_generator.generate_performance_report(time_range)
            financial_report = await self.report_generator.generate_financial_report(time_range)
            
            # Combine reports
            comprehensive_report = {
                "report_id": str(uuid4()),
                "report_type": "comprehensive",
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "performance_report": performance_report,
                "financial_report": financial_report,
                "executive_summary": self._generate_executive_summary(performance_report, financial_report)
            }
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            raise create_bi_error("comprehensive_report", "reporting", e)
    
    def _generate_executive_summary(self, performance_report: Dict[str, Any], 
                                  financial_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        summary = {
            "overall_health": "excellent",
            "key_achievements": [],
            "critical_issues": [],
            "strategic_recommendations": []
        }
        
        # Analyze performance insights
        performance_insights = performance_report.get("insights", {})
        if performance_insights.get("overall_performance") == "excellent":
            summary["key_achievements"].append("Excellent overall performance across all metrics")
        
        # Add key insights as achievements
        for insight in performance_insights.get("key_insights", []):
            summary["key_achievements"].append(insight)
        
        # Add alerts as critical issues
        for alert in performance_insights.get("alerts", []):
            summary["critical_issues"].append(alert)
        
        # Add recommendations as strategic recommendations
        for recommendation in performance_insights.get("recommendations", []):
            summary["strategic_recommendations"].append(recommendation)
        
        # Analyze financial recommendations
        financial_recommendations = financial_report.get("recommendations", [])
        for recommendation in financial_recommendations:
            summary["strategic_recommendations"].append(recommendation)
        
        return summary
    
    def get_bi_statistics(self) -> Dict[str, Any]:
        """Get business intelligence system statistics"""
        try:
            return {
                "total_metrics": len(self.business_metrics),
                "total_dashboards": len(self.dashboard_manager.dashboards),
                "total_reports": len(self.reports),
                "metrics_by_category": self._get_metrics_by_category(),
                "dashboard_types": self._get_dashboard_types(),
                "last_report_generated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"BI statistics failed: {e}")
            return {}
    
    def _get_metrics_by_category(self) -> Dict[str, int]:
        """Get metrics count by category"""
        categories = {}
        for metric in self.business_metrics.values():
            category = metric.category
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _get_dashboard_types(self) -> Dict[str, int]:
        """Get dashboard count by type"""
        types = {}
        for dashboard in self.dashboard_manager.dashboards.values():
            dashboard_type = dashboard.dashboard_type.value
            types[dashboard_type] = types.get(dashboard_type, 0) + 1
        return types


# Global business intelligence manager
bi_manager = BusinessIntelligenceManager()





























