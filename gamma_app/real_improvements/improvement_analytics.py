"""
Gamma App - Real Improvement Analytics
Advanced analytics for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Analytics types"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"

class AnalyticsMetric(Enum):
    """Analytics metrics"""
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"
    IMPACT = "impact"
    COST = "cost"
    QUALITY = "quality"
    TIME_TO_COMPLETION = "time_to_completion"

@dataclass
class AnalyticsReport:
    """Analytics report"""
    report_id: str
    name: str
    type: AnalyticsType
    metrics: List[AnalyticsMetric]
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    created_at: datetime = None
    generated_by: str = "system"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class AnalyticsDashboard:
    """Analytics dashboard"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    filters: Dict[str, Any]
    refresh_interval: int = 300  # seconds
    created_at: datetime = None
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementAnalytics:
    """
    Advanced analytics for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize analytics engine"""
        self.project_root = Path(project_root)
        self.reports: Dict[str, AnalyticsReport] = {}
        self.dashboards: Dict[str, AnalyticsDashboard] = {}
        self.analytics_data: Dict[str, pd.DataFrame] = {}
        self.analytics_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default dashboards
        self._initialize_default_dashboards()
        
        logger.info(f"Real Improvement Analytics initialized for {self.project_root}")
    
    def _initialize_default_dashboards(self):
        """Initialize default analytics dashboards"""
        # Performance Dashboard
        performance_dashboard = AnalyticsDashboard(
            dashboard_id="performance_dashboard",
            name="Performance Analytics Dashboard",
            description="Comprehensive performance analytics for improvements",
            widgets=[
                {
                    "widget_id": "success_rate_chart",
                    "type": "line_chart",
                    "title": "Success Rate Over Time",
                    "metrics": ["success_rate"],
                    "time_range": "30_days"
                },
                {
                    "widget_id": "efficiency_metrics",
                    "type": "bar_chart",
                    "title": "Efficiency Metrics",
                    "metrics": ["efficiency", "time_to_completion"],
                    "group_by": "category"
                },
                {
                    "widget_id": "impact_analysis",
                    "type": "scatter_plot",
                    "title": "Impact vs Effort Analysis",
                    "metrics": ["impact", "effort_hours"],
                    "color_by": "priority"
                }
            ],
            filters={
                "date_range": "30_days",
                "categories": "all",
                "priorities": "all"
            }
        )
        self.dashboards[performance_dashboard.dashboard_id] = performance_dashboard
        
        # Quality Dashboard
        quality_dashboard = AnalyticsDashboard(
            dashboard_id="quality_dashboard",
            name="Quality Analytics Dashboard",
            description="Code quality and improvement analytics",
            widgets=[
                {
                    "widget_id": "quality_trends",
                    "type": "line_chart",
                    "title": "Quality Trends",
                    "metrics": ["quality_score"],
                    "time_range": "90_days"
                },
                {
                    "widget_id": "bug_analysis",
                    "type": "pie_chart",
                    "title": "Bug Distribution",
                    "metrics": ["bug_count"],
                    "group_by": "severity"
                },
                {
                    "widget_id": "test_coverage",
                    "type": "gauge_chart",
                    "title": "Test Coverage",
                    "metrics": ["test_coverage"],
                    "threshold": 80
                }
            ],
            filters={
                "date_range": "90_days",
                "quality_threshold": 70
            }
        )
        self.dashboards[quality_dashboard.dashboard_id] = quality_dashboard
    
    def create_analytics_report(self, name: str, type: AnalyticsType, 
                              metrics: List[AnalyticsMetric], data: Dict[str, Any]) -> str:
        """Create analytics report"""
        try:
            report_id = f"report_{int(time.time() * 1000)}"
            
            # Generate insights and recommendations
            insights = self._generate_insights(data, metrics)
            recommendations = self._generate_recommendations(data, metrics)
            
            report = AnalyticsReport(
                report_id=report_id,
                name=name,
                type=type,
                metrics=metrics,
                data=data,
                insights=insights,
                recommendations=recommendations
            )
            
            self.reports[report_id] = report
            
            logger.info(f"Analytics report created: {name}")
            return report_id
            
        except Exception as e:
            logger.error(f"Failed to create analytics report: {e}")
            raise
    
    def _generate_insights(self, data: Dict[str, Any], metrics: List[AnalyticsMetric]) -> List[str]:
        """Generate insights from data"""
        insights = []
        
        try:
            # Success rate insights
            if AnalyticsMetric.SUCCESS_RATE in metrics:
                success_rate = data.get("success_rate", 0)
                if success_rate > 80:
                    insights.append(f"Excellent success rate of {success_rate:.1f}% indicates effective improvement processes")
                elif success_rate > 60:
                    insights.append(f"Good success rate of {success_rate:.1f}% with room for improvement")
                else:
                    insights.append(f"Low success rate of {success_rate:.1f}% requires immediate attention")
            
            # Efficiency insights
            if AnalyticsMetric.EFFICIENCY in metrics:
                efficiency = data.get("efficiency", 0)
                if efficiency > 0.8:
                    insights.append(f"High efficiency score of {efficiency:.2f} shows optimized processes")
                elif efficiency > 0.6:
                    insights.append(f"Moderate efficiency score of {efficiency:.2f} suggests process optimization opportunities")
                else:
                    insights.append(f"Low efficiency score of {efficiency:.2f} indicates significant process issues")
            
            # Impact insights
            if AnalyticsMetric.IMPACT in metrics:
                impact = data.get("impact", 0)
                if impact > 8:
                    insights.append(f"High impact score of {impact} demonstrates significant value creation")
                elif impact > 5:
                    insights.append(f"Moderate impact score of {impact} shows reasonable value creation")
                else:
                    insights.append(f"Low impact score of {impact} suggests need for better impact assessment")
            
            # Cost insights
            if AnalyticsMetric.COST in metrics:
                cost = data.get("cost", 0)
                if cost < 1000:
                    insights.append(f"Low cost of ${cost} indicates efficient resource utilization")
                elif cost < 5000:
                    insights.append(f"Moderate cost of ${cost} shows reasonable resource investment")
                else:
                    insights.append(f"High cost of ${cost} suggests need for cost optimization")
            
            # Quality insights
            if AnalyticsMetric.QUALITY in metrics:
                quality = data.get("quality", 0)
                if quality > 8:
                    insights.append(f"High quality score of {quality} indicates excellent code quality")
                elif quality > 6:
                    insights.append(f"Good quality score of {quality} shows solid code quality")
                else:
                    insights.append(f"Low quality score of {quality} suggests need for quality improvements")
            
            # Time to completion insights
            if AnalyticsMetric.TIME_TO_COMPLETION in metrics:
                time_to_completion = data.get("time_to_completion", 0)
                if time_to_completion < 24:
                    insights.append(f"Fast completion time of {time_to_completion} hours shows efficient execution")
                elif time_to_completion < 72:
                    insights.append(f"Reasonable completion time of {time_to_completion} hours")
                else:
                    insights.append(f"Slow completion time of {time_to_completion} hours suggests process bottlenecks")
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            insights.append("Unable to generate insights due to data processing error")
        
        return insights
    
    def _generate_recommendations(self, data: Dict[str, Any], metrics: List[AnalyticsMetric]) -> List[str]:
        """Generate recommendations from data"""
        recommendations = []
        
        try:
            # Success rate recommendations
            if AnalyticsMetric.SUCCESS_RATE in metrics:
                success_rate = data.get("success_rate", 0)
                if success_rate < 70:
                    recommendations.append("Implement better testing and validation processes to improve success rate")
                    recommendations.append("Provide additional training for team members on improvement methodologies")
                    recommendations.append("Establish clearer success criteria and acceptance standards")
            
            # Efficiency recommendations
            if AnalyticsMetric.EFFICIENCY in metrics:
                efficiency = data.get("efficiency", 0)
                if efficiency < 0.7:
                    recommendations.append("Automate repetitive tasks to improve efficiency")
                    recommendations.append("Implement better project management tools and processes")
                    recommendations.append("Reduce unnecessary meetings and focus on execution")
            
            # Impact recommendations
            if AnalyticsMetric.IMPACT in metrics:
                impact = data.get("impact", 0)
                if impact < 6:
                    recommendations.append("Focus on high-impact improvements with clear business value")
                    recommendations.append("Implement better impact measurement and tracking")
                    recommendations.append("Prioritize improvements based on ROI and business objectives")
            
            # Cost recommendations
            if AnalyticsMetric.COST in metrics:
                cost = data.get("cost", 0)
                if cost > 5000:
                    recommendations.append("Implement cost controls and budget monitoring")
                    recommendations.append("Consider outsourcing non-critical improvements")
                    recommendations.append("Optimize resource allocation and utilization")
            
            # Quality recommendations
            if AnalyticsMetric.QUALITY in metrics:
                quality = data.get("quality", 0)
                if quality < 7:
                    recommendations.append("Implement code review processes and quality gates")
                    recommendations.append("Provide training on coding best practices and standards")
                    recommendations.append("Establish quality metrics and monitoring")
            
            # Time to completion recommendations
            if AnalyticsMetric.TIME_TO_COMPLETION in metrics:
                time_to_completion = data.get("time_to_completion", 0)
                if time_to_completion > 72:
                    recommendations.append("Break down large improvements into smaller, manageable tasks")
                    recommendations.append("Implement better project planning and resource allocation")
                    recommendations.append("Identify and eliminate process bottlenecks")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to data processing error")
        
        return recommendations
    
    async def generate_descriptive_analytics(self, data: pd.DataFrame, 
                                          group_by: str = None) -> Dict[str, Any]:
        """Generate descriptive analytics"""
        try:
            analytics = {}
            
            # Basic statistics
            analytics["total_improvements"] = len(data)
            analytics["success_rate"] = (data["success"].sum() / len(data)) * 100 if "success" in data.columns else 0
            analytics["average_effort"] = data["effort_hours"].mean() if "effort_hours" in data.columns else 0
            analytics["average_impact"] = data["impact_score"].mean() if "impact_score" in data.columns else 0
            
            # Group by analysis
            if group_by and group_by in data.columns:
                group_stats = data.groupby(group_by).agg({
                    "success": "sum",
                    "effort_hours": "mean",
                    "impact_score": "mean"
                }).to_dict()
                analytics["group_analysis"] = group_stats
            
            # Time series analysis
            if "created_at" in data.columns:
                data["created_at"] = pd.to_datetime(data["created_at"])
                data["month"] = data["created_at"].dt.to_period("M")
                monthly_stats = data.groupby("month").agg({
                    "success": "sum",
                    "effort_hours": "sum",
                    "impact_score": "mean"
                }).to_dict()
                analytics["monthly_trends"] = monthly_stats
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate descriptive analytics: {e}")
            return {}
    
    async def generate_diagnostic_analytics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate diagnostic analytics"""
        try:
            analytics = {}
            
            # Correlation analysis
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                correlation_matrix = data[numeric_columns].corr()
                analytics["correlations"] = correlation_matrix.to_dict()
            
            # Success factor analysis
            if "success" in data.columns:
                success_data = data[data["success"] == True]
                failure_data = data[data["success"] == False]
                
                if len(success_data) > 0 and len(failure_data) > 0:
                    success_factors = {}
                    for column in numeric_columns:
                        if column != "success":
                            success_mean = success_data[column].mean()
                            failure_mean = failure_data[column].mean()
                            success_factors[column] = {
                                "success_mean": success_mean,
                                "failure_mean": failure_mean,
                                "difference": success_mean - failure_mean
                            }
                    analytics["success_factors"] = success_factors
            
            # Outlier analysis
            for column in numeric_columns:
                if column != "success":
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                    if len(outliers) > 0:
                        analytics[f"{column}_outliers"] = {
                            "count": len(outliers),
                            "percentage": (len(outliers) / len(data)) * 100,
                            "values": outliers[column].tolist()
                        }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostic analytics: {e}")
            return {}
    
    async def generate_predictive_analytics(self, data: pd.DataFrame, 
                                         target_column: str = "success") -> Dict[str, Any]:
        """Generate predictive analytics"""
        try:
            analytics = {}
            
            # Feature importance analysis
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Prepare data
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_column] if target_column in data.columns else [])
            y = data[target_column] if target_column in data.columns else None
            
            if y is not None and len(X.columns) > 0:
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Feature importance
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                analytics["feature_importance"] = feature_importance
                
                # Model performance
                accuracy = model.score(X_test, y_test)
                analytics["model_accuracy"] = accuracy
            
            # Trend analysis
            if "created_at" in data.columns:
                data["created_at"] = pd.to_datetime(data["created_at"])
                data["month"] = data["created_at"].dt.to_period("M")
                
                # Calculate trends
                monthly_data = data.groupby("month").agg({
                    target_column: "mean" if target_column in data.columns else "count"
                })
                
                if len(monthly_data) > 1:
                    trend = np.polyfit(range(len(monthly_data)), monthly_data[target_column], 1)[0]
                    analytics["trend"] = trend
                    analytics["trend_direction"] = "increasing" if trend > 0 else "decreasing"
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate predictive analytics: {e}")
            return {}
    
    async def generate_prescriptive_analytics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate prescriptive analytics"""
        try:
            analytics = {}
            
            # Optimization recommendations
            if "effort_hours" in data.columns and "impact_score" in data.columns:
                # Calculate efficiency (impact per hour)
                data["efficiency"] = data["impact_score"] / data["effort_hours"]
                
                # Find optimal improvements
                optimal_improvements = data.nlargest(10, "efficiency")
                analytics["optimal_improvements"] = optimal_improvements[["effort_hours", "impact_score", "efficiency"]].to_dict()
            
            # Resource allocation recommendations
            if "category" in data.columns and "effort_hours" in data.columns:
                category_effort = data.groupby("category")["effort_hours"].sum()
                category_impact = data.groupby("category")["impact_score"].sum()
                
                # Calculate ROI by category
                roi_by_category = (category_impact / category_effort).to_dict()
                analytics["roi_by_category"] = roi_by_category
                
                # Recommend resource allocation
                recommended_allocation = {}
                total_effort = category_effort.sum()
                for category, effort in category_effort.items():
                    recommended_allocation[category] = (effort / total_effort) * 100
                analytics["recommended_allocation"] = recommended_allocation
            
            # Priority recommendations
            if "priority" in data.columns and "success" in data.columns:
                priority_success = data.groupby("priority")["success"].mean()
                analytics["priority_success_rates"] = priority_success.to_dict()
                
                # Recommend priority adjustments
                low_success_priorities = priority_success[priority_success < 0.5].index.tolist()
                analytics["priority_adjustments"] = {
                    "low_success_priorities": low_success_priorities,
                    "recommendation": "Consider adjusting priorities for low-success categories"
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate prescriptive analytics: {e}")
            return {}
    
    def create_analytics_dashboard(self, name: str, description: str, 
                                 widgets: List[Dict[str, Any]], 
                                 filters: Dict[str, Any] = None) -> str:
        """Create analytics dashboard"""
        try:
            dashboard_id = f"dashboard_{int(time.time() * 1000)}"
            
            dashboard = AnalyticsDashboard(
                dashboard_id=dashboard_id,
                name=name,
                description=description,
                widgets=widgets,
                filters=filters or {}
            )
            
            self.dashboards[dashboard_id] = dashboard
            
            logger.info(f"Analytics dashboard created: {name}")
            return dashboard_id
            
        except Exception as e:
            logger.error(f"Failed to create analytics dashboard: {e}")
            raise
    
    async def update_dashboard_data(self, dashboard_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Update dashboard data"""
        try:
            if dashboard_id not in self.dashboards:
                return {"success": False, "error": "Dashboard not found"}
            
            dashboard = self.dashboards[dashboard_id]
            dashboard.last_updated = datetime.utcnow()
            
            # Process widgets
            widget_data = {}
            for widget in dashboard.widgets:
                widget_id = widget["widget_id"]
                widget_type = widget["type"]
                metrics = widget["metrics"]
                
                # Generate widget data based on type
                if widget_type == "line_chart":
                    widget_data[widget_id] = await self._generate_line_chart_data(data, metrics, widget)
                elif widget_type == "bar_chart":
                    widget_data[widget_id] = await self._generate_bar_chart_data(data, metrics, widget)
                elif widget_type == "scatter_plot":
                    widget_data[widget_id] = await self._generate_scatter_plot_data(data, metrics, widget)
                elif widget_type == "pie_chart":
                    widget_data[widget_id] = await self._generate_pie_chart_data(data, metrics, widget)
                elif widget_type == "gauge_chart":
                    widget_data[widget_id] = await self._generate_gauge_chart_data(data, metrics, widget)
            
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "widget_data": widget_data,
                "last_updated": dashboard.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_line_chart_data(self, data: pd.DataFrame, metrics: List[str], widget: Dict[str, Any]) -> Dict[str, Any]:
        """Generate line chart data"""
        try:
            # Group by time if time column exists
            if "created_at" in data.columns:
                data["created_at"] = pd.to_datetime(data["created_at"])
                data["date"] = data["created_at"].dt.date
                
                chart_data = {}
                for metric in metrics:
                    if metric in data.columns:
                        metric_data = data.groupby("date")[metric].mean()
                        chart_data[metric] = {
                            "dates": [str(date) for date in metric_data.index],
                            "values": metric_data.values.tolist()
                        }
                
                return {"type": "line_chart", "data": chart_data}
            else:
                return {"type": "line_chart", "data": {}}
                
        except Exception as e:
            logger.error(f"Failed to generate line chart data: {e}")
            return {"type": "line_chart", "data": {}}
    
    async def _generate_bar_chart_data(self, data: pd.DataFrame, metrics: List[str], widget: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bar chart data"""
        try:
            group_by = widget.get("group_by")
            
            if group_by and group_by in data.columns:
                chart_data = {}
                for metric in metrics:
                    if metric in data.columns:
                        metric_data = data.groupby(group_by)[metric].mean()
                        chart_data[metric] = {
                            "categories": metric_data.index.tolist(),
                            "values": metric_data.values.tolist()
                        }
                
                return {"type": "bar_chart", "data": chart_data}
            else:
                return {"type": "bar_chart", "data": {}}
                
        except Exception as e:
            logger.error(f"Failed to generate bar chart data: {e}")
            return {"type": "bar_chart", "data": {}}
    
    async def _generate_scatter_plot_data(self, data: pd.DataFrame, metrics: List[str], widget: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scatter plot data"""
        try:
            if len(metrics) >= 2:
                x_metric = metrics[0]
                y_metric = metrics[1]
                color_by = widget.get("color_by")
                
                if x_metric in data.columns and y_metric in data.columns:
                    scatter_data = {
                        "x": data[x_metric].tolist(),
                        "y": data[y_metric].tolist(),
                        "x_label": x_metric,
                        "y_label": y_metric
                    }
                    
                    if color_by and color_by in data.columns:
                        scatter_data["color"] = data[color_by].tolist()
                        scatter_data["color_label"] = color_by
                    
                    return {"type": "scatter_plot", "data": scatter_data}
            
            return {"type": "scatter_plot", "data": {}}
                
        except Exception as e:
            logger.error(f"Failed to generate scatter plot data: {e}")
            return {"type": "scatter_plot", "data": {}}
    
    async def _generate_pie_chart_data(self, data: pd.DataFrame, metrics: List[str], widget: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pie chart data"""
        try:
            group_by = widget.get("group_by")
            metric = metrics[0] if metrics else "count"
            
            if group_by and group_by in data.columns:
                if metric == "count":
                    pie_data = data[group_by].value_counts()
                else:
                    pie_data = data.groupby(group_by)[metric].sum()
                
                return {
                    "type": "pie_chart",
                    "data": {
                        "labels": pie_data.index.tolist(),
                        "values": pie_data.values.tolist()
                    }
                }
            
            return {"type": "pie_chart", "data": {}}
                
        except Exception as e:
            logger.error(f"Failed to generate pie chart data: {e}")
            return {"type": "pie_chart", "data": {}}
    
    async def _generate_gauge_chart_data(self, data: pd.DataFrame, metrics: List[str], widget: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gauge chart data"""
        try:
            metric = metrics[0] if metrics else "value"
            threshold = widget.get("threshold", 50)
            
            if metric in data.columns:
                value = data[metric].mean()
                return {
                    "type": "gauge_chart",
                    "data": {
                        "value": value,
                        "threshold": threshold,
                        "label": metric
                    }
                }
            
            return {"type": "gauge_chart", "data": {}}
                
        except Exception as e:
            logger.error(f"Failed to generate gauge chart data: {e}")
            return {"type": "gauge_chart", "data": {}}
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        total_reports = len(self.reports)
        total_dashboards = len(self.dashboards)
        
        return {
            "total_reports": total_reports,
            "total_dashboards": total_dashboards,
            "analytics_types": list(set(r.type.value for r in self.reports.values())),
            "dashboard_widgets": sum(len(d.widgets) for d in self.dashboards.values())
        }
    
    def get_report_data(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report data"""
        if report_id not in self.reports:
            return None
        
        report = self.reports[report_id]
        
        return {
            "report_id": report_id,
            "name": report.name,
            "type": report.type.value,
            "metrics": [m.value for m in report.metrics],
            "data": report.data,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "created_at": report.created_at.isoformat(),
            "generated_by": report.generated_by
        }
    
    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data"""
        if dashboard_id not in self.dashboards:
            return None
        
        dashboard = self.dashboards[dashboard_id]
        
        return {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "widgets": dashboard.widgets,
            "filters": dashboard.filters,
            "refresh_interval": dashboard.refresh_interval,
            "created_at": dashboard.created_at.isoformat(),
            "last_updated": dashboard.last_updated.isoformat() if dashboard.last_updated else None
        }

# Global analytics instance
improvement_analytics = None

def get_improvement_analytics() -> RealImprovementAnalytics:
    """Get improvement analytics instance"""
    global improvement_analytics
    if not improvement_analytics:
        improvement_analytics = RealImprovementAnalytics()
    return improvement_analytics













