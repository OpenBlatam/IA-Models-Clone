"""
Advanced Analytics Service
==========================

Comprehensive analytics service with predictive analytics, business intelligence,
and advanced reporting capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from ..models import Workflow, WorkflowExecution, AgentExecution, BusinessAgent, User, Document
from ..services.database_service import DatabaseService
from ..services.ai_service import AIService

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types of analytics."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    REAL_TIME = "real_time"

class ReportType(Enum):
    """Types of reports."""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    USER = "user"
    SYSTEM = "system"
    CUSTOM = "custom"

class VisualizationType(Enum):
    """Types of visualizations."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    DASHBOARD = "dashboard"
    INTERACTIVE = "interactive"

@dataclass
class AnalyticsMetric:
    """Analytics metric definition."""
    metric_id: str
    name: str
    description: str
    value: float
    unit: str
    category: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AnalyticsReport:
    """Analytics report definition."""
    report_id: str
    title: str
    description: str
    report_type: ReportType
    analytics_type: AnalyticsType
    metrics: List[AnalyticsMetric]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    generated_by: str
    parameters: Dict[str, Any]

@dataclass
class PredictiveModel:
    """Predictive model definition."""
    model_id: str
    name: str
    model_type: str
    accuracy: float
    features: List[str]
    target: str
    trained_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]
    model_data: Dict[str, Any]

@dataclass
class BusinessInsight:
    """Business insight definition."""
    insight_id: str
    title: str
    description: str
    category: str
    impact_level: str
    confidence_score: float
    data_sources: List[str]
    actionable: bool
    recommendations: List[str]
    created_at: datetime

class AdvancedAnalyticsService:
    """
    Advanced analytics service with comprehensive reporting and insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_service = DatabaseService(config)
        self.ai_service = AIService(config)
        self.predictive_models = {}
        self.analytics_cache = {}
        self.report_templates = {}
        self.insights_history = []
        
        # Analytics configurations
        self.analytics_configs = {
            "performance": {
                "metrics": ["execution_time", "success_rate", "throughput", "error_rate"],
                "visualizations": ["line", "bar", "heatmap"],
                "time_windows": ["1h", "24h", "7d", "30d", "90d"]
            },
            "business": {
                "metrics": ["revenue", "cost", "roi", "customer_satisfaction"],
                "visualizations": ["bar", "pie", "scatter"],
                "time_windows": ["24h", "7d", "30d", "90d", "1y"]
            },
            "operational": {
                "metrics": ["resource_usage", "capacity", "efficiency", "quality"],
                "visualizations": ["line", "bar", "heatmap"],
                "time_windows": ["1h", "24h", "7d", "30d"]
            }
        }
        
    async def initialize(self):
        """Initialize the analytics service."""
        try:
            await self.db_service.initialize()
            await self.ai_service.initialize()
            await self._load_predictive_models()
            await self._load_report_templates()
            await self._generate_initial_insights()
            logger.info("Advanced Analytics Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Analytics Service: {str(e)}")
            raise
            
    async def _load_predictive_models(self):
        """Load or train predictive models."""
        try:
            # Load historical data for training
            historical_data = await self._get_historical_analytics_data()
            
            if len(historical_data) > 100:  # Minimum data for training
                # Train performance prediction model
                await self._train_performance_model(historical_data)
                
                # Train business outcome model
                await self._train_business_model(historical_data)
                
                # Train operational efficiency model
                await self._train_operational_model(historical_data)
                
                logger.info(f"Loaded {len(self.predictive_models)} predictive models")
                
        except Exception as e:
            logger.error(f"Failed to load predictive models: {str(e)}")
            
    async def _get_historical_analytics_data(self) -> List[Dict[str, Any]]:
        """Get historical analytics data."""
        try:
            # Get data from last 6 months
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=180)
            
            # This would query the database for historical data
            # For now, return sample data
            return [
                {
                    "timestamp": datetime.utcnow() - timedelta(days=i),
                    "execution_time": 120 + np.random.normal(0, 20),
                    "success_rate": 0.9 + np.random.normal(0, 0.05),
                    "throughput": 100 + np.random.normal(0, 10),
                    "error_rate": 0.05 + np.random.normal(0, 0.02),
                    "resource_usage": 0.7 + np.random.normal(0, 0.1),
                    "revenue": 10000 + np.random.normal(0, 1000),
                    "cost": 5000 + np.random.normal(0, 500),
                    "customer_satisfaction": 4.2 + np.random.normal(0, 0.3)
                }
                for i in range(180)
            ]
        except Exception as e:
            logger.error(f"Failed to get historical analytics data: {str(e)}")
            return []
            
    async def _train_performance_model(self, data: List[Dict[str, Any]]):
        """Train performance prediction model."""
        try:
            df = pd.DataFrame(data)
            
            # Prepare features and target
            features = ["execution_time", "success_rate", "throughput", "error_rate"]
            target = "resource_usage"
            
            X = df[features].fillna(0).values
            y = df[target].fillna(0).values
            
            if len(X) > 0:
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Calculate performance metrics
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                r2 = 1 - (mse / np.var(y))
                
                self.predictive_models["performance"] = PredictiveModel(
                    model_id="perf_model_001",
                    name="Performance Prediction Model",
                    model_type="RandomForestRegressor",
                    accuracy=r2,
                    features=features,
                    target=target,
                    trained_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    performance_metrics={"mse": mse, "r2": r2},
                    model_data={"model": model}
                )
                
        except Exception as e:
            logger.error(f"Failed to train performance model: {str(e)}")
            
    async def _train_business_model(self, data: List[Dict[str, Any]]):
        """Train business outcome prediction model."""
        try:
            df = pd.DataFrame(data)
            
            # Prepare features and target
            features = ["execution_time", "success_rate", "resource_usage", "customer_satisfaction"]
            target = "revenue"
            
            X = df[features].fillna(0).values
            y = df[target].fillna(0).values
            
            if len(X) > 0:
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Calculate performance metrics
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                r2 = 1 - (mse / np.var(y))
                
                self.predictive_models["business"] = PredictiveModel(
                    model_id="business_model_001",
                    name="Business Outcome Prediction Model",
                    model_type="RandomForestRegressor",
                    accuracy=r2,
                    features=features,
                    target=target,
                    trained_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    performance_metrics={"mse": mse, "r2": r2},
                    model_data={"model": model}
                )
                
        except Exception as e:
            logger.error(f"Failed to train business model: {str(e)}")
            
    async def _train_operational_model(self, data: List[Dict[str, Any]]):
        """Train operational efficiency model."""
        try:
            df = pd.DataFrame(data)
            
            # Prepare features and target
            features = ["execution_time", "success_rate", "throughput", "error_rate"]
            target = "resource_usage"
            
            X = df[features].fillna(0).values
            y = df[target].fillna(0).values
            
            if len(X) > 0:
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Calculate performance metrics
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                r2 = 1 - (mse / np.var(y))
                
                self.predictive_models["operational"] = PredictiveModel(
                    model_id="op_model_001",
                    name="Operational Efficiency Model",
                    model_type="RandomForestRegressor",
                    accuracy=r2,
                    features=features,
                    target=target,
                    trained_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    performance_metrics={"mse": mse, "r2": r2},
                    model_data={"model": model}
                )
                
        except Exception as e:
            logger.error(f"Failed to train operational model: {str(e)}")
            
    async def _load_report_templates(self):
        """Load report templates."""
        try:
            self.report_templates = {
                "performance_dashboard": {
                    "title": "Performance Dashboard",
                    "description": "Comprehensive performance metrics dashboard",
                    "metrics": ["execution_time", "success_rate", "throughput", "error_rate"],
                    "visualizations": ["line", "bar", "heatmap"],
                    "time_window": "24h"
                },
                "business_report": {
                    "title": "Business Intelligence Report",
                    "description": "Business metrics and insights report",
                    "metrics": ["revenue", "cost", "roi", "customer_satisfaction"],
                    "visualizations": ["bar", "pie", "scatter"],
                    "time_window": "30d"
                },
                "operational_report": {
                    "title": "Operational Efficiency Report",
                    "description": "Operational metrics and efficiency analysis",
                    "metrics": ["resource_usage", "capacity", "efficiency", "quality"],
                    "visualizations": ["line", "bar", "heatmap"],
                    "time_window": "7d"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load report templates: {str(e)}")
            
    async def _generate_initial_insights(self):
        """Generate initial business insights."""
        try:
            # Get current system state
            current_metrics = await self._get_current_metrics()
            
            # Generate insights based on current state
            insights = []
            
            # Performance insights
            if current_metrics.get("success_rate", 0) < 0.9:
                insights.append(BusinessInsight(
                    insight_id="insight_001",
                    title="Low Success Rate Detected",
                    description="System success rate is below optimal threshold",
                    category="performance",
                    impact_level="high",
                    confidence_score=0.9,
                    data_sources=["workflow_executions", "error_logs"],
                    actionable=True,
                    recommendations=["Review error patterns", "Implement additional error handling", "Optimize workflow steps"],
                    created_at=datetime.utcnow()
                ))
                
            # Resource insights
            if current_metrics.get("resource_usage", 0) > 0.8:
                insights.append(BusinessInsight(
                    insight_id="insight_002",
                    title="High Resource Usage",
                    description="System resource usage is above optimal threshold",
                    category="operational",
                    impact_level="medium",
                    confidence_score=0.8,
                    data_sources=["system_metrics", "resource_monitoring"],
                    actionable=True,
                    recommendations=["Scale resources", "Optimize resource allocation", "Implement load balancing"],
                    created_at=datetime.utcnow()
                ))
                
            # Business insights
            if current_metrics.get("revenue", 0) > current_metrics.get("cost", 0) * 2:
                insights.append(BusinessInsight(
                    insight_id="insight_003",
                    title="Strong ROI Performance",
                    description="System is generating strong return on investment",
                    category="business",
                    impact_level="positive",
                    confidence_score=0.95,
                    data_sources=["financial_metrics", "business_analytics"],
                    actionable=False,
                    recommendations=["Continue current strategy", "Consider scaling operations"],
                    created_at=datetime.utcnow()
                ))
                
            self.insights_history.extend(insights)
            
        except Exception as e:
            logger.error(f"Failed to generate initial insights: {str(e)}")
            
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            # This would query the database for current metrics
            # For now, return sample data
            return {
                "execution_time": 120.5,
                "success_rate": 0.92,
                "throughput": 95.3,
                "error_rate": 0.08,
                "resource_usage": 0.75,
                "revenue": 12000,
                "cost": 6000,
                "customer_satisfaction": 4.3
            }
        except Exception as e:
            logger.error(f"Failed to get current metrics: {str(e)}")
            return {}
            
    async def generate_analytics_report(
        self, 
        report_type: ReportType, 
        analytics_type: AnalyticsType = AnalyticsType.DESCRIPTIVE,
        time_window: str = "24h",
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        try:
            # Get data for the specified time window
            data = await self._get_data_for_time_window(time_window)
            
            # Generate metrics
            metrics = await self._generate_metrics(data, report_type)
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(data, report_type)
            
            # Generate insights
            insights = await self._generate_insights(data, report_type, analytics_type)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(insights, report_type)
            
            # Create report
            report = AnalyticsReport(
                report_id=f"report_{report_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                title=f"{report_type.value.title()} Analytics Report",
                description=f"Comprehensive {report_type.value} analytics report",
                report_type=report_type,
                analytics_type=analytics_type,
                metrics=metrics,
                visualizations=visualizations,
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.utcnow(),
                generated_by="system",
                parameters=parameters or {}
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {str(e)}")
            raise
            
    async def _get_data_for_time_window(self, time_window: str) -> List[Dict[str, Any]]:
        """Get data for specified time window."""
        try:
            # Calculate time range
            now = datetime.utcnow()
            if time_window == "1h":
                start_time = now - timedelta(hours=1)
            elif time_window == "24h":
                start_time = now - timedelta(days=1)
            elif time_window == "7d":
                start_time = now - timedelta(days=7)
            elif time_window == "30d":
                start_time = now - timedelta(days=30)
            elif time_window == "90d":
                start_time = now - timedelta(days=90)
            else:
                start_time = now - timedelta(days=1)
                
            # This would query the database for data in the time range
            # For now, return sample data
            return [
                {
                    "timestamp": start_time + timedelta(minutes=i),
                    "execution_time": 120 + np.random.normal(0, 20),
                    "success_rate": 0.9 + np.random.normal(0, 0.05),
                    "throughput": 100 + np.random.normal(0, 10),
                    "error_rate": 0.05 + np.random.normal(0, 0.02),
                    "resource_usage": 0.7 + np.random.normal(0, 0.1),
                    "revenue": 10000 + np.random.normal(0, 1000),
                    "cost": 5000 + np.random.normal(0, 500),
                    "customer_satisfaction": 4.2 + np.random.normal(0, 0.3)
                }
                for i in range(0, int((now - start_time).total_seconds() / 60), 5)
            ]
        except Exception as e:
            logger.error(f"Failed to get data for time window: {str(e)}")
            return []
            
    async def _generate_metrics(self, data: List[Dict[str, Any]], report_type: ReportType) -> List[AnalyticsMetric]:
        """Generate metrics for the report."""
        try:
            if not data:
                return []
                
            df = pd.DataFrame(data)
            metrics = []
            
            # Performance metrics
            if report_type in [ReportType.PERFORMANCE, ReportType.OPERATIONAL]:
                metrics.extend([
                    AnalyticsMetric(
                        metric_id="execution_time_avg",
                        name="Average Execution Time",
                        description="Average workflow execution time",
                        value=float(df["execution_time"].mean()),
                        unit="seconds",
                        category="performance",
                        timestamp=datetime.utcnow(),
                        metadata={"min": float(df["execution_time"].min()), "max": float(df["execution_time"].max())}
                    ),
                    AnalyticsMetric(
                        metric_id="success_rate_avg",
                        name="Average Success Rate",
                        description="Average workflow success rate",
                        value=float(df["success_rate"].mean()),
                        unit="percentage",
                        category="performance",
                        timestamp=datetime.utcnow(),
                        metadata={"min": float(df["success_rate"].min()), "max": float(df["success_rate"].max())}
                    ),
                    AnalyticsMetric(
                        metric_id="throughput_avg",
                        name="Average Throughput",
                        description="Average system throughput",
                        value=float(df["throughput"].mean()),
                        unit="requests/hour",
                        category="performance",
                        timestamp=datetime.utcnow(),
                        metadata={"min": float(df["throughput"].min()), "max": float(df["throughput"].max())}
                    )
                ])
                
            # Business metrics
            if report_type in [ReportType.BUSINESS, ReportType.FINANCIAL]:
                metrics.extend([
                    AnalyticsMetric(
                        metric_id="revenue_total",
                        name="Total Revenue",
                        description="Total revenue generated",
                        value=float(df["revenue"].sum()),
                        unit="currency",
                        category="business",
                        timestamp=datetime.utcnow(),
                        metadata={"avg": float(df["revenue"].mean())}
                    ),
                    AnalyticsMetric(
                        metric_id="cost_total",
                        name="Total Cost",
                        description="Total operational cost",
                        value=float(df["cost"].sum()),
                        unit="currency",
                        category="business",
                        timestamp=datetime.utcnow(),
                        metadata={"avg": float(df["cost"].mean())}
                    ),
                    AnalyticsMetric(
                        metric_id="roi",
                        name="Return on Investment",
                        description="Return on investment ratio",
                        value=float(df["revenue"].sum() / df["cost"].sum()) if df["cost"].sum() > 0 else 0,
                        unit="ratio",
                        category="business",
                        timestamp=datetime.utcnow(),
                        metadata={}
                    )
                ])
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to generate metrics: {str(e)}")
            return []
            
    async def _generate_visualizations(self, data: List[Dict[str, Any]], report_type: ReportType) -> List[Dict[str, Any]]:
        """Generate visualizations for the report."""
        try:
            if not data:
                return []
                
            df = pd.DataFrame(data)
            visualizations = []
            
            # Time series visualization
            if len(df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["execution_time"],
                    mode="lines",
                    name="Execution Time",
                    line=dict(color="blue")
                ))
                fig.update_layout(
                    title="Execution Time Trend",
                    xaxis_title="Time",
                    yaxis_title="Execution Time (seconds)"
                )
                visualizations.append({
                    "type": "line",
                    "title": "Execution Time Trend",
                    "data": fig.to_dict(),
                    "description": "Trend of workflow execution time over time"
                })
                
            # Bar chart for success rate
            if "success_rate" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Success Rate"],
                    y=[df["success_rate"].mean()],
                    name="Average Success Rate",
                    marker_color="green"
                ))
                fig.update_layout(
                    title="Average Success Rate",
                    yaxis_title="Success Rate"
                )
                visualizations.append({
                    "type": "bar",
                    "title": "Average Success Rate",
                    "data": fig.to_dict(),
                    "description": "Average success rate across all workflows"
                })
                
            # Heatmap for resource usage
            if "resource_usage" in df.columns:
                # Create a simple heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=[df["resource_usage"].values],
                    colorscale="Viridis"
                ))
                fig.update_layout(
                    title="Resource Usage Heatmap",
                    xaxis_title="Time Points",
                    yaxis_title="Resource Usage"
                )
                visualizations.append({
                    "type": "heatmap",
                    "title": "Resource Usage Heatmap",
                    "data": fig.to_dict(),
                    "description": "Resource usage patterns over time"
                })
                
            return visualizations
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            return []
            
    async def _generate_insights(self, data: List[Dict[str, Any]], report_type: ReportType, analytics_type: AnalyticsType) -> List[str]:
        """Generate insights for the report."""
        try:
            if not data:
                return []
                
            df = pd.DataFrame(data)
            insights = []
            
            # Performance insights
            if report_type in [ReportType.PERFORMANCE, ReportType.OPERATIONAL]:
                avg_execution_time = df["execution_time"].mean()
                if avg_execution_time > 150:
                    insights.append(f"Average execution time ({avg_execution_time:.1f}s) is above optimal threshold. Consider workflow optimization.")
                    
                success_rate = df["success_rate"].mean()
                if success_rate < 0.9:
                    insights.append(f"Success rate ({success_rate:.1%}) is below target. Review error patterns and implement improvements.")
                    
                throughput = df["throughput"].mean()
                if throughput < 80:
                    insights.append(f"System throughput ({throughput:.1f} req/h) is below capacity. Consider scaling resources.")
                    
            # Business insights
            if report_type in [ReportType.BUSINESS, ReportType.FINANCIAL]:
                total_revenue = df["revenue"].sum()
                total_cost = df["cost"].sum()
                roi = total_revenue / total_cost if total_cost > 0 else 0
                
                if roi > 2:
                    insights.append(f"Strong ROI performance ({roi:.1f}x). System is generating significant value.")
                elif roi < 1.5:
                    insights.append(f"ROI ({roi:.1f}x) is below target. Consider cost optimization strategies.")
                    
                customer_satisfaction = df["customer_satisfaction"].mean()
                if customer_satisfaction > 4.0:
                    insights.append(f"High customer satisfaction ({customer_satisfaction:.1f}/5.0). Maintain current service quality.")
                elif customer_satisfaction < 3.5:
                    insights.append(f"Customer satisfaction ({customer_satisfaction:.1f}/5.0) needs improvement. Focus on user experience.")
                    
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return []
            
    async def _generate_recommendations(self, insights: List[str], report_type: ReportType) -> List[str]:
        """Generate recommendations based on insights."""
        try:
            recommendations = []
            
            # Performance recommendations
            if report_type in [ReportType.PERFORMANCE, ReportType.OPERATIONAL]:
                recommendations.extend([
                    "Implement parallel execution for long-running workflows",
                    "Add comprehensive error handling and recovery mechanisms",
                    "Optimize resource allocation based on usage patterns",
                    "Implement performance monitoring and alerting"
                ])
                
            # Business recommendations
            if report_type in [ReportType.BUSINESS, ReportType.FINANCIAL]:
                recommendations.extend([
                    "Focus on high-value workflow optimization",
                    "Implement cost tracking and optimization strategies",
                    "Enhance customer experience and satisfaction",
                    "Develop predictive analytics for business planning"
                ])
                
            # Operational recommendations
            if report_type in [ReportType.OPERATIONAL, ReportType.SYSTEM]:
                recommendations.extend([
                    "Implement automated scaling based on demand",
                    "Enhance system monitoring and alerting",
                    "Optimize database queries and caching strategies",
                    "Implement disaster recovery and backup procedures"
                ])
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return []
            
    async def predict_future_performance(self, time_horizon: str = "7d") -> Dict[str, Any]:
        """Predict future performance using ML models."""
        try:
            if "performance" not in self.predictive_models:
                raise ValueError("Performance prediction model not available")
                
            model = self.predictive_models["performance"]
            model_data = model.model_data["model"]
            
            # Get recent data for prediction
            recent_data = await self._get_data_for_time_window("24h")
            if not recent_data:
                raise ValueError("No recent data available for prediction")
                
            df = pd.DataFrame(recent_data)
            
            # Prepare features for prediction
            features = model.features
            X = df[features].fillna(0).values
            
            # Make predictions
            predictions = model_data.predict(X)
            
            # Calculate confidence intervals
            confidence_intervals = []
            for pred in predictions:
                ci_lower = pred * 0.9
                ci_upper = pred * 1.1
                confidence_intervals.append((ci_lower, ci_upper))
                
            return {
                "predictions": predictions.tolist(),
                "confidence_intervals": confidence_intervals,
                "model_accuracy": model.accuracy,
                "time_horizon": time_horizon,
                "predicted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict future performance: {str(e)}")
            raise
            
    async def get_business_insights(self, category: Optional[str] = None) -> List[BusinessInsight]:
        """Get business insights."""
        try:
            insights = self.insights_history
            
            if category:
                insights = [insight for insight in insights if insight.category == category]
                
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get business insights: {str(e)}")
            return []
            
    async def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom analytics dashboard."""
        try:
            dashboard_id = f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate dashboard data
            dashboard_data = {
                "dashboard_id": dashboard_id,
                "title": dashboard_config.get("title", "Custom Dashboard"),
                "description": dashboard_config.get("description", "Custom analytics dashboard"),
                "widgets": [],
                "layout": dashboard_config.get("layout", "grid"),
                "created_at": datetime.utcnow().isoformat(),
                "created_by": dashboard_config.get("created_by", "system")
            }
            
            # Generate widgets based on configuration
            widgets_config = dashboard_config.get("widgets", [])
            for widget_config in widgets_config:
                widget = await self._generate_dashboard_widget(widget_config)
                dashboard_data["widgets"].append(widget)
                
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to create custom dashboard: {str(e)}")
            raise
            
    async def _generate_dashboard_widget(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard widget."""
        try:
            widget_type = widget_config.get("type", "metric")
            widget_title = widget_config.get("title", "Widget")
            
            if widget_type == "metric":
                # Generate metric widget
                data = await self._get_data_for_time_window(widget_config.get("time_window", "24h"))
                if data:
                    df = pd.DataFrame(data)
                    metric_name = widget_config.get("metric", "execution_time")
                    if metric_name in df.columns:
                        value = float(df[metric_name].mean())
                        widget = {
                            "type": "metric",
                            "title": widget_title,
                            "value": value,
                            "unit": widget_config.get("unit", ""),
                            "trend": "stable"  # Simplified trend calculation
                        }
                    else:
                        widget = {
                            "type": "metric",
                            "title": widget_title,
                            "value": 0,
                            "unit": widget_config.get("unit", ""),
                            "trend": "stable"
                        }
                else:
                    widget = {
                        "type": "metric",
                        "title": widget_title,
                        "value": 0,
                        "unit": widget_config.get("unit", ""),
                        "trend": "stable"
                    }
                    
            elif widget_type == "chart":
                # Generate chart widget
                data = await self._get_data_for_time_window(widget_config.get("time_window", "24h"))
                if data:
                    df = pd.DataFrame(data)
                    chart_type = widget_config.get("chart_type", "line")
                    
                    if chart_type == "line" and len(df) > 1:
                        fig = go.Figure()
                        metric_name = widget_config.get("metric", "execution_time")
                        if metric_name in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df["timestamp"],
                                y=df[metric_name],
                                mode="lines",
                                name=metric_name
                            ))
                            
                        widget = {
                            "type": "chart",
                            "title": widget_title,
                            "chart_type": chart_type,
                            "data": fig.to_dict()
                        }
                    else:
                        widget = {
                            "type": "chart",
                            "title": widget_title,
                            "chart_type": chart_type,
                            "data": {}
                        }
                else:
                    widget = {
                        "type": "chart",
                        "title": widget_title,
                        "chart_type": widget_config.get("chart_type", "line"),
                        "data": {}
                    }
                    
            else:
                # Default widget
                widget = {
                    "type": widget_type,
                    "title": widget_title,
                    "data": {}
                }
                
            return widget
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard widget: {str(e)}")
            return {
                "type": "error",
                "title": widget_config.get("title", "Widget"),
                "error": str(e)
            }
            
    async def export_analytics_data(self, format: str = "json", filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export analytics data in specified format."""
        try:
            # Get data based on filters
            time_window = filters.get("time_window", "24h") if filters else "24h"
            data = await self._get_data_for_time_window(time_window)
            
            if format == "json":
                return {
                    "format": "json",
                    "data": data,
                    "exported_at": datetime.utcnow().isoformat(),
                    "record_count": len(data)
                }
            elif format == "csv":
                # Convert to CSV format
                df = pd.DataFrame(data)
                csv_data = df.to_csv(index=False)
                return {
                    "format": "csv",
                    "data": csv_data,
                    "exported_at": datetime.utcnow().isoformat(),
                    "record_count": len(data)
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export analytics data: {str(e)}")
            raise
            
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics service summary."""
        try:
            return {
                "total_reports_generated": len(self.analytics_cache),
                "predictive_models_available": len(self.predictive_models),
                "insights_generated": len(self.insights_history),
                "report_templates_available": len(self.report_templates),
                "service_status": "active",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return {}




























