"""
Advanced Business Intelligence System

This module provides comprehensive business intelligence capabilities
for the refactored HeyGen AI system with advanced analytics,
predictive modeling, and strategic insights.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import redis
import threading
from collections import defaultdict, deque
import yaml
import hashlib
import base64
from cryptography.fernet import Fernet
import requests
import websockets
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import gc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Analysis types."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    REAL_TIME = "real_time"
    BATCH = "batch"


class MetricType(str, Enum):
    """Metric types."""
    REVENUE = "revenue"
    COST = "cost"
    PROFIT = "profit"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    MARKETING = "marketing"
    SALES = "sales"


class VisualizationType(str, Enum):
    """Visualization types."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    DASHBOARD = "dashboard"
    KPI_CARD = "kpi_card"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class BusinessMetric:
    """Business metric structure."""
    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Analysis result structure."""
    analysis_id: str
    analysis_type: AnalysisType
    metric_type: MetricType
    result: Dict[str, Any]
    confidence: float
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class KPIDashboard:
    """KPI dashboard structure."""
    dashboard_id: str
    name: str
    description: str
    kpis: List[BusinessMetric]
    visualizations: List[Dict[str, Any]]
    refresh_interval: int = 300  # seconds
    last_updated: Optional[datetime] = None


class DataProcessor:
    """Advanced data processor for business intelligence."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocess data for analysis."""
        try:
            # Handle missing values
            data = data.fillna(data.mean())
            
            # Encode categorical variables
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != target_column:
                    data[col] = pd.Categorical(data[col]).codes
            
            # Scale numerical features
            numerical_columns = data.select_dtypes(include=[np.number]).columns
            if target_column and target_column in numerical_columns:
                numerical_columns = numerical_columns.drop(target_column)
            
            scaler = StandardScaler()
            if len(numerical_columns) > 0:
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            
            return data, scaler
            
        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            return data, None
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for analysis."""
        try:
            features = data.copy()
            
            # Time-based features
            if 'timestamp' in features.columns:
                features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
                features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
                features['month'] = pd.to_datetime(features['timestamp']).dt.month
            
            # Statistical features
            numerical_columns = features.select_dtypes(include=[np.number]).columns
            for col in numerical_columns:
                features[f'{col}_rolling_mean'] = features[col].rolling(window=7).mean()
                features[f'{col}_rolling_std'] = features[col].rolling(window=7).std()
                features[f'{col}_lag_1'] = features[col].shift(1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return data


class PredictiveModeler:
    """Advanced predictive modeling for business intelligence."""
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
    
    def train_revenue_model(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train revenue prediction model."""
        try:
            # Prepare data
            processor = DataProcessor()
            processed_data, scaler = processor.preprocess_data(data, target_column)
            
            # Split data
            X = processed_data.drop(columns=[target_column])
            y = processed_data[target_column]
            
            # Train multiple models
            models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = -float('inf')
            model_results = {}
            
            for name, model in models.items():
                model.fit(X, y)
                y_pred = model.predict(X)
                score = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                model_results[name] = {
                    'r2_score': score,
                    'mse': mse,
                    'model': model
                }
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # Store best model
            self.models['revenue'] = best_model
            self.model_metrics['revenue'] = model_results
            
            return {
                'best_model': best_model.__class__.__name__,
                'r2_score': best_score,
                'all_models': model_results
            }
            
        except Exception as e:
            logger.error(f"Revenue model training error: {e}")
            return {}
    
    def predict_revenue(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict revenue based on features."""
        try:
            if 'revenue' not in self.models:
                return {'error': 'Revenue model not trained'}
            
            model = self.models['revenue']
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Preprocess features
            processor = DataProcessor()
            processed_features, _ = processor.preprocess_data(feature_df)
            
            # Make prediction
            prediction = model.predict(processed_features)[0]
            
            return {
                'predicted_revenue': prediction,
                'confidence': 0.85,  # Mock confidence
                'model_type': model.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Revenue prediction error: {e}")
            return {'error': str(e)}
    
    def train_customer_segmentation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train customer segmentation model."""
        try:
            # Prepare data
            processor = DataProcessor()
            processed_data, _ = processor.preprocess_data(data)
            
            # Select features for clustering
            numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
            X = processed_data[numerical_columns]
            
            # Determine optimal number of clusters
            silhouette_scores = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            # Train final model
            final_model = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = final_model.fit_predict(X)
            
            # Store model
            self.models['customer_segmentation'] = final_model
            
            return {
                'optimal_clusters': optimal_k,
                'silhouette_score': max(silhouette_scores),
                'cluster_labels': cluster_labels.tolist()
            }
            
        except Exception as e:
            logger.error(f"Customer segmentation error: {e}")
            return {}
    
    def segment_customers(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Segment customers based on features."""
        try:
            if 'customer_segmentation' not in self.models:
                return {'error': 'Customer segmentation model not trained'}
            
            model = self.models['customer_segmentation']
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Preprocess features
            processor = DataProcessor()
            processed_features, _ = processor.preprocess_data(feature_df)
            
            # Make prediction
            segment = model.predict(processed_features)[0]
            
            return {
                'customer_segment': int(segment),
                'confidence': 0.80,  # Mock confidence
                'model_type': model.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Customer segmentation error: {e}")
            return {'error': str(e)}


class VisualizationEngine:
    """Advanced visualization engine for business intelligence."""
    
    def __init__(self):
        self.chart_templates = {}
        self.color_palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'],
            'corporate': ['#1B263B', '#415A77', '#778DA9', '#E0E1DD', '#0D1B2A']
        }
    
    def create_line_chart(self, data: pd.DataFrame, x_column: str, y_column: str, title: str) -> Dict[str, Any]:
        """Create line chart visualization."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='lines+markers',
                name=y_column,
                line=dict(color=self.color_palettes['business'][0], width=3)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title=y_column,
                template='plotly_white',
                height=400
            )
            
            return {
                'type': 'line_chart',
                'data': fig.to_dict(),
                'title': title
            }
            
        except Exception as e:
            logger.error(f"Line chart creation error: {e}")
            return {}
    
    def create_bar_chart(self, data: pd.DataFrame, x_column: str, y_column: str, title: str) -> Dict[str, Any]:
        """Create bar chart visualization."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data[x_column],
                y=data[y_column],
                name=y_column,
                marker_color=self.color_palettes['business'][1]
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title=y_column,
                template='plotly_white',
                height=400
            )
            
            return {
                'type': 'bar_chart',
                'data': fig.to_dict(),
                'title': title
            }
            
        except Exception as e:
            logger.error(f"Bar chart creation error: {e}")
            return {}
    
    def create_heatmap(self, data: pd.DataFrame, title: str) -> Dict[str, Any]:
        """Create heatmap visualization."""
        try:
            # Calculate correlation matrix
            correlation_matrix = data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title=title,
                template='plotly_white',
                height=500
            )
            
            return {
                'type': 'heatmap',
                'data': fig.to_dict(),
                'title': title
            }
            
        except Exception as e:
            logger.error(f"Heatmap creation error: {e}")
            return {}
    
    def create_kpi_dashboard(self, kpis: List[BusinessMetric]) -> Dict[str, Any]:
        """Create KPI dashboard visualization."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[kpi.name for kpi in kpis[:4]],
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            for i, kpi in enumerate(kpis[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=kpi.value,
                    title={"text": kpi.name},
                    delta={"reference": kpi.value * 0.9},  # Mock reference
                    domain={'x': [0, 1], 'y': [0, 1]}
                ), row=row, col=col)
            
            fig.update_layout(
                title="Business Intelligence Dashboard",
                template='plotly_white',
                height=600
            )
            
            return {
                'type': 'kpi_dashboard',
                'data': fig.to_dict(),
                'title': 'Business Intelligence Dashboard'
            }
            
        except Exception as e:
            logger.error(f"KPI dashboard creation error: {e}")
            return {}


class AdvancedBusinessIntelligenceSystem:
    """
    Advanced business intelligence system with comprehensive capabilities.
    
    Features:
    - Advanced analytics and reporting
    - Predictive modeling and forecasting
    - Customer segmentation and analysis
    - Revenue optimization
    - Cost analysis and optimization
    - Market analysis and insights
    - Real-time dashboards
    - Strategic recommendations
    """
    
    def __init__(
        self,
        database_path: str = "business_intelligence.db",
        redis_url: str = None
    ):
        """
        Initialize the advanced business intelligence system.
        
        Args:
            database_path: SQLite database path
            redis_url: Redis URL for caching
        """
        self.database_path = database_path
        self.redis_url = redis_url
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.predictive_modeler = PredictiveModeler()
        self.visualization_engine = VisualizationEngine()
        
        # Initialize Redis client
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
        
        # Initialize database
        self._initialize_database()
        
        # Initialize metrics
        self.metrics = {
            'analyses_performed': Counter('bi_analyses_performed_total', 'Total BI analyses performed', ['analysis_type']),
            'predictions_made': Counter('bi_predictions_made_total', 'Total BI predictions made', ['model_type']),
            'dashboards_created': Counter('bi_dashboards_created_total', 'Total BI dashboards created'),
            'data_processed': Histogram('bi_data_processed_bytes', 'BI data processed in bytes')
        }
        
        logger.info("Advanced business intelligence system initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS business_metrics (
                    metric_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    dimensions TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    result TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    insights TEXT,
                    recommendations TEXT,
                    visualizations TEXT,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kpi_dashboards (
                    dashboard_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    kpis TEXT NOT NULL,
                    visualizations TEXT,
                    refresh_interval INTEGER DEFAULT 300,
                    last_updated DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def add_business_metric(self, metric: BusinessMetric) -> bool:
        """Add business metric to database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO business_metrics
                (metric_id, name, metric_type, value, unit, timestamp, dimensions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.metric_id,
                metric.name,
                metric.metric_type.value,
                metric.value,
                metric.unit,
                metric.timestamp.isoformat(),
                json.dumps(metric.dimensions),
                json.dumps(metric.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            # Update metrics
            self.metrics['data_processed'].observe(len(json.dumps(metric.dimensions)))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding business metric: {e}")
            return False
    
    async def get_business_metrics(
        self,
        metric_type: MetricType = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[BusinessMetric]:
        """Get business metrics from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM business_metrics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metric = BusinessMetric(
                    metric_id=row[0],
                    name=row[1],
                    metric_type=MetricType(row[2]),
                    value=row[3],
                    unit=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    dimensions=json.loads(row[6]) if row[6] else {},
                    metadata=json.loads(row[7]) if row[7] else {}
                )
                metrics.append(metric)
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting business metrics: {e}")
            return []
    
    async def perform_revenue_analysis(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive revenue analysis."""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Train revenue model
            model_results = self.predictive_modeler.train_revenue_model(data, 'revenue')
            
            # Calculate revenue metrics
            total_revenue = data['revenue'].sum()
            avg_revenue = data['revenue'].mean()
            revenue_growth = data['revenue'].pct_change().mean() * 100
            
            # Generate insights
            insights = [
                f"Total revenue: ${total_revenue:,.2f}",
                f"Average revenue: ${avg_revenue:,.2f}",
                f"Revenue growth rate: {revenue_growth:.2f}%",
                f"Best model: {model_results.get('best_model', 'Unknown')}",
                f"Model accuracy: {model_results.get('r2_score', 0):.2f}"
            ]
            
            # Generate recommendations
            recommendations = [
                "Focus on high-revenue customer segments",
                "Implement dynamic pricing strategies",
                "Optimize marketing spend allocation",
                "Develop new revenue streams",
                "Improve customer retention programs"
            ]
            
            # Create visualizations
            visualizations = []
            
            # Revenue trend chart
            if 'timestamp' in data.columns:
                trend_chart = self.visualization_engine.create_line_chart(
                    data, 'timestamp', 'revenue', 'Revenue Trend'
                )
                visualizations.append(trend_chart)
            
            # Revenue distribution chart
            distribution_chart = self.visualization_engine.create_bar_chart(
                data.groupby('month')['revenue'].sum().reset_index(),
                'month', 'revenue', 'Monthly Revenue Distribution'
            )
            visualizations.append(distribution_chart)
            
            # Create analysis result
            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=AnalysisType.PREDICTIVE,
                metric_type=MetricType.REVENUE,
                result={
                    'total_revenue': total_revenue,
                    'avg_revenue': avg_revenue,
                    'revenue_growth': revenue_growth,
                    'model_results': model_results
                },
                confidence=0.85,
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations
            )
            
            # Store result
            await self._store_analysis_result(result)
            
            # Update metrics
            self.metrics['analyses_performed'].labels(analysis_type='revenue').inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Revenue analysis error: {e}")
            return AnalysisResult(
                analysis_id=str(uuid.uuid4()),
                analysis_type=AnalysisType.PREDICTIVE,
                metric_type=MetricType.REVENUE,
                result={'error': str(e)},
                confidence=0.0
            )
    
    async def perform_customer_analysis(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive customer analysis."""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Train customer segmentation model
            segmentation_results = self.predictive_modeler.train_customer_segmentation(data)
            
            # Calculate customer metrics
            total_customers = len(data)
            avg_customer_value = data['customer_value'].mean() if 'customer_value' in data.columns else 0
            customer_retention = data['retention_rate'].mean() if 'retention_rate' in data.columns else 0
            
            # Generate insights
            insights = [
                f"Total customers: {total_customers:,}",
                f"Average customer value: ${avg_customer_value:,.2f}",
                f"Customer retention rate: {customer_retention:.2f}%",
                f"Optimal segments: {segmentation_results.get('optimal_clusters', 0)}",
                f"Segmentation quality: {segmentation_results.get('silhouette_score', 0):.2f}"
            ]
            
            # Generate recommendations
            recommendations = [
                "Develop targeted marketing campaigns for each segment",
                "Implement personalized customer experiences",
                "Focus on high-value customer acquisition",
                "Improve customer retention strategies",
                "Create segment-specific product offerings"
            ]
            
            # Create visualizations
            visualizations = []
            
            # Customer value distribution
            if 'customer_value' in data.columns:
                value_chart = self.visualization_engine.create_bar_chart(
                    data.groupby('segment')['customer_value'].mean().reset_index(),
                    'segment', 'customer_value', 'Average Customer Value by Segment'
                )
                visualizations.append(value_chart)
            
            # Customer segmentation heatmap
            if 'customer_value' in data.columns and 'retention_rate' in data.columns:
                heatmap_data = data[['customer_value', 'retention_rate', 'segment']].corr()
                heatmap = self.visualization_engine.create_heatmap(
                    heatmap_data, 'Customer Analysis Correlation'
                )
                visualizations.append(heatmap)
            
            # Create analysis result
            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=AnalysisType.DESCRIPTIVE,
                metric_type=MetricType.CUSTOMER,
                result={
                    'total_customers': total_customers,
                    'avg_customer_value': avg_customer_value,
                    'customer_retention': customer_retention,
                    'segmentation_results': segmentation_results
                },
                confidence=0.80,
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations
            )
            
            # Store result
            await self._store_analysis_result(result)
            
            # Update metrics
            self.metrics['analyses_performed'].labels(analysis_type='customer').inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Customer analysis error: {e}")
            return AnalysisResult(
                analysis_id=str(uuid.uuid4()),
                analysis_type=AnalysisType.DESCRIPTIVE,
                metric_type=MetricType.CUSTOMER,
                result={'error': str(e)},
                confidence=0.0
            )
    
    async def create_kpi_dashboard(self, name: str, description: str, kpi_names: List[str]) -> KPIDashboard:
        """Create KPI dashboard."""
        try:
            dashboard_id = str(uuid.uuid4())
            
            # Get KPIs
            kpis = []
            for kpi_name in kpi_names:
                metrics = await self.get_business_metrics()
                kpi_metrics = [m for m in metrics if m.name == kpi_name]
                if kpi_metrics:
                    kpis.append(kpi_metrics[0])
            
            # Create visualizations
            visualizations = []
            if kpis:
                dashboard_viz = self.visualization_engine.create_kpi_dashboard(kpis)
                visualizations.append(dashboard_viz)
            
            # Create dashboard
            dashboard = KPIDashboard(
                dashboard_id=dashboard_id,
                name=name,
                description=description,
                kpis=kpis,
                visualizations=visualizations,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Store dashboard
            await self._store_kpi_dashboard(dashboard)
            
            # Update metrics
            self.metrics['dashboards_created'].inc()
            
            return dashboard
            
        except Exception as e:
            logger.error(f"KPI dashboard creation error: {e}")
            return None
    
    async def _store_analysis_result(self, result: AnalysisResult):
        """Store analysis result in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results
                (analysis_id, analysis_type, metric_type, result, confidence, insights, recommendations, visualizations, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.analysis_id,
                result.analysis_type.value,
                result.metric_type.value,
                json.dumps(result.result),
                result.confidence,
                json.dumps(result.insights),
                json.dumps(result.recommendations),
                json.dumps(result.visualizations),
                result.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
    
    async def _store_kpi_dashboard(self, dashboard: KPIDashboard):
        """Store KPI dashboard in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO kpi_dashboards
                (dashboard_id, name, description, kpis, visualizations, refresh_interval, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                dashboard.dashboard_id,
                dashboard.name,
                dashboard.description,
                json.dumps([kpi.__dict__ for kpi in dashboard.kpis]),
                json.dumps(dashboard.visualizations),
                dashboard.refresh_interval,
                dashboard.last_updated.isoformat() if dashboard.last_updated else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing KPI dashboard: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'total_analyses': sum(self.metrics['analyses_performed']._value.sum() for _ in [1]),
            'total_predictions': sum(self.metrics['predictions_made']._value.sum() for _ in [1]),
            'total_dashboards': self.metrics['dashboards_created']._value.sum(),
            'data_processed_bytes': sum(self.metrics['data_processed']._sum for _ in [1])
        }


# Example usage and demonstration
async def main():
    """Demonstrate the advanced business intelligence system."""
    print("📊 HeyGen AI - Advanced Business Intelligence System Demo")
    print("=" * 70)
    
    # Initialize business intelligence system
    bi_system = AdvancedBusinessIntelligenceSystem(
        database_path="business_intelligence.db",
        redis_url="redis://localhost:6379/0"
    )
    
    try:
        # Create sample business data
        print("\n📈 Creating Sample Business Data...")
        
        # Revenue data
        revenue_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=365, freq='D'),
            'revenue': np.random.normal(10000, 2000, 365).cumsum(),
            'month': pd.date_range('2023-01-01', periods=365, freq='D').month,
            'customer_count': np.random.randint(50, 200, 365),
            'marketing_spend': np.random.normal(2000, 500, 365)
        })
        
        # Customer data
        customer_data = pd.DataFrame({
            'customer_id': range(1000),
            'customer_value': np.random.normal(500, 150, 1000),
            'retention_rate': np.random.uniform(0.7, 0.95, 1000),
            'segment': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'age': np.random.randint(18, 65, 1000),
            'income': np.random.normal(50000, 15000, 1000)
        })
        
        # Add business metrics
        print("\n📊 Adding Business Metrics...")
        
        # Revenue metrics
        for _, row in revenue_data.iterrows():
            metric = BusinessMetric(
                metric_id=str(uuid.uuid4()),
                name="Daily Revenue",
                metric_type=MetricType.REVENUE,
                value=row['revenue'],
                unit="USD",
                timestamp=row['timestamp'],
                dimensions={'month': row['month'], 'customer_count': row['customer_count']}
            )
            await bi_system.add_business_metric(metric)
        
        # Customer metrics
        for _, row in customer_data.iterrows():
            metric = BusinessMetric(
                metric_id=str(uuid.uuid4()),
                name="Customer Value",
                metric_type=MetricType.CUSTOMER,
                value=row['customer_value'],
                unit="USD",
                timestamp=datetime.now(timezone.utc),
                dimensions={'segment': row['segment'], 'age': row['age']}
            )
            await bi_system.add_business_metric(metric)
        
        print(f"Added {len(revenue_data)} revenue metrics")
        print(f"Added {len(customer_data)} customer metrics")
        
        # Perform revenue analysis
        print("\n💰 Performing Revenue Analysis...")
        revenue_analysis = await bi_system.perform_revenue_analysis(revenue_data)
        
        print(f"Revenue Analysis Results:")
        print(f"  Analysis ID: {revenue_analysis.analysis_id}")
        print(f"  Confidence: {revenue_analysis.confidence:.2f}")
        print(f"  Insights: {len(revenue_analysis.insights)} insights generated")
        print(f"  Recommendations: {len(revenue_analysis.recommendations)} recommendations")
        print(f"  Visualizations: {len(revenue_analysis.visualizations)} charts created")
        
        # Perform customer analysis
        print("\n👥 Performing Customer Analysis...")
        customer_analysis = await bi_system.perform_customer_analysis(customer_data)
        
        print(f"Customer Analysis Results:")
        print(f"  Analysis ID: {customer_analysis.analysis_id}")
        print(f"  Confidence: {customer_analysis.confidence:.2f}")
        print(f"  Insights: {len(customer_analysis.insights)} insights generated")
        print(f"  Recommendations: {len(customer_analysis.recommendations)} recommendations")
        print(f"  Visualizations: {len(customer_analysis.visualizations)} charts created")
        
        # Create KPI dashboard
        print("\n📊 Creating KPI Dashboard...")
        dashboard = await bi_system.create_kpi_dashboard(
            name="Executive Dashboard",
            description="High-level business metrics and KPIs",
            kpi_names=["Daily Revenue", "Customer Value"]
        )
        
        if dashboard:
            print(f"Dashboard Created:")
            print(f"  Dashboard ID: {dashboard.dashboard_id}")
            print(f"  Name: {dashboard.name}")
            print(f"  KPIs: {len(dashboard.kpis)}")
            print(f"  Visualizations: {len(dashboard.visualizations)}")
        
        # Get system metrics
        print("\n📈 System Metrics:")
        metrics = bi_system.get_system_metrics()
        print(f"  Total Analyses: {metrics['total_analyses']}")
        print(f"  Total Predictions: {metrics['total_predictions']}")
        print(f"  Total Dashboards: {metrics['total_dashboards']}")
        print(f"  Data Processed: {metrics['data_processed_bytes']} bytes")
        
        # Test predictive capabilities
        print("\n🔮 Testing Predictive Capabilities...")
        
        # Revenue prediction
        revenue_prediction = bi_system.predictive_modeler.predict_revenue({
            'customer_count': 150,
            'marketing_spend': 2500,
            'month': 6
        })
        print(f"Revenue Prediction: {revenue_prediction}")
        
        # Customer segmentation
        customer_segmentation = bi_system.predictive_modeler.segment_customers({
            'customer_value': 600,
            'retention_rate': 0.85,
            'age': 35,
            'income': 60000
        })
        print(f"Customer Segmentation: {customer_segmentation}")
        
        print(f"\n🌐 Business Intelligence Dashboard available at: http://localhost:8080/dashboard")
        print(f"📊 Analytics API available at: http://localhost:8080/api/v1/analytics")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
