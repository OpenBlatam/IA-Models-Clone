"""
Gamma App - Business Intelligence Service
Advanced business intelligence with predictive analytics, data visualization, and insights generation
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, deque
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import statsmodels.api as sm
from scipy import stats
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Analysis types"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    EXPLORATORY = "exploratory"

class DataSource(Enum):
    """Data source types"""
    DATABASE = "database"
    CSV = "csv"
    JSON = "json"
    API = "api"
    EXCEL = "excel"
    PARQUET = "parquet"

class VisualizationType(Enum):
    """Visualization types"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    DASHBOARD = "dashboard"
    KPI_CARD = "kpi_card"

@dataclass
class DataSource:
    """Data source definition"""
    source_id: str
    name: str
    source_type: DataSource
    connection_string: str
    query: Optional[str] = None
    file_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisRequest:
    """Analysis request definition"""
    request_id: str
    data_source_id: str
    analysis_type: AnalysisType
    target_variable: Optional[str] = None
    features: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime = None

@dataclass
class AnalysisResult:
    """Analysis result definition"""
    result_id: str
    request_id: str
    analysis_type: AnalysisType
    insights: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    predictions: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    confidence_score: float = 0.0
    created_at: datetime = None

@dataclass
class KPI:
    """KPI definition"""
    kpi_id: str
    name: str
    description: str
    value: float
    target: Optional[float] = None
    unit: str = ""
    trend: str = "stable"
    period: str = "monthly"
    category: str = "general"
    created_at: datetime = None

@dataclass
class Dashboard:
    """Dashboard definition"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None
    refresh_interval: int = 300
    is_public: bool = False
    created_at: datetime = None

class AdvancedBusinessIntelligenceService:
    """Advanced Business Intelligence Service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "business_intelligence.db")
        self.redis_client = None
        self.data_sources = {}
        self.analysis_requests = {}
        self.analysis_results = {}
        self.kpis = {}
        self.dashboards = {}
        self.cached_data = {}
        self.model_cache = {}
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_default_data_sources()
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize business intelligence database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create data sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    connection_string TEXT,
                    query TEXT,
                    file_path TEXT,
                    api_endpoint TEXT,
                    headers TEXT,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create analysis requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_requests (
                    request_id TEXT PRIMARY KEY,
                    data_source_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    target_variable TEXT,
                    features TEXT,
                    filters TEXT,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_source_id) REFERENCES data_sources (source_id)
                )
            """)
            
            # Create analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    result_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    insights TEXT NOT NULL,
                    visualizations TEXT NOT NULL,
                    predictions TEXT,
                    recommendations TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (request_id) REFERENCES analysis_requests (request_id)
                )
            """)
            
            # Create KPIs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kpis (
                    kpi_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    value REAL NOT NULL,
                    target REAL,
                    unit TEXT DEFAULT '',
                    trend TEXT DEFAULT 'stable',
                    period TEXT DEFAULT 'monthly',
                    category TEXT DEFAULT 'general',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create dashboards table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dashboards (
                    dashboard_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    widgets TEXT NOT NULL,
                    layout TEXT NOT NULL,
                    filters TEXT,
                    refresh_interval INTEGER DEFAULT 300,
                    is_public BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        logger.info("Business intelligence database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for business intelligence")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_default_data_sources(self):
        """Initialize default data sources"""
        
        # Sample data source
        sample_source = DataSource(
            source_id="sample_data_001",
            name="Sample Sales Data",
            source_type=DataSource.CSV,
            connection_string="",
            file_path="data/sample_sales.csv"
        )
        
        self.data_sources[sample_source.source_id] = sample_source
        
        # Create sample data if it doesn't exist
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        
        sample_data_path = Path("data/sample_sales.csv")
        sample_data_path.parent.mkdir(exist_ok=True)
        
        if not sample_data_path.exists():
            # Generate sample sales data
            np.random.seed(42)
            n_records = 1000
            
            data = {
                'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_records),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
                'sales_amount': np.random.normal(1000, 300, n_records),
                'quantity': np.random.randint(1, 10, n_records),
                'customer_age': np.random.randint(18, 80, n_records),
                'customer_gender': np.random.choice(['M', 'F'], n_records),
                'discount_percent': np.random.uniform(0, 0.3, n_records),
                'profit_margin': np.random.uniform(0.1, 0.4, n_records)
            }
            
            df = pd.DataFrame(data)
            df.to_csv(sample_data_path, index=False)
            logger.info(f"Sample data created at {sample_data_path}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        asyncio.create_task(self._data_refresh_task())
        asyncio.create_task(self._kpi_calculation_task())
        asyncio.create_task(self._cache_cleanup_task())
    
    async def add_data_source(
        self,
        name: str,
        source_type: DataSource,
        connection_string: str = None,
        query: str = None,
        file_path: str = None,
        api_endpoint: str = None,
        headers: Dict[str, str] = None,
        parameters: Dict[str, Any] = None
    ) -> DataSource:
        """Add new data source"""
        
        data_source = DataSource(
            source_id=str(uuid.uuid4()),
            name=name,
            source_type=source_type,
            connection_string=connection_string,
            query=query,
            file_path=file_path,
            api_endpoint=api_endpoint,
            headers=headers or {},
            parameters=parameters or {}
        )
        
        self.data_sources[data_source.source_id] = data_source
        await self._store_data_source(data_source)
        
        logger.info(f"Data source added: {data_source.source_id}")
        return data_source
    
    async def get_data(self, source_id: str, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get data from source"""
        
        try:
            # Check cache first
            cache_key = f"data_{source_id}_{hash(str(filters))}"
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)
            
            data_source = self.data_sources.get(source_id)
            if not data_source:
                raise ValueError(f"Data source {source_id} not found")
            
            # Load data based on source type
            if data_source.source_type == DataSource.CSV:
                df = pd.read_csv(data_source.file_path)
            elif data_source.source_type == DataSource.JSON:
                df = pd.read_json(data_source.file_path)
            elif data_source.source_type == DataSource.EXCEL:
                df = pd.read_excel(data_source.file_path)
            elif data_source.source_type == DataSource.PARQUET:
                df = pd.read_parquet(data_source.file_path)
            elif data_source.source_type == DataSource.DATABASE:
                # This would require actual database connection
                df = pd.DataFrame()  # Placeholder
            elif data_source.source_type == DataSource.API:
                # This would require actual API call
                df = pd.DataFrame()  # Placeholder
            
            # Apply filters
            if filters:
                for column, value in filters.items():
                    if column in df.columns:
                        if isinstance(value, list):
                            df = df[df[column].isin(value)]
                        else:
                            df = df[df[column] == value]
            
            # Cache the result
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour cache
                    df.to_json()
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame()
    
    async def perform_analysis(
        self,
        data_source_id: str,
        analysis_type: AnalysisType,
        target_variable: str = None,
        features: List[str] = None,
        filters: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None
    ) -> AnalysisResult:
        """Perform business intelligence analysis"""
        
        try:
            # Create analysis request
            request = AnalysisRequest(
                request_id=str(uuid.uuid4()),
                data_source_id=data_source_id,
                analysis_type=analysis_type,
                target_variable=target_variable,
                features=features or [],
                filters=filters or {},
                parameters=parameters or {},
                created_at=datetime.now()
            )
            
            self.analysis_requests[request.request_id] = request
            await self._store_analysis_request(request)
            
            # Get data
            df = await self.get_data(data_source_id, filters)
            if df.empty:
                raise ValueError("No data available for analysis")
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.DESCRIPTIVE:
                result = await self._perform_descriptive_analysis(df, request)
            elif analysis_type == AnalysisType.DIAGNOSTIC:
                result = await self._perform_diagnostic_analysis(df, request)
            elif analysis_type == AnalysisType.PREDICTIVE:
                result = await self._perform_predictive_analysis(df, request)
            elif analysis_type == AnalysisType.PRESCRIPTIVE:
                result = await self._perform_prescriptive_analysis(df, request)
            elif analysis_type == AnalysisType.EXPLORATORY:
                result = await self._perform_exploratory_analysis(df, request)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Store result
            self.analysis_results[result.result_id] = result
            await self._store_analysis_result(result)
            
            logger.info(f"Analysis completed: {result.result_id}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def _perform_descriptive_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
        """Perform descriptive analysis"""
        
        insights = []
        visualizations = []
        
        try:
            # Basic statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Summary statistics
            summary_stats = df.describe().to_dict()
            insights.append({
                "type": "summary_statistics",
                "title": "Data Summary",
                "content": f"Dataset contains {len(df)} records with {len(df.columns)} columns",
                "details": summary_stats
            })
            
            # Missing values analysis
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                insights.append({
                    "type": "data_quality",
                    "title": "Missing Values",
                    "content": f"Found {missing_values.sum()} missing values across {len(missing_values[missing_values > 0])} columns",
                    "details": missing_values[missing_values > 0].to_dict()
                })
            
            # Distribution analysis for numeric columns
            for column in numeric_columns[:5]:  # Limit to first 5 numeric columns
                if column in df.columns:
                    # Histogram
                    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                    visualizations.append({
                        "type": "histogram",
                        "title": f"{column} Distribution",
                        "data": fig.to_dict(),
                        "column": column
                    })
                    
                    # Box plot
                    fig = px.box(df, y=column, title=f"Box Plot of {column}")
                    visualizations.append({
                        "type": "box_plot",
                        "title": f"{column} Box Plot",
                        "data": fig.to_dict(),
                        "column": column
                    })
            
            # Categorical analysis
            for column in categorical_columns[:3]:  # Limit to first 3 categorical columns
                if column in df.columns:
                    value_counts = df[column].value_counts()
                    insights.append({
                        "type": "categorical_analysis",
                        "title": f"{column} Distribution",
                        "content": f"Most common value: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)",
                        "details": value_counts.head(10).to_dict()
                    })
                    
                    # Pie chart
                    fig = px.pie(
                        values=value_counts.head(10).values,
                        names=value_counts.head(10).index,
                        title=f"Distribution of {column}"
                    )
                    visualizations.append({
                        "type": "pie_chart",
                        "title": f"{column} Distribution",
                        "data": fig.to_dict(),
                        "column": column
                    })
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                fig = px.imshow(
                    correlation_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                visualizations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data": fig.to_dict(),
                    "columns": list(numeric_columns)
                })
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append({
                                "variables": [correlation_matrix.columns[i], correlation_matrix.columns[j]],
                                "correlation": corr_value
                            })
                
                if strong_correlations:
                    insights.append({
                        "type": "correlation_analysis",
                        "title": "Strong Correlations",
                        "content": f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)",
                        "details": strong_correlations
                    })
            
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=insights,
                visualizations=visualizations,
                confidence_score=0.9,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Descriptive analysis failed: {e}")
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=[{"type": "error", "title": "Analysis Error", "content": str(e)}],
                visualizations=[],
                confidence_score=0.0,
                created_at=datetime.now()
            )
    
    async def _perform_diagnostic_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
        """Perform diagnostic analysis"""
        
        insights = []
        visualizations = []
        
        try:
            # Anomaly detection
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns[:3]:  # Limit to first 3 numeric columns
                if column in df.columns:
                    # Z-score based anomaly detection
                    z_scores = np.abs(stats.zscore(df[column].dropna()))
                    anomalies = df[z_scores > 3]
                    
                    if len(anomalies) > 0:
                        insights.append({
                            "type": "anomaly_detection",
                            "title": f"Anomalies in {column}",
                            "content": f"Found {len(anomalies)} anomalies (Z-score > 3)",
                            "details": {
                                "anomaly_count": len(anomalies),
                                "anomaly_percentage": (len(anomalies) / len(df)) * 100,
                                "anomaly_values": anomalies[column].tolist()
                            }
                        })
                        
                        # Anomaly visualization
                        fig = px.scatter(
                            df, x=df.index, y=column,
                            title=f"Anomalies in {column}",
                            color=z_scores > 3,
                            color_discrete_map={True: 'red', False: 'blue'}
                        )
                        visualizations.append({
                            "type": "scatter_plot",
                            "title": f"{column} Anomalies",
                            "data": fig.to_dict(),
                            "column": column
                        })
            
            # Trend analysis
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0 and len(numeric_columns) > 0:
                date_col = date_columns[0]
                numeric_col = numeric_columns[0]
                
                # Time series analysis
                df_sorted = df.sort_values(date_col)
                df_grouped = df_sorted.groupby(df_sorted[date_col].dt.date)[numeric_col].mean()
                
                # Trend line
                fig = px.line(
                    x=df_grouped.index,
                    y=df_grouped.values,
                    title=f"Trend Analysis: {numeric_col} over time"
                )
                visualizations.append({
                    "type": "line_chart",
                    "title": f"{numeric_col} Trend",
                    "data": fig.to_dict(),
                    "column": numeric_col
                })
                
                # Calculate trend
                if len(df_grouped) > 1:
                    x = np.arange(len(df_grouped))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_grouped.values)
                    
                    trend_direction = "increasing" if slope > 0 else "decreasing"
                    trend_strength = "strong" if abs(r_value) > 0.7 else "weak"
                    
                    insights.append({
                        "type": "trend_analysis",
                        "title": f"Trend in {numeric_col}",
                        "content": f"Trend is {trend_direction} and {trend_strength} (R² = {r_value**2:.3f})",
                        "details": {
                            "slope": slope,
                            "r_squared": r_value**2,
                            "p_value": p_value,
                            "trend_direction": trend_direction,
                            "trend_strength": trend_strength
                        }
                    })
            
            # Segment analysis
            categorical_columns = df.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0 and len(numeric_columns) > 0:
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                
                # Segment performance
                segment_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).reset_index()
                segment_stats = segment_stats.sort_values('mean', ascending=False)
                
                insights.append({
                    "type": "segment_analysis",
                    "title": f"Segment Performance: {num_col} by {cat_col}",
                    "content": f"Best performing segment: {segment_stats.iloc[0][cat_col]} (mean: {segment_stats.iloc[0]['mean']:.2f})",
                    "details": segment_stats.to_dict('records')
                })
                
                # Segment comparison chart
                fig = px.bar(
                    segment_stats, x=cat_col, y='mean',
                    title=f"Segment Performance: {num_col} by {cat_col}",
                    error_y='std'
                )
                visualizations.append({
                    "type": "bar_chart",
                    "title": f"Segment Performance: {num_col}",
                    "data": fig.to_dict(),
                    "columns": [cat_col, num_col]
                })
            
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=insights,
                visualizations=visualizations,
                confidence_score=0.85,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Diagnostic analysis failed: {e}")
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=[{"type": "error", "title": "Analysis Error", "content": str(e)}],
                visualizations=[],
                confidence_score=0.0,
                created_at=datetime.now()
            )
    
    async def _perform_predictive_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
        """Perform predictive analysis"""
        
        insights = []
        visualizations = []
        predictions = {}
        
        try:
            target_variable = request.target_variable
            features = request.features
            
            if not target_variable or target_variable not in df.columns:
                raise ValueError("Target variable not specified or not found in data")
            
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Select features
            if not features:
                features = [col for col in numeric_columns if col != target_variable]
            else:
                features = [col for col in features if col in df.columns]
            
            if not features:
                raise ValueError("No valid features found for prediction")
            
            # Prepare feature matrix
            X = df[features].copy()
            y = df[target_variable].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Encode categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Determine if regression or classification
            is_regression = y.dtype in ['int64', 'float64'] and len(y.unique()) > 10
            
            if is_regression:
                # Regression models
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'SVR': SVR(),
                    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
                }
                
                results = {}
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[name] = {'mse': mse, 'r2': r2, 'model': model}
                    except Exception as e:
                        logger.warning(f"Model {name} failed: {e}")
                
                # Select best model
                if results:
                    best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
                    best_model = results[best_model_name]['model']
                    best_r2 = results[best_model_name]['r2']
                    
                    insights.append({
                        "type": "model_performance",
                        "title": "Best Regression Model",
                        "content": f"Best model: {best_model_name} (R² = {best_r2:.3f})",
                        "details": {name: {'mse': res['mse'], 'r2': res['r2']} for name, res in results.items()}
                    })
                    
                    # Feature importance
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': features,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        insights.append({
                            "type": "feature_importance",
                            "title": "Feature Importance",
                            "content": f"Most important feature: {feature_importance.iloc[0]['feature']}",
                            "details": feature_importance.to_dict('records')
                        })
                        
                        # Feature importance chart
                        fig = px.bar(
                            feature_importance, x='importance', y='feature',
                            title="Feature Importance",
                            orientation='h'
                        )
                        visualizations.append({
                            "type": "bar_chart",
                            "title": "Feature Importance",
                            "data": fig.to_dict(),
                            "columns": features
                        })
                    
                    # Predictions
                    y_pred = best_model.predict(X_test)
                    predictions = {
                        "model_name": best_model_name,
                        "predictions": y_pred.tolist(),
                        "actual_values": y_test.tolist(),
                        "r2_score": best_r2,
                        "mse": results[best_model_name]['mse']
                    }
                    
                    # Prediction vs actual plot
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        title="Predictions vs Actual",
                        labels={'x': 'Actual', 'y': 'Predicted'}
                    )
                    fig.add_shape(
                        type="line", line=dict(dash="dash"),
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max()
                    )
                    visualizations.append({
                        "type": "scatter_plot",
                        "title": "Predictions vs Actual",
                        "data": fig.to_dict(),
                        "columns": [target_variable]
                    })
            
            else:
                # Classification models
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'SVC': SVC(random_state=42),
                    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
                }
                
                results = {}
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        results[name] = {'accuracy': accuracy, 'model': model}
                    except Exception as e:
                        logger.warning(f"Model {name} failed: {e}")
                
                # Select best model
                if results:
                    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                    best_model = results[best_model_name]['model']
                    best_accuracy = results[best_model_name]['accuracy']
                    
                    insights.append({
                        "type": "model_performance",
                        "title": "Best Classification Model",
                        "content": f"Best model: {best_model_name} (Accuracy = {best_accuracy:.3f})",
                        "details": {name: {'accuracy': res['accuracy']} for name, res in results.items()}
                    })
                    
                    # Feature importance
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': features,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        insights.append({
                            "type": "feature_importance",
                            "title": "Feature Importance",
                            "content": f"Most important feature: {feature_importance.iloc[0]['feature']}",
                            "details": feature_importance.to_dict('records')
                        })
                    
                    # Predictions
                    y_pred = best_model.predict(X_test)
                    predictions = {
                        "model_name": best_model_name,
                        "predictions": y_pred.tolist(),
                        "actual_values": y_test.tolist(),
                        "accuracy": best_accuracy
                    }
            
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=insights,
                visualizations=visualizations,
                predictions=predictions,
                confidence_score=0.8,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=[{"type": "error", "title": "Analysis Error", "content": str(e)}],
                visualizations=[],
                confidence_score=0.0,
                created_at=datetime.now()
            )
    
    async def _perform_prescriptive_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
        """Perform prescriptive analysis"""
        
        insights = []
        visualizations = []
        recommendations = []
        
        try:
            # Optimization analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) >= 2:
                # Find optimal values
                target_col = numeric_columns[0]
                feature_col = numeric_columns[1]
                
                # Correlation analysis
                correlation = df[target_col].corr(df[feature_col])
                
                if abs(correlation) > 0.5:
                    # Strong correlation found
                    if correlation > 0:
                        recommendation = f"Increase {feature_col} to improve {target_col}"
                    else:
                        recommendation = f"Decrease {feature_col} to improve {target_col}"
                    
                    recommendations.append(recommendation)
                    
                    insights.append({
                        "type": "optimization_opportunity",
                        "title": "Optimization Opportunity",
                        "content": f"Strong correlation ({correlation:.3f}) between {target_col} and {feature_col}",
                        "details": {
                            "correlation": correlation,
                            "recommendation": recommendation
                        }
                    })
                    
                    # Scatter plot with trend line
                    fig = px.scatter(
                        df, x=feature_col, y=target_col,
                        title=f"Optimization: {target_col} vs {feature_col}",
                        trendline="ols"
                    )
                    visualizations.append({
                        "type": "scatter_plot",
                        "title": f"Optimization Analysis",
                        "data": fig.to_dict(),
                        "columns": [feature_col, target_col]
                    })
            
            # Segment optimization
            categorical_columns = df.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0 and len(numeric_columns) > 0:
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                
                # Find best performing segments
                segment_performance = df.groupby(cat_col)[num_col].agg(['mean', 'count']).reset_index()
                segment_performance = segment_performance.sort_values('mean', ascending=False)
                
                best_segment = segment_performance.iloc[0]
                worst_segment = segment_performance.iloc[-1]
                
                recommendations.append(f"Focus on {best_segment[cat_col]} segment (performance: {best_segment['mean']:.2f})")
                recommendations.append(f"Investigate {worst_segment[cat_col]} segment (performance: {worst_segment['mean']:.2f})")
                
                insights.append({
                    "type": "segment_optimization",
                    "title": "Segment Optimization",
                    "content": f"Best segment: {best_segment[cat_col]}, Worst segment: {worst_segment[cat_col]}",
                    "details": {
                        "best_segment": best_segment.to_dict(),
                        "worst_segment": worst_segment.to_dict(),
                        "performance_gap": best_segment['mean'] - worst_segment['mean']
                    }
                })
            
            # Resource allocation optimization
            if len(numeric_columns) >= 3:
                # Simple resource allocation based on performance
                performance_col = numeric_columns[0]
                cost_col = numeric_columns[1]
                volume_col = numeric_columns[2]
                
                # Calculate efficiency (performance per cost)
                df['efficiency'] = df[performance_col] / df[cost_col]
                df['total_impact'] = df[performance_col] * df[volume_col]
                
                # Find most efficient items
                efficient_items = df.nlargest(5, 'efficiency')
                
                recommendations.append("Focus resources on most efficient items")
                recommendations.append("Consider scaling up high-impact, efficient operations")
                
                insights.append({
                    "type": "resource_optimization",
                    "title": "Resource Allocation",
                    "content": f"Top 5 most efficient items identified",
                    "details": efficient_items[['efficiency', 'total_impact']].to_dict('records')
                })
            
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=insights,
                visualizations=visualizations,
                recommendations=recommendations,
                confidence_score=0.75,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Prescriptive analysis failed: {e}")
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=[{"type": "error", "title": "Analysis Error", "content": str(e)}],
                visualizations=[],
                confidence_score=0.0,
                created_at=datetime.now()
            )
    
    async def _perform_exploratory_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResult:
        """Perform exploratory analysis"""
        
        insights = []
        visualizations = []
        
        try:
            # Data overview
            insights.append({
                "type": "data_overview",
                "title": "Data Overview",
                "content": f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
                "details": {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "data_types": df.dtypes.value_counts().to_dict()
                }
            })
            
            # Univariate analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns[:3]:  # Limit to first 3
                if column in df.columns:
                    # Distribution analysis
                    skewness = df[column].skew()
                    kurtosis = df[column].kurtosis()
                    
                    insights.append({
                        "type": "distribution_analysis",
                        "title": f"{column} Distribution",
                        "content": f"Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}",
                        "details": {
                            "skewness": skewness,
                            "kurtosis": kurtosis,
                            "mean": df[column].mean(),
                            "median": df[column].median(),
                            "std": df[column].std()
                        }
                    })
            
            # Multivariate analysis
            if len(numeric_columns) > 1:
                # PCA analysis
                pca_data = df[numeric_columns].fillna(df[numeric_columns].mean())
                pca = PCA(n_components=min(3, len(numeric_columns)))
                pca_result = pca.fit_transform(pca_data)
                
                insights.append({
                    "type": "pca_analysis",
                    "title": "Principal Component Analysis",
                    "content": f"First 3 components explain {pca.explained_variance_ratio_.sum():.1%} of variance",
                    "details": {
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist()
                    }
                })
                
                # PCA visualization
                if pca_result.shape[1] >= 2:
                    fig = px.scatter(
                        x=pca_result[:, 0], y=pca_result[:, 1],
                        title="PCA Analysis - First Two Components"
                    )
                    visualizations.append({
                        "type": "scatter_plot",
                        "title": "PCA Analysis",
                        "data": fig.to_dict(),
                        "columns": numeric_columns
                    })
            
            # Clustering analysis
            if len(numeric_columns) > 1:
                # K-means clustering
                cluster_data = df[numeric_columns].fillna(df[numeric_columns].mean())
                scaler = StandardScaler()
                cluster_data_scaled = scaler.fit_transform(cluster_data)
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(cluster_data_scaled)
                
                insights.append({
                    "type": "clustering_analysis",
                    "title": "K-means Clustering",
                    "content": f"Identified 3 distinct clusters in the data",
                    "details": {
                        "cluster_centers": kmeans.cluster_centers_.tolist(),
                        "cluster_sizes": np.bincount(clusters).tolist()
                    }
                })
                
                # Cluster visualization
                if cluster_data.shape[1] >= 2:
                    fig = px.scatter(
                        x=cluster_data.iloc[:, 0], y=cluster_data.iloc[:, 1],
                        color=clusters,
                        title="K-means Clustering"
                    )
                    visualizations.append({
                        "type": "scatter_plot",
                        "title": "Clustering Analysis",
                        "data": fig.to_dict(),
                        "columns": numeric_columns
                    })
            
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=insights,
                visualizations=visualizations,
                confidence_score=0.7,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Exploratory analysis failed: {e}")
            return AnalysisResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                insights=[{"type": "error", "title": "Analysis Error", "content": str(e)}],
                visualizations=[],
                confidence_score=0.0,
                created_at=datetime.now()
            )
    
    async def create_kpi(
        self,
        name: str,
        description: str,
        value: float,
        target: float = None,
        unit: str = "",
        period: str = "monthly",
        category: str = "general"
    ) -> KPI:
        """Create KPI"""
        
        kpi = KPI(
            kpi_id=str(uuid.uuid4()),
            name=name,
            description=description,
            value=value,
            target=target,
            unit=unit,
            period=period,
            category=category,
            created_at=datetime.now()
        )
        
        self.kpis[kpi.kpi_id] = kpi
        await self._store_kpi(kpi)
        
        logger.info(f"KPI created: {kpi.kpi_id}")
        return kpi
    
    async def create_dashboard(
        self,
        name: str,
        description: str,
        widgets: List[Dict[str, Any]],
        layout: Dict[str, Any],
        filters: Dict[str, Any] = None,
        refresh_interval: int = 300,
        is_public: bool = False
    ) -> Dashboard:
        """Create dashboard"""
        
        dashboard = Dashboard(
            dashboard_id=str(uuid.uuid4()),
            name=name,
            description=description,
            widgets=widgets,
            layout=layout,
            filters=filters,
            refresh_interval=refresh_interval,
            is_public=is_public,
            created_at=datetime.now()
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        await self._store_dashboard(dashboard)
        
        logger.info(f"Dashboard created: {dashboard.dashboard_id}")
        return dashboard
    
    async def _data_refresh_task(self):
        """Background data refresh task"""
        while True:
            try:
                # Refresh cached data
                for source_id in self.data_sources:
                    if self.redis_client:
                        cache_keys = self.redis_client.keys(f"data_{source_id}_*")
                        for key in cache_keys:
                            self.redis_client.delete(key)
                
                await asyncio.sleep(3600)  # Refresh every hour
                
            except Exception as e:
                logger.error(f"Data refresh task error: {e}")
                await asyncio.sleep(3600)
    
    async def _kpi_calculation_task(self):
        """Background KPI calculation task"""
        while True:
            try:
                # Recalculate KPIs
                for kpi in self.kpis.values():
                    # This would involve actual KPI calculation logic
                    # For now, just update the timestamp
                    kpi.created_at = datetime.now()
                    await self._store_kpi(kpi)
                
                await asyncio.sleep(1800)  # Recalculate every 30 minutes
                
            except Exception as e:
                logger.error(f"KPI calculation task error: {e}")
                await asyncio.sleep(1800)
    
    async def _cache_cleanup_task(self):
        """Background cache cleanup task"""
        while True:
            try:
                if self.redis_client:
                    # Clean up expired cache entries
                    expired_keys = self.redis_client.keys("*")
                    for key in expired_keys:
                        ttl = self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            self.redis_client.expire(key, 3600)  # Set 1 hour expiration
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Cache cleanup task error: {e}")
                await asyncio.sleep(3600)
    
    async def _store_data_source(self, data_source: DataSource):
        """Store data source in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO data_sources
                (source_id, name, source_type, connection_string, query, file_path, api_endpoint, headers, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_source.source_id,
                data_source.name,
                data_source.source_type.value,
                data_source.connection_string,
                data_source.query,
                data_source.file_path,
                data_source.api_endpoint,
                json.dumps(data_source.headers),
                json.dumps(data_source.parameters),
                data_source.created_at.isoformat() if data_source.created_at else None
            ))
            conn.commit()
    
    async def _store_analysis_request(self, request: AnalysisRequest):
        """Store analysis request in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_requests
                (request_id, data_source_id, analysis_type, target_variable, features, filters, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.request_id,
                request.data_source_id,
                request.analysis_type.value,
                request.target_variable,
                json.dumps(request.features),
                json.dumps(request.filters),
                json.dumps(request.parameters),
                request.created_at.isoformat()
            ))
            conn.commit()
    
    async def _store_analysis_result(self, result: AnalysisResult):
        """Store analysis result in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results
                (result_id, request_id, analysis_type, insights, visualizations, predictions, recommendations, confidence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.request_id,
                result.analysis_type.value,
                json.dumps(result.insights),
                json.dumps(result.visualizations),
                json.dumps(result.predictions) if result.predictions else None,
                json.dumps(result.recommendations) if result.recommendations else None,
                result.confidence_score,
                result.created_at.isoformat()
            ))
            conn.commit()
    
    async def _store_kpi(self, kpi: KPI):
        """Store KPI in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO kpis
                (kpi_id, name, description, value, target, unit, trend, period, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kpi.kpi_id,
                kpi.name,
                kpi.description,
                kpi.value,
                kpi.target,
                kpi.unit,
                kpi.trend,
                kpi.period,
                kpi.category,
                kpi.created_at.isoformat()
            ))
            conn.commit()
    
    async def _store_dashboard(self, dashboard: Dashboard):
        """Store dashboard in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dashboards
                (dashboard_id, name, description, widgets, layout, filters, refresh_interval, is_public, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dashboard.dashboard_id,
                dashboard.name,
                dashboard.description,
                json.dumps(dashboard.widgets),
                json.dumps(dashboard.layout),
                json.dumps(dashboard.filters) if dashboard.filters else None,
                dashboard.refresh_interval,
                dashboard.is_public,
                dashboard.created_at.isoformat()
            ))
            conn.commit()
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Business intelligence service cleanup completed")

# Global instance
business_intelligence_service = None

async def get_business_intelligence_service() -> AdvancedBusinessIntelligenceService:
    """Get global business intelligence service instance"""
    global business_intelligence_service
    if not business_intelligence_service:
        config = {
            "database_path": "data/business_intelligence.db",
            "redis_url": "redis://localhost:6379"
        }
        business_intelligence_service = AdvancedBusinessIntelligenceService(config)
    return business_intelligence_service



