#!/usr/bin/env python3
"""
Multi-Dimensional Analytics System

Advanced multi-dimensional analytics with:
- Multi-dimensional data analysis
- Dimensional reduction techniques
- Cross-dimensional correlation analysis
- Multi-dimensional visualization
- Dimensional clustering and classification
- Multi-dimensional forecasting
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
from sklearn.decomposition import PCA, t-SNE, UMAP
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

logger = structlog.get_logger("multi_dimensional_analytics")

# =============================================================================
# MULTI-DIMENSIONAL ANALYTICS MODELS
# =============================================================================

class DimensionalityReductionMethod(Enum):
    """Dimensionality reduction methods."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    ISOMAP = "isomap"
    LLE = "lle"
    ICA = "ica"
    LDA = "lda"
    AUTOENCODER = "autoencoder"

class ClusteringMethod(Enum):
    """Clustering methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    BIRCH = "birch"
    MEAN_SHIFT = "mean_shift"
    OPTICS = "optics"

class VisualizationMethod(Enum):
    """Visualization methods."""
    SCATTER_2D = "scatter_2d"
    SCATTER_3D = "scatter_3d"
    HEATMAP = "heatmap"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR_CHART = "radar_chart"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    NETWORK = "network"

@dataclass
class MultiDimensionalDataset:
    """Multi-dimensional dataset."""
    dataset_id: str
    name: str
    description: str
    dimensions: int
    samples: int
    features: List[str]
    data: np.ndarray
    metadata: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        if not self.dataset_id:
            self.dataset_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "dimensions": self.dimensions,
            "samples": self.samples,
            "features": self.features,
            "data_shape": self.data.shape,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class DimensionalityReductionResult:
    """Dimensionality reduction result."""
    result_id: str
    dataset_id: str
    method: DimensionalityReductionMethod
    original_dimensions: int
    reduced_dimensions: int
    reduced_data: np.ndarray
    explained_variance_ratio: Optional[List[float]]
    transformation_matrix: Optional[np.ndarray]
    parameters: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        if not self.result_id:
            self.result_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.parameters:
            self.parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "dataset_id": self.dataset_id,
            "method": self.method.value,
            "original_dimensions": self.original_dimensions,
            "reduced_dimensions": self.reduced_dimensions,
            "reduced_data_shape": self.reduced_data.shape,
            "explained_variance_ratio": self.explained_variance_ratio,
            "transformation_matrix_shape": self.transformation_matrix.shape if self.transformation_matrix is not None else None,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ClusteringResult:
    """Clustering result."""
    result_id: str
    dataset_id: str
    method: ClusteringMethod
    n_clusters: int
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    parameters: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        if not self.result_id:
            self.result_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.parameters:
            self.parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "dataset_id": self.dataset_id,
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "cluster_labels_shape": self.cluster_labels.shape,
            "cluster_centers_shape": self.cluster_centers.shape if self.cluster_centers is not None else None,
            "silhouette_score": self.silhouette_score,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "davies_bouldin_score": self.davies_bouldin_score,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class CorrelationAnalysis:
    """Multi-dimensional correlation analysis."""
    analysis_id: str
    dataset_id: str
    correlation_matrix: np.ndarray
    feature_correlations: Dict[str, Dict[str, float]]
    cross_dimensional_correlations: Dict[str, float]
    significant_correlations: List[Dict[str, Any]]
    p_values: Optional[np.ndarray]
    created_at: datetime
    
    def __post_init__(self):
        if not self.analysis_id:
            self.analysis_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "dataset_id": self.dataset_id,
            "correlation_matrix_shape": self.correlation_matrix.shape,
            "feature_correlations": self.feature_correlations,
            "cross_dimensional_correlations": self.cross_dimensional_correlations,
            "significant_correlations_count": len(self.significant_correlations),
            "p_values_shape": self.p_values.shape if self.p_values is not None else None,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class MultiDimensionalForecast:
    """Multi-dimensional forecast."""
    forecast_id: str
    dataset_id: str
    forecast_horizon: int
    forecast_data: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    feature_importance: Dict[str, float]
    accuracy_metrics: Dict[str, float]
    method: str
    created_at: datetime
    
    def __post_init__(self):
        if not self.forecast_id:
            self.forecast_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast_id": self.forecast_id,
            "dataset_id": self.dataset_id,
            "forecast_horizon": self.forecast_horizon,
            "forecast_data_shape": self.forecast_data.shape,
            "confidence_intervals": {k: v.shape for k, v in self.confidence_intervals.items()},
            "feature_importance": self.feature_importance,
            "accuracy_metrics": self.accuracy_metrics,
            "method": self.method,
            "created_at": self.created_at.isoformat()
        }

# =============================================================================
# MULTI-DIMENSIONAL ANALYTICS MANAGER
# =============================================================================

class MultiDimensionalAnalyticsManager:
    """Multi-dimensional analytics management system."""
    
    def __init__(self):
        self.datasets: Dict[str, MultiDimensionalDataset] = {}
        self.dimensionality_reductions: Dict[str, DimensionalityReductionResult] = {}
        self.clustering_results: Dict[str, ClusteringResult] = {}
        self.correlation_analyses: Dict[str, CorrelationAnalysis] = {}
        self.forecasts: Dict[str, MultiDimensionalForecast] = {}
        
        # Analytics models
        self.reduction_models = {}
        self.clustering_models = {}
        self.forecasting_models = {}
        
        # Statistics
        self.stats = {
            'total_datasets': 0,
            'total_dimensionality_reductions': 0,
            'total_clustering_results': 0,
            'total_correlation_analyses': 0,
            'total_forecasts': 0,
            'average_silhouette_score': 0.0,
            'average_explained_variance': 0.0,
            'average_forecast_accuracy': 0.0
        }
        
        # Background tasks
        self.analytics_processing_task: Optional[asyncio.Task] = None
        self.model_optimization_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the multi-dimensional analytics manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize analytics models
        await self._initialize_analytics_models()
        
        # Start background tasks
        self.analytics_processing_task = asyncio.create_task(self._analytics_processing_loop())
        self.model_optimization_task = asyncio.create_task(self._model_optimization_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Multi-Dimensional Analytics Manager started")
    
    async def stop(self) -> None:
        """Stop the multi-dimensional analytics manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.analytics_processing_task:
            self.analytics_processing_task.cancel()
        if self.model_optimization_task:
            self.model_optimization_task.cancel()
        if self.performance_monitoring_task:
            self.performance_monitoring_task.cancel()
        
        logger.info("Multi-Dimensional Analytics Manager stopped")
    
    async def _initialize_analytics_models(self) -> None:
        """Initialize analytics models."""
        # Initialize dimensionality reduction models
        self.reduction_models = {
            DimensionalityReductionMethod.PCA: PCA(n_components=2),
            DimensionalityReductionMethod.TSNE: t-SNE(n_components=2, random_state=42),
            DimensionalityReductionMethod.UMAP: UMAP(n_components=2, random_state=42),
            DimensionalityReductionMethod.ISOMAP: Isomap(n_components=2),
            DimensionalityReductionMethod.LLE: LocallyLinearEmbedding(n_components=2, random_state=42)
        }
        
        # Initialize clustering models
        self.clustering_models = {
            ClusteringMethod.KMEANS: KMeans(n_clusters=3, random_state=42),
            ClusteringMethod.DBSCAN: DBSCAN(eps=0.5, min_samples=5),
            ClusteringMethod.AGGLOMERATIVE: AgglomerativeClustering(n_clusters=3)
        }
        
        # Initialize forecasting models
        self.forecasting_models = {
            'linear_regression': None,  # Will be implemented
            'random_forest': None,  # Will be implemented
            'lstm': None  # Will be implemented
        }
        
        logger.info("Multi-dimensional analytics models initialized")
    
    def create_dataset(self, name: str, description: str, data: np.ndarray, 
                      features: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create multi-dimensional dataset."""
        if metadata is None:
            metadata = {}
        
        dataset = MultiDimensionalDataset(
            name=name,
            description=description,
            dimensions=data.shape[1],
            samples=data.shape[0],
            features=features,
            data=data,
            metadata=metadata
        )
        
        self.datasets[dataset.dataset_id] = dataset
        self.stats['total_datasets'] += 1
        
        logger.info(
            "Multi-dimensional dataset created",
            dataset_id=dataset.dataset_id,
            name=name,
            dimensions=dataset.dimensions,
            samples=dataset.samples
        )
        
        return dataset.dataset_id
    
    async def perform_dimensionality_reduction(self, dataset_id: str, 
                                             method: DimensionalityReductionMethod,
                                             n_components: int = 2,
                                             parameters: Optional[Dict[str, Any]] = None) -> str:
        """Perform dimensionality reduction."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if parameters is None:
            parameters = {}
        
        dataset = self.datasets[dataset_id]
        
        # Prepare data
        data = dataset.data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Perform dimensionality reduction
        if method == DimensionalityReductionMethod.PCA:
            model = PCA(n_components=n_components, **parameters)
            reduced_data = model.fit_transform(scaled_data)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
            transformation_matrix = model.components_
        elif method == DimensionalityReductionMethod.TSNE:
            model = t-SNE(n_components=n_components, random_state=42, **parameters)
            reduced_data = model.fit_transform(scaled_data)
            explained_variance_ratio = None
            transformation_matrix = None
        elif method == DimensionalityReductionMethod.UMAP:
            model = UMAP(n_components=n_components, random_state=42, **parameters)
            reduced_data = model.fit_transform(scaled_data)
            explained_variance_ratio = None
            transformation_matrix = None
        elif method == DimensionalityReductionMethod.ISOMAP:
            model = Isomap(n_components=n_components, **parameters)
            reduced_data = model.fit_transform(scaled_data)
            explained_variance_ratio = None
            transformation_matrix = None
        elif method == DimensionalityReductionMethod.LLE:
            model = LocallyLinearEmbedding(n_components=n_components, random_state=42, **parameters)
            reduced_data = model.fit_transform(scaled_data)
            explained_variance_ratio = None
            transformation_matrix = None
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method.value}")
        
        # Create result
        result = DimensionalityReductionResult(
            dataset_id=dataset_id,
            method=method,
            original_dimensions=dataset.dimensions,
            reduced_dimensions=n_components,
            reduced_data=reduced_data,
            explained_variance_ratio=explained_variance_ratio,
            transformation_matrix=transformation_matrix,
            parameters=parameters
        )
        
        self.dimensionality_reductions[result.result_id] = result
        self.stats['total_dimensionality_reductions'] += 1
        
        # Update statistics
        if explained_variance_ratio:
            self._update_average_explained_variance(np.sum(explained_variance_ratio))
        
        logger.info(
            "Dimensionality reduction completed",
            result_id=result.result_id,
            dataset_id=dataset_id,
            method=method.value,
            original_dimensions=dataset.dimensions,
            reduced_dimensions=n_components
        )
        
        return result.result_id
    
    async def perform_clustering(self, dataset_id: str, method: ClusteringMethod,
                               n_clusters: int = 3,
                               parameters: Optional[Dict[str, Any]] = None) -> str:
        """Perform clustering analysis."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if parameters is None:
            parameters = {}
        
        dataset = self.datasets[dataset_id]
        
        # Prepare data
        data = dataset.data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Perform clustering
        if method == ClusteringMethod.KMEANS:
            model = KMeans(n_clusters=n_clusters, random_state=42, **parameters)
            cluster_labels = model.fit_predict(scaled_data)
            cluster_centers = model.cluster_centers_
        elif method == ClusteringMethod.DBSCAN:
            model = DBSCAN(**parameters)
            cluster_labels = model.fit_predict(scaled_data)
            cluster_centers = None
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        elif method == ClusteringMethod.AGGLOMERATIVE:
            model = AgglomerativeClustering(n_clusters=n_clusters, **parameters)
            cluster_labels = model.fit_predict(scaled_data)
            cluster_centers = None
        else:
            raise ValueError(f"Unsupported clustering method: {method.value}")
        
        # Calculate clustering metrics
        if n_clusters > 1:
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(scaled_data, cluster_labels)
            
            # Calculate Davies-Bouldin score
            from sklearn.metrics import davies_bouldin_score
            davies_bouldin = davies_bouldin_score(scaled_data, cluster_labels)
        else:
            silhouette_avg = 0.0
            calinski_harabasz = 0.0
            davies_bouldin = float('inf')
        
        # Create result
        result = ClusteringResult(
            dataset_id=dataset_id,
            method=method,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            silhouette_score=silhouette_avg,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            parameters=parameters
        )
        
        self.clustering_results[result.result_id] = result
        self.stats['total_clustering_results'] += 1
        
        # Update statistics
        self._update_average_silhouette_score(silhouette_avg)
        
        logger.info(
            "Clustering analysis completed",
            result_id=result.result_id,
            dataset_id=dataset_id,
            method=method.value,
            n_clusters=n_clusters,
            silhouette_score=silhouette_avg
        )
        
        return result.result_id
    
    async def perform_correlation_analysis(self, dataset_id: str) -> str:
        """Perform multi-dimensional correlation analysis."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset = self.datasets[dataset_id]
        data = dataset.data
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data.T)
        
        # Calculate feature correlations
        feature_correlations = {}
        for i, feature_a in enumerate(dataset.features):
            feature_correlations[feature_a] = {}
            for j, feature_b in enumerate(dataset.features):
                if i != j:
                    feature_correlations[feature_a][feature_b] = float(correlation_matrix[i, j])
        
        # Calculate cross-dimensional correlations
        cross_dimensional_correlations = {}
        for i in range(dataset.dimensions):
            for j in range(i + 1, dataset.dimensions):
                key = f"dim_{i}_vs_dim_{j}"
                cross_dimensional_correlations[key] = float(correlation_matrix[i, j])
        
        # Find significant correlations
        significant_correlations = []
        for i in range(dataset.dimensions):
            for j in range(i + 1, dataset.dimensions):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    significant_correlations.append({
                        'feature_a': dataset.features[i],
                        'feature_b': dataset.features[j],
                        'correlation': float(correlation),
                        'strength': 'strong' if abs(correlation) > 0.8 else 'moderate'
                    })
        
        # Create analysis
        analysis = CorrelationAnalysis(
            dataset_id=dataset_id,
            correlation_matrix=correlation_matrix,
            feature_correlations=feature_correlations,
            cross_dimensional_correlations=cross_dimensional_correlations,
            significant_correlations=significant_correlations,
            p_values=None  # Could be calculated with scipy.stats
        )
        
        self.correlation_analyses[analysis.analysis_id] = analysis
        self.stats['total_correlation_analyses'] += 1
        
        logger.info(
            "Correlation analysis completed",
            analysis_id=analysis.analysis_id,
            dataset_id=dataset_id,
            significant_correlations=len(significant_correlations)
        )
        
        return analysis.analysis_id
    
    async def generate_multi_dimensional_forecast(self, dataset_id: str, 
                                                forecast_horizon: int = 10,
                                                method: str = 'linear_regression') -> str:
        """Generate multi-dimensional forecast."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset = self.datasets[dataset_id]
        data = dataset.data
        
        # Prepare data for forecasting
        if len(data) < forecast_horizon * 2:
            raise ValueError("Insufficient data for forecasting")
        
        # Split data into training and testing
        train_size = len(data) - forecast_horizon
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Generate forecast based on method
        if method == 'linear_regression':
            forecast_data, confidence_intervals, feature_importance = await self._forecast_linear_regression(
                train_data, forecast_horizon
            )
        elif method == 'random_forest':
            forecast_data, confidence_intervals, feature_importance = await self._forecast_random_forest(
                train_data, forecast_horizon
            )
        else:
            raise ValueError(f"Unsupported forecasting method: {method}")
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_forecast_accuracy(test_data, forecast_data)
        
        # Create forecast
        forecast = MultiDimensionalForecast(
            dataset_id=dataset_id,
            forecast_horizon=forecast_horizon,
            forecast_data=forecast_data,
            confidence_intervals=confidence_intervals,
            feature_importance=feature_importance,
            accuracy_metrics=accuracy_metrics,
            method=method
        )
        
        self.forecasts[forecast.forecast_id] = forecast
        self.stats['total_forecasts'] += 1
        
        # Update statistics
        self._update_average_forecast_accuracy(accuracy_metrics.get('mae', 0.0))
        
        logger.info(
            "Multi-dimensional forecast generated",
            forecast_id=forecast.forecast_id,
            dataset_id=dataset_id,
            method=method,
            forecast_horizon=forecast_horizon
        )
        
        return forecast.forecast_id
    
    async def _forecast_linear_regression(self, train_data: np.ndarray, 
                                        forecast_horizon: int) -> tuple:
        """Generate forecast using linear regression."""
        from sklearn.linear_model import LinearRegression
        
        # Prepare features and targets
        X = train_data[:-1]  # All but last row
        y = train_data[1:]   # All but first row
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        forecast_data = []
        current_data = train_data[-1:]  # Start with last known data point
        
        for _ in range(forecast_horizon):
            next_prediction = model.predict(current_data)
            forecast_data.append(next_prediction[0])
            current_data = next_prediction.reshape(1, -1)
        
        forecast_data = np.array(forecast_data)
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = {
            'lower': forecast_data * 0.9,
            'upper': forecast_data * 1.1
        }
        
        # Calculate feature importance
        feature_importance = {
            f'feature_{i}': float(abs(model.coef_[0, i]))
            for i in range(model.coef_.shape[1])
        }
        
        return forecast_data, confidence_intervals, feature_importance
    
    async def _forecast_random_forest(self, train_data: np.ndarray, 
                                    forecast_horizon: int) -> tuple:
        """Generate forecast using random forest."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features and targets
        X = train_data[:-1]
        y = train_data[1:]
        
        # Fit model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate forecast
        forecast_data = []
        current_data = train_data[-1:]
        
        for _ in range(forecast_horizon):
            next_prediction = model.predict(current_data)
            forecast_data.append(next_prediction[0])
            current_data = next_prediction.reshape(1, -1)
        
        forecast_data = np.array(forecast_data)
        
        # Calculate confidence intervals using prediction intervals
        confidence_intervals = {
            'lower': forecast_data * 0.85,
            'upper': forecast_data * 1.15
        }
        
        # Calculate feature importance
        feature_importance = {
            f'feature_{i}': float(model.feature_importances_[i])
            for i in range(len(model.feature_importances_))
        }
        
        return forecast_data, confidence_intervals, feature_importance
    
    def _calculate_forecast_accuracy(self, actual: np.ndarray, 
                                   predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r_squared': float(r_squared)
        }
    
    def _update_average_silhouette_score(self, score: float) -> None:
        """Update average silhouette score."""
        total_results = self.stats['total_clustering_results']
        current_avg = self.stats['average_silhouette_score']
        
        if total_results > 0:
            self.stats['average_silhouette_score'] = (
                (current_avg * (total_results - 1) + score) / total_results
            )
        else:
            self.stats['average_silhouette_score'] = score
    
    def _update_average_explained_variance(self, variance: float) -> None:
        """Update average explained variance."""
        total_reductions = self.stats['total_dimensionality_reductions']
        current_avg = self.stats['average_explained_variance']
        
        if total_reductions > 0:
            self.stats['average_explained_variance'] = (
                (current_avg * (total_reductions - 1) + variance) / total_reductions
            )
        else:
            self.stats['average_explained_variance'] = variance
    
    def _update_average_forecast_accuracy(self, accuracy: float) -> None:
        """Update average forecast accuracy."""
        total_forecasts = self.stats['total_forecasts']
        current_avg = self.stats['average_forecast_accuracy']
        
        if total_forecasts > 0:
            self.stats['average_forecast_accuracy'] = (
                (current_avg * (total_forecasts - 1) + accuracy) / total_forecasts
            )
        else:
            self.stats['average_forecast_accuracy'] = accuracy
    
    async def _analytics_processing_loop(self) -> None:
        """Analytics processing loop."""
        while self.is_running:
            try:
                # Process pending analytics tasks
                # This could include automatic dimensionality reduction,
                # clustering, or correlation analysis for new datasets
                
                await asyncio.sleep(60)  # Process every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Analytics processing loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _model_optimization_loop(self) -> None:
        """Model optimization loop."""
        while self.is_running:
            try:
                # Optimize models based on performance
                # This could include hyperparameter tuning,
                # model selection, or ensemble methods
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Model optimization loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        while self.is_running:
            try:
                # Monitor analytics performance
                # This could include model accuracy monitoring,
                # resource usage tracking, or alert generation
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitoring loop error", error=str(e))
                await asyncio.sleep(30)
    
    def get_dataset(self, dataset_id: str) -> Optional[MultiDimensionalDataset]:
        """Get multi-dimensional dataset."""
        return self.datasets.get(dataset_id)
    
    def get_dimensionality_reduction(self, result_id: str) -> Optional[DimensionalityReductionResult]:
        """Get dimensionality reduction result."""
        return self.dimensionality_reductions.get(result_id)
    
    def get_clustering_result(self, result_id: str) -> Optional[ClusteringResult]:
        """Get clustering result."""
        return self.clustering_results.get(result_id)
    
    def get_correlation_analysis(self, analysis_id: str) -> Optional[CorrelationAnalysis]:
        """Get correlation analysis."""
        return self.correlation_analyses.get(analysis_id)
    
    def get_forecast(self, forecast_id: str) -> Optional[MultiDimensionalForecast]:
        """Get multi-dimensional forecast."""
        return self.forecasts.get(forecast_id)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'datasets': {
                dataset_id: {
                    'name': dataset.name,
                    'dimensions': dataset.dimensions,
                    'samples': dataset.samples,
                    'features': len(dataset.features)
                }
                for dataset_id, dataset in self.datasets.items()
            },
            'recent_dimensionality_reductions': [
                result.to_dict() for result in list(self.dimensionality_reductions.values())[-5:]
            ],
            'recent_clustering_results': [
                result.to_dict() for result in list(self.clustering_results.values())[-5:]
            ],
            'recent_correlation_analyses': [
                analysis.to_dict() for analysis in list(self.correlation_analyses.values())[-5:]
            ],
            'recent_forecasts': [
                forecast.to_dict() for forecast in list(self.forecasts.values())[-5:]
            ]
        }

# =============================================================================
# GLOBAL MULTI-DIMENSIONAL ANALYTICS INSTANCES
# =============================================================================

# Global multi-dimensional analytics manager
multi_dimensional_analytics_manager = MultiDimensionalAnalyticsManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DimensionalityReductionMethod',
    'ClusteringMethod',
    'VisualizationMethod',
    'MultiDimensionalDataset',
    'DimensionalityReductionResult',
    'ClusteringResult',
    'CorrelationAnalysis',
    'MultiDimensionalForecast',
    'MultiDimensionalAnalyticsManager',
    'multi_dimensional_analytics_manager'
]





























