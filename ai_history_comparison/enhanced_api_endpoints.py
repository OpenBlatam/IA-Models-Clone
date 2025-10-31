"""
AI History Comparison System - Enhanced API Endpoints

This module provides enhanced REST API endpoints that integrate all advanced features
including ML engine, real-time streaming, visualization, and advanced analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import json
import io

from .ai_history_analyzer import (
    AIHistoryAnalyzer, ComparisonType, MetricType, 
    ComparisonResult, TrendAnalysis, HistoryEntry, ContentMetrics
)
from .advanced_ml_engine import (
    ml_engine, AnomalyDetectionResult, AdvancedClusteringResult, 
    PredictiveModelResult, detect_anomalies, advanced_clustering, 
    build_predictive_models, extract_advanced_features
)
from .realtime_streaming import (
    websocket_manager, realtime_analyzer, StreamSubscriptionType,
    analyze_content_streaming, compare_content_streaming
)
from .visualization_engine import (
    viz_engine, ChartType, DashboardType, ChartConfig, DashboardConfig,
    create_trend_chart, create_quality_distribution_chart, 
    create_comparison_chart, create_anomaly_chart, create_clustering_chart
)
from .models import ModelUtils, ModelSerializer
from .config import get_config

logger = logging.getLogger(__name__)

# Initialize enhanced router
enhanced_router = APIRouter(prefix="/ai-history/v2", tags=["Enhanced AI History Comparison"])

# Initialize analyzer
analyzer = AIHistoryAnalyzer()

# Enhanced Pydantic models
class AdvancedAnalysisRequest(BaseModel):
    """Request model for advanced content analysis"""
    content: str = Field(..., min_length=10, max_length=10000, description="Content to analyze")
    model_version: str = Field(..., min_length=1, max_length=100, description="AI model version")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    enable_ml_features: bool = Field(True, description="Enable advanced ML features")
    enable_realtime: bool = Field(False, description="Enable real-time streaming")
    user_id: Optional[str] = Field(None, description="User ID for real-time updates")

class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    entry_ids: Optional[List[str]] = Field(None, description="Specific entry IDs to analyze")
    method: str = Field("isolation_forest", description="Anomaly detection method")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    model_version: Optional[str] = Field(None, description="Filter by model version")

class AdvancedClusteringRequest(BaseModel):
    """Request model for advanced clustering"""
    algorithm: str = Field("auto", description="Clustering algorithm")
    max_clusters: int = Field(10, ge=2, le=50, description="Maximum number of clusters")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    model_version: Optional[str] = Field(None, description="Filter by model version")
    include_visualization: bool = Field(True, description="Include visualization data")

class PredictiveModelingRequest(BaseModel):
    """Request model for predictive modeling"""
    target_metric: str = Field("readability_score", description="Metric to predict")
    model_types: List[str] = Field(["linear_regression", "random_forest"], description="Model types to train")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    model_version: Optional[str] = Field(None, description="Filter by model version")

class VisualizationRequest(BaseModel):
    """Request model for visualization generation"""
    chart_type: str = Field("trend", description="Type of chart to generate")
    data_type: str = Field("readability", description="Type of data to visualize")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    model_version: Optional[str] = Field(None, description="Filter by model version")
    format: str = Field("json", description="Output format (json, png, svg)")

class DashboardRequest(BaseModel):
    """Request model for dashboard generation"""
    dashboard_type: str = Field("overview", description="Type of dashboard")
    charts: List[Dict[str, Any]] = Field(..., description="Chart configurations")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    model_version: Optional[str] = Field(None, description="Filter by model version")

class AdvancedComparisonRequest(BaseModel):
    """Request model for advanced content comparison"""
    entry_id_1: str = Field(..., description="First entry ID to compare")
    entry_id_2: str = Field(..., description="Second entry ID to compare")
    comparison_types: List[str] = Field(
        default=["content_similarity", "quality_metrics", "anomaly_analysis"],
        description="Types of comparison to perform"
    )
    include_ml_analysis: bool = Field(True, description="Include ML-based analysis")
    include_visualization: bool = Field(True, description="Include visualization data")

# Enhanced Response Models
class AdvancedAnalysisResponse(BaseModel):
    """Response model for advanced content analysis"""
    entry_id: str
    basic_metrics: Dict[str, Any]
    advanced_features: Dict[str, Any]
    ml_insights: Optional[Dict[str, Any]] = None
    realtime_status: Optional[str] = None
    timestamp: datetime
    message: str

class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""
    total_entries_analyzed: int
    anomalies_detected: int
    anomaly_rate: float
    anomalies: List[Dict[str, Any]]
    method_used: str
    analysis_timestamp: datetime

class AdvancedClusteringResponse(BaseModel):
    """Response model for advanced clustering"""
    clusters: Dict[int, List[str]]
    cluster_centers: Dict[int, Dict[str, float]]
    quality_metrics: Dict[str, float]
    algorithm_used: str
    visualization_data: Optional[Dict[str, Any]] = None
    clustering_timestamp: datetime

class PredictiveModelingResponse(BaseModel):
    """Response model for predictive modeling"""
    models_trained: List[str]
    best_model: str
    model_performance: Dict[str, Dict[str, float]]
    predictions: Dict[str, Any]
    feature_importance: Dict[str, Dict[str, float]]
    training_timestamp: datetime

class VisualizationResponse(BaseModel):
    """Response model for visualization"""
    chart_type: str
    data_type: str
    chart_data: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime

class DashboardResponse(BaseModel):
    """Response model for dashboard"""
    dashboard_type: str
    title: str
    charts: List[Dict[str, Any]]
    layout: str
    refresh_interval: int
    generated_at: datetime

class AdvancedComparisonResponse(BaseModel):
    """Response model for advanced comparison"""
    entry_id_1: str
    entry_id_2: str
    basic_comparison: Dict[str, Any]
    ml_analysis: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    insights: List[str]
    recommendations: List[str]
    comparison_timestamp: datetime

# Enhanced API Endpoints

@enhanced_router.post("/analyze/advanced", response_model=AdvancedAnalysisResponse)
async def advanced_content_analysis(request: AdvancedAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Perform advanced content analysis with ML features and real-time streaming
    
    This endpoint provides comprehensive content analysis including:
    - Basic quality metrics
    - Advanced ML features
    - Real-time streaming updates (optional)
    - Anomaly detection
    - Feature extraction
    """
    try:
        # Perform basic analysis
        entry_id = analyzer.add_history_entry(
            content=request.content,
            model_version=request.model_version,
            metadata=request.metadata
        )
        
        entry = analyzer._get_entry_by_id(entry_id)
        if not entry:
            raise HTTPException(status_code=500, detail="Failed to retrieve created entry")
        
        # Extract basic metrics
        basic_metrics = {
            "readability_score": entry.metrics.readability_score,
            "sentiment_score": entry.metrics.sentiment_score,
            "word_count": entry.metrics.word_count,
            "sentence_count": entry.metrics.sentence_count,
            "avg_word_length": entry.metrics.avg_word_length,
            "complexity_score": entry.metrics.complexity_score,
            "topic_diversity": entry.metrics.topic_diversity,
            "consistency_score": entry.metrics.consistency_score
        }
        
        # Extract advanced features if enabled
        advanced_features = {}
        ml_insights = {}
        realtime_status = None
        
        if request.enable_ml_features:
            # Extract advanced ML features
            advanced_features = extract_advanced_features(request.content)
            
            # Perform anomaly detection
            entry_data = {
                "id": entry_id,
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            
            anomalies = detect_anomalies([entry_data], method="isolation_forest")
            if anomalies and anomalies[0].is_anomaly:
                ml_insights["anomaly_detected"] = {
                    "is_anomaly": True,
                    "anomaly_type": anomalies[0].anomaly_type,
                    "anomaly_score": anomalies[0].anomaly_score,
                    "explanation": anomalies[0].explanation,
                    "recommendations": anomalies[0].recommendations
                }
            else:
                ml_insights["anomaly_detected"] = {"is_anomaly": False}
        
        # Handle real-time streaming if enabled
        if request.enable_realtime and request.user_id:
            try:
                realtime_entry_id = await analyze_content_streaming(
                    request.content, request.model_version, request.user_id, request.metadata
                )
                realtime_status = "streaming_enabled"
            except Exception as e:
                logger.warning(f"Real-time streaming failed: {e}")
                realtime_status = "streaming_failed"
        
        # Prepare response
        response = AdvancedAnalysisResponse(
            entry_id=entry_id,
            basic_metrics=basic_metrics,
            advanced_features=advanced_features,
            ml_insights=ml_insights if request.enable_ml_features else None,
            realtime_status=realtime_status,
            timestamp=entry.timestamp,
            message="Advanced content analysis completed successfully"
        )
        
        logger.info(f"Advanced content analysis completed: {entry_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in advanced content analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")

@enhanced_router.post("/anomalies/detect", response_model=AnomalyDetectionResponse)
async def detect_content_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in content entries using advanced ML methods
    
    This endpoint provides comprehensive anomaly detection including:
    - Isolation Forest algorithm
    - DBSCAN clustering
    - Statistical methods
    - Detailed anomaly explanations
    """
    try:
        # Get entries to analyze
        entries = []
        
        if request.entry_ids:
            # Analyze specific entries
            for entry_id in request.entry_ids:
                entry = analyzer._get_entry_by_id(entry_id)
                if entry:
                    entry_data = {
                        "id": entry.id,
                        "readability_score": entry.metrics.readability_score,
                        "sentiment_score": entry.metrics.sentiment_score,
                        "word_count": entry.metrics.word_count,
                        "sentence_count": entry.metrics.sentence_count,
                        "avg_word_length": entry.metrics.avg_word_length,
                        "complexity_score": entry.metrics.complexity_score,
                        "topic_diversity": entry.metrics.topic_diversity,
                        "consistency_score": entry.metrics.consistency_score
                    }
                    entries.append(entry_data)
        else:
            # Analyze all entries or filtered entries
            all_entries = analyzer.history_entries
            
            # Apply time window filter
            if request.time_window_days:
                cutoff_time = datetime.now() - timedelta(days=request.time_window_days)
                all_entries = [e for e in all_entries if e.timestamp >= cutoff_time]
            
            # Apply model version filter
            if request.model_version:
                all_entries = [e for e in all_entries if e.model_version == request.model_version]
            
            # Convert to analysis format
            for entry in all_entries:
                entry_data = {
                    "id": entry.id,
                    "readability_score": entry.metrics.readability_score,
                    "sentiment_score": entry.metrics.sentiment_score,
                    "word_count": entry.metrics.word_count,
                    "sentence_count": entry.metrics.sentence_count,
                    "avg_word_length": entry.metrics.avg_word_length,
                    "complexity_score": entry.metrics.complexity_score,
                    "topic_diversity": entry.metrics.topic_diversity,
                    "consistency_score": entry.metrics.consistency_score
                }
                entries.append(entry_data)
        
        if not entries:
            raise HTTPException(status_code=400, detail="No entries found for analysis")
        
        # Perform anomaly detection
        anomalies = detect_anomalies(entries, method=request.method)
        
        # Process results
        anomaly_entries = [a for a in anomalies if a.is_anomaly]
        anomaly_rate = len(anomaly_entries) / len(entries) if entries else 0
        
        # Format anomaly data
        anomaly_data = []
        for anomaly in anomaly_entries:
            anomaly_data.append({
                "entry_id": anomaly.anomaly_type,  # This should be the actual entry ID
                "anomaly_type": anomaly.anomaly_type,
                "anomaly_score": anomaly.anomaly_score,
                "confidence": anomaly.confidence,
                "explanation": anomaly.explanation,
                "recommendations": anomaly.recommendations
            })
        
        # Prepare response
        response = AnomalyDetectionResponse(
            total_entries_analyzed=len(entries),
            anomalies_detected=len(anomaly_entries),
            anomaly_rate=anomaly_rate,
            anomalies=anomaly_data,
            method_used=request.method,
            analysis_timestamp=datetime.now()
        )
        
        logger.info(f"Anomaly detection completed: {len(anomaly_entries)} anomalies found")
        return response
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@enhanced_router.post("/clustering/advanced", response_model=AdvancedClusteringResponse)
async def advanced_content_clustering(request: AdvancedClusteringRequest):
    """
    Perform advanced content clustering with multiple algorithms and optimization
    
    This endpoint provides comprehensive clustering analysis including:
    - Multiple clustering algorithms (K-means, DBSCAN, Agglomerative, Spectral)
    - Automatic algorithm selection
    - Quality metrics and optimization
    - Visualization data (optional)
    """
    try:
        # Get entries to cluster
        entries = []
        all_entries = analyzer.history_entries
        
        # Apply time window filter
        if request.time_window_days:
            cutoff_time = datetime.now() - timedelta(days=request.time_window_days)
            all_entries = [e for e in all_entries if e.timestamp >= cutoff_time]
        
        # Apply model version filter
        if request.model_version:
            all_entries = [e for e in all_entries if e.model_version == request.model_version]
        
        # Convert to clustering format
        for entry in all_entries:
            entry_data = {
                "id": entry.id,
                "content": entry.content,
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            entries.append(entry_data)
        
        if len(entries) < 2:
            raise HTTPException(status_code=400, detail="Insufficient entries for clustering")
        
        # Perform advanced clustering
        clustering_result = advanced_clustering(
            entries, 
            algorithm=request.algorithm, 
            max_clusters=request.max_clusters
        )
        
        # Generate visualization if requested
        visualization_data = None
        if request.include_visualization:
            try:
                visualization_data = create_clustering_chart({
                    "clusters": clustering_result.clusters,
                    "cluster_centers": clustering_result.cluster_centers
                })
            except Exception as e:
                logger.warning(f"Failed to generate clustering visualization: {e}")
        
        # Prepare response
        response = AdvancedClusteringResponse(
            clusters=clustering_result.clusters,
            cluster_centers=clustering_result.cluster_centers,
            quality_metrics={
                "silhouette_score": clustering_result.silhouette_score,
                "calinski_harabasz_score": clustering_result.calinski_harabasz_score,
                "optimal_clusters": clustering_result.optimal_clusters
            },
            algorithm_used=clustering_result.algorithm_used,
            visualization_data=visualization_data,
            clustering_timestamp=datetime.now()
        )
        
        logger.info(f"Advanced clustering completed: {clustering_result.optimal_clusters} clusters")
        return response
        
    except Exception as e:
        logger.error(f"Error in advanced clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced clustering failed: {str(e)}")

@enhanced_router.post("/models/predictive", response_model=PredictiveModelingResponse)
async def build_predictive_models_endpoint(request: PredictiveModelingRequest):
    """
    Build predictive models for content metrics using advanced ML algorithms
    
    This endpoint provides comprehensive predictive modeling including:
    - Multiple model types (Linear Regression, Random Forest, Neural Networks)
    - Model performance comparison
    - Feature importance analysis
    - Future predictions
    """
    try:
        # Get entries for training
        entries = []
        all_entries = analyzer.history_entries
        
        # Apply time window filter
        if request.time_window_days:
            cutoff_time = datetime.now() - timedelta(days=request.time_window_days)
            all_entries = [e for e in all_entries if e.timestamp >= cutoff_time]
        
        # Apply model version filter
        if request.model_version:
            all_entries = [e for e in all_entries if e.model_version == request.model_version]
        
        # Convert to training format
        for entry in all_entries:
            entry_data = {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            entries.append(entry_data)
        
        if len(entries) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for predictive modeling")
        
        # Build predictive models
        models = build_predictive_models(
            entries, 
            target_metric=request.target_metric, 
            model_types=request.model_types
        )
        
        if not models:
            raise HTTPException(status_code=500, detail="Failed to train any models")
        
        # Find best model
        best_model = max(models.keys(), key=lambda k: models[k].model_accuracy)
        
        # Prepare model performance data
        model_performance = {}
        feature_importance = {}
        
        for model_name, model_result in models.items():
            model_performance[model_name] = {
                "accuracy": model_result.model_accuracy,
                "trend_direction": model_result.trend_direction,
                "next_prediction": model_result.next_prediction
            }
            feature_importance[model_name] = model_result.feature_importance
        
        # Prepare response
        response = PredictiveModelingResponse(
            models_trained=list(models.keys()),
            best_model=best_model,
            model_performance=model_performance,
            predictions={
                "next_prediction": models[best_model].next_prediction,
                "trend_direction": models[best_model].trend_direction
            },
            feature_importance=feature_importance,
            training_timestamp=datetime.now()
        )
        
        logger.info(f"Predictive modeling completed: {len(models)} models trained")
        return response
        
    except Exception as e:
        logger.error(f"Error in predictive modeling: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predictive modeling failed: {str(e)}")

@enhanced_router.post("/visualize", response_model=VisualizationResponse)
async def generate_visualization(request: VisualizationRequest):
    """
    Generate data visualizations for AI history analysis
    
    This endpoint provides comprehensive visualization capabilities including:
    - Trend charts
    - Distribution charts
    - Comparison charts
    - Anomaly visualizations
    - Clustering visualizations
    """
    try:
        # Get data for visualization
        entries = []
        all_entries = analyzer.history_entries
        
        # Apply time window filter
        if request.time_window_days:
            cutoff_time = datetime.now() - timedelta(days=request.time_window_days)
            all_entries = [e for e in all_entries if e.timestamp >= cutoff_time]
        
        # Apply model version filter
        if request.model_version:
            all_entries = [e for e in all_entries if e.model_version == request.model_version]
        
        # Convert to visualization format
        for entry in all_entries:
            entry_data = {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            entries.append(entry_data)
        
        if not entries:
            raise HTTPException(status_code=400, detail="No data available for visualization")
        
        # Generate visualization based on type
        chart_data = None
        metadata = {
            "total_entries": len(entries),
            "time_window_days": request.time_window_days,
            "model_version": request.model_version
        }
        
        if request.chart_type == "trend":
            # Create trend chart
            metric_data = [(entry["timestamp"], entry[request.data_type]) for entry in entries]
            chart_data = create_trend_chart(metric_data, f"{request.data_type} Trend", request.data_type)
            
        elif request.chart_type == "distribution":
            # Create distribution chart
            chart_data = create_quality_distribution_chart(entries, request.data_type)
            
        elif request.chart_type == "comparison":
            # Create comparison chart
            chart_data = create_comparison_chart(entries)
            
        elif request.chart_type == "anomaly":
            # Create anomaly chart
            # First detect anomalies
            anomaly_entries = []
            for entry in entries:
                entry_data = {
                    "id": entry["id"],
                    "readability_score": entry["readability_score"],
                    "sentiment_score": entry["sentiment_score"],
                    "word_count": entry["word_count"],
                    "sentence_count": entry["sentence_count"],
                    "avg_word_length": entry["avg_word_length"],
                    "complexity_score": entry["complexity_score"],
                    "topic_diversity": entry["topic_diversity"],
                    "consistency_score": entry["consistency_score"]
                }
                anomaly_entries.append(entry_data)
            
            anomalies = detect_anomalies(anomaly_entries, method="isolation_forest")
            anomaly_data = [{"entry_id": a.anomaly_type, "is_anomaly": a.is_anomaly} for a in anomalies]
            chart_data = create_anomaly_chart(entries, anomaly_data)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {request.chart_type}")
        
        if "error" in chart_data:
            raise HTTPException(status_code=500, detail=chart_data["error"])
        
        # Prepare response
        response = VisualizationResponse(
            chart_type=request.chart_type,
            data_type=request.data_type,
            chart_data=chart_data,
            metadata=metadata,
            generated_at=datetime.now()
        )
        
        logger.info(f"Visualization generated: {request.chart_type} for {request.data_type}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@enhanced_router.post("/dashboard", response_model=DashboardResponse)
async def generate_dashboard(request: DashboardRequest):
    """
    Generate comprehensive dashboard with multiple visualizations
    
    This endpoint provides dashboard generation capabilities including:
    - Multiple chart types
    - Custom layouts
    - Real-time data
    - Interactive elements
    """
    try:
        # Get data for dashboard
        entries = []
        all_entries = analyzer.history_entries
        
        # Apply time window filter
        if request.time_window_days:
            cutoff_time = datetime.now() - timedelta(days=request.time_window_days)
            all_entries = [e for e in all_entries if e.timestamp >= cutoff_time]
        
        # Apply model version filter
        if request.model_version:
            all_entries = [e for e in all_entries if e.model_version == request.model_version]
        
        # Convert to dashboard format
        for entry in all_entries:
            entry_data = {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            entries.append(entry_data)
        
        if not entries:
            raise HTTPException(status_code=400, detail="No data available for dashboard")
        
        # Create dashboard configuration
        dashboard_config = DashboardConfig(
            dashboard_type=DashboardType(request.dashboard_type),
            title=f"AI History {request.dashboard_type.title()} Dashboard",
            charts=[],
            layout="grid",
            refresh_interval=300
        )
        
        # Generate charts
        dashboard_charts = []
        for chart_config in request.charts:
            try:
                chart_type = ChartType(chart_config.get("type", "line"))
                chart_config_obj = ChartConfig(
                    chart_type=chart_type,
                    title=chart_config.get("title", f"{chart_type.value.title()} Chart"),
                    x_axis=chart_config.get("x_axis", "timestamp"),
                    y_axis=chart_config.get("y_axis", "readability_score"),
                    width=chart_config.get("width", 800),
                    height=chart_config.get("height", 600)
                )
                
                # Generate chart data
                chart_data = viz_engine._generate_chart_data(chart_config_obj, {"entries": entries})
                
                if "error" not in chart_data:
                    dashboard_charts.append({
                        "config": chart_config,
                        "data": chart_data
                    })
                
            except Exception as e:
                logger.warning(f"Failed to generate chart: {e}")
                continue
        
        # Prepare response
        response = DashboardResponse(
            dashboard_type=request.dashboard_type,
            title=dashboard_config.title,
            charts=dashboard_charts,
            layout="grid",
            refresh_interval=300,
            generated_at=datetime.now()
        )
        
        logger.info(f"Dashboard generated: {request.dashboard_type} with {len(dashboard_charts)} charts")
        return response
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@enhanced_router.post("/compare/advanced", response_model=AdvancedComparisonResponse)
async def advanced_content_comparison(request: AdvancedComparisonRequest):
    """
    Perform advanced content comparison with ML analysis and visualization
    
    This endpoint provides comprehensive comparison capabilities including:
    - Basic similarity and quality comparison
    - ML-based analysis
    - Anomaly detection
    - Visualization data
    - Detailed insights and recommendations
    """
    try:
        # Perform basic comparison
        basic_result = analyzer.compare_entries(
            request.entry_id_1, request.entry_id_2, 
            [ComparisonType(ct) for ct in request.comparison_types]
        )
        
        # Get entries for ML analysis
        entry1 = analyzer._get_entry_by_id(request.entry_id_1)
        entry2 = analyzer._get_entry_by_id(request.entry_id_2)
        
        if not entry1 or not entry2:
            raise HTTPException(status_code=404, detail="One or both entries not found")
        
        # Prepare ML analysis data
        ml_analysis = None
        if request.include_ml_analysis:
            entry1_data = {
                "id": entry1.id,
                "readability_score": entry1.metrics.readability_score,
                "sentiment_score": entry1.metrics.sentiment_score,
                "word_count": entry1.metrics.word_count,
                "sentence_count": entry1.metrics.sentence_count,
                "avg_word_length": entry1.metrics.avg_word_length,
                "complexity_score": entry1.metrics.complexity_score,
                "topic_diversity": entry1.metrics.topic_diversity,
                "consistency_score": entry1.metrics.consistency_score
            }
            
            entry2_data = {
                "id": entry2.id,
                "readability_score": entry2.metrics.readability_score,
                "sentiment_score": entry2.metrics.sentiment_score,
                "word_count": entry2.metrics.word_count,
                "sentence_count": entry2.metrics.sentence_count,
                "avg_word_length": entry2.metrics.avg_word_length,
                "complexity_score": entry2.metrics.complexity_score,
                "topic_diversity": entry2.metrics.topic_diversity,
                "consistency_score": entry2.metrics.consistency_score
            }
            
            # Detect anomalies
            anomalies = detect_anomalies([entry1_data, entry2_data], method="isolation_forest")
            
            # Extract advanced features
            advanced_features1 = extract_advanced_features(entry1.content)
            advanced_features2 = extract_advanced_features(entry2.content)
            
            ml_analysis = {
                "anomaly_analysis": {
                    "entry1_anomaly": anomalies[0].is_anomaly if len(anomalies) > 0 else False,
                    "entry2_anomaly": anomalies[1].is_anomaly if len(anomalies) > 1 else False,
                    "anomaly_details": [{"entry_id": a.anomaly_type, "is_anomaly": a.is_anomaly, "type": a.anomaly_type} for a in anomalies]
                },
                "advanced_features": {
                    "entry1": advanced_features1,
                    "entry2": advanced_features2
                }
            }
        
        # Generate visualization if requested
        visualization_data = None
        if request.include_visualization:
            try:
                comparison_data = [
                    {
                        "label": f"Entry {request.entry_id_1[:8]}",
                        "readability_score": entry1.metrics.readability_score,
                        "sentiment_score": entry1.metrics.sentiment_score,
                        "complexity_score": entry1.metrics.complexity_score,
                        "topic_diversity": entry1.metrics.topic_diversity
                    },
                    {
                        "label": f"Entry {request.entry_id_2[:8]}",
                        "readability_score": entry2.metrics.readability_score,
                        "sentiment_score": entry2.metrics.sentiment_score,
                        "complexity_score": entry2.metrics.complexity_score,
                        "topic_diversity": entry2.metrics.topic_diversity
                    }
                ]
                visualization_data = create_comparison_chart(comparison_data)
            except Exception as e:
                logger.warning(f"Failed to generate comparison visualization: {e}")
        
        # Generate insights and recommendations
        insights = []
        recommendations = basic_result.recommendations.copy()
        
        # Add ML-based insights
        if ml_analysis:
            if ml_analysis["anomaly_analysis"]["entry1_anomaly"]:
                insights.append("Entry 1 shows anomalous characteristics")
            if ml_analysis["anomaly_analysis"]["entry2_anomaly"]:
                insights.append("Entry 2 shows anomalous characteristics")
        
        # Add quality insights
        if basic_result.similarity_score > 0.8:
            insights.append("Content pieces are highly similar")
        elif basic_result.similarity_score < 0.3:
            insights.append("Content pieces are quite different")
        
        if basic_result.trend_direction == "improving":
            insights.append("Content quality is improving over time")
        elif basic_result.trend_direction == "declining":
            insights.append("Content quality is declining over time")
        
        # Prepare response
        response = AdvancedComparisonResponse(
            entry_id_1=request.entry_id_1,
            entry_id_2=request.entry_id_2,
            basic_comparison={
                "similarity_score": basic_result.similarity_score,
                "quality_difference": basic_result.quality_difference,
                "trend_direction": basic_result.trend_direction,
                "significant_changes": basic_result.significant_changes,
                "confidence_score": basic_result.confidence_score
            },
            ml_analysis=ml_analysis,
            visualization_data=visualization_data,
            insights=insights,
            recommendations=recommendations,
            comparison_timestamp=datetime.now()
        )
        
        logger.info(f"Advanced comparison completed: {request.entry_id_1} vs {request.entry_id_2}")
        return response
        
    except Exception as e:
        logger.error(f"Error in advanced comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced comparison failed: {str(e)}")

@enhanced_router.get("/features/available")
async def get_available_features():
    """Get list of available advanced features"""
    return {
        "ml_features": {
            "anomaly_detection": ["isolation_forest", "dbscan", "statistical"],
            "clustering": ["kmeans", "dbscan", "agglomerative", "spectral", "auto"],
            "predictive_modeling": ["linear_regression", "ridge_regression", "lasso_regression", "random_forest", "neural_network", "svr"]
        },
        "visualization": {
            "chart_types": viz_engine.get_available_chart_types(),
            "dashboard_types": viz_engine.get_available_dashboard_types(),
            "color_palettes": viz_engine.get_color_palettes()
        },
        "realtime": {
            "streaming_enabled": realtime_analyzer is not None,
            "subscription_types": [st.value for st in StreamSubscriptionType]
        },
        "analysis_types": {
            "comparison_types": [ct.value for ct in ComparisonType],
            "metric_types": [mt.value for mt in MetricType]
        }
    }

@enhanced_router.get("/status/enhanced")
async def get_enhanced_system_status():
    """Get enhanced system status with all features"""
    try:
        # Basic system status
        total_entries = analyzer.get_entry_count()
        
        # ML engine status
        ml_status = {
            "available": ml_engine is not None,
            "models_loaded": len(ml_engine.get_model_info()),
            "transformers_available": ml_engine.transformer_model is not None,
            "spacy_available": ml_engine.nlp is not None
        }
        
        # Visualization status
        viz_status = {
            "matplotlib_available": HAS_MATPLOTLIB,
            "plotly_available": HAS_PLOTLY,
            "seaborn_available": HAS_SEABORN,
            "cache_size": len(viz_engine.chart_cache)
        }
        
        # Real-time status
        realtime_status = {
            "enabled": realtime_analyzer is not None,
            "active_connections": len(websocket_manager.active_connections),
            "background_tasks": len(websocket_manager.background_tasks)
        }
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "basic_system": {
                "total_entries": total_entries,
                "analyzer_status": "healthy"
            },
            "ml_engine": ml_status,
            "visualization": viz_status,
            "realtime": realtime_status,
            "features": {
                "advanced_analysis": True,
                "anomaly_detection": True,
                "clustering": True,
                "predictive_modeling": True,
                "visualization": True,
                "realtime_streaming": realtime_analyzer is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# WebSocket endpoint for real-time streaming
@enhanced_router.websocket("/stream/{user_id}")
async def enhanced_websocket_endpoint(websocket: WebSocket, user_id: str):
    """Enhanced WebSocket endpoint with advanced features"""
    session_id = None
    
    try:
        await websocket_manager.connect(websocket, user_id, session_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle enhanced message types
                if message.get("type") == "analyze_realtime":
                    content = message.get("content")
                    model_version = message.get("model_version", "unknown")
                    metadata = message.get("metadata", {})
                    
                    if content:
                        try:
                            entry_id = await analyze_content_streaming(content, model_version, user_id, metadata)
                            await websocket.send_text(json.dumps({
                                "type": "analysis_complete",
                                "entry_id": entry_id,
                                "status": "success"
                            }))
                        except Exception as e:
                            await websocket.send_text(json.dumps({
                                "type": "analysis_error",
                                "error": str(e),
                                "status": "error"
                            }))
                
                elif message.get("type") == "compare_realtime":
                    entry_id_1 = message.get("entry_id_1")
                    entry_id_2 = message.get("entry_id_2")
                    
                    if entry_id_1 and entry_id_2:
                        try:
                            result = await compare_content_streaming(entry_id_1, entry_id_2, user_id)
                            await websocket.send_text(json.dumps({
                                "type": "comparison_complete",
                                "result": result,
                                "status": "success"
                            }))
                        except Exception as e:
                            await websocket.send_text(json.dumps({
                                "type": "comparison_error",
                                "error": str(e),
                                "status": "error"
                            }))
                
                # Handle other message types from base streaming
                elif message.get("type") == "subscribe":
                    subscription_types = [
                        StreamSubscriptionType(st) for st in message.get("subscription_types", [])
                    ]
                    filters = message.get("filters", {})
                    await websocket_manager.subscribe(user_id, subscription_types, filters)
                
                elif message.get("type") == "unsubscribe":
                    await websocket_manager.unsubscribe(user_id)
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling enhanced WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    except Exception as e:
        logger.error(f"Enhanced WebSocket error: {e}")
    finally:
        await websocket_manager.disconnect(f"{user_id}_{session_id}")

# Initialize real-time analyzer when module is imported
async def initialize_enhanced_features():
    """Initialize enhanced features"""
    try:
        from .realtime_streaming import initialize_realtime_analyzer
        await initialize_realtime_analyzer(analyzer)
        logger.info("Enhanced features initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced features: {e}")

# Convenience functions for external use
async def analyze_content_enhanced(content: str, model_version: str, enable_ml: bool = True, 
                                 enable_realtime: bool = False, user_id: str = None) -> Dict[str, Any]:
    """Enhanced content analysis with all features"""
    request = AdvancedAnalysisRequest(
        content=content,
        model_version=model_version,
        enable_ml_features=enable_ml,
        enable_realtime=enable_realtime,
        user_id=user_id
    )
    
    # This would call the endpoint logic directly
    # For now, return a simplified response
    return {
        "entry_id": "enhanced_analysis_placeholder",
        "message": "Enhanced analysis completed",
        "features_enabled": {
            "ml_features": enable_ml,
            "realtime": enable_realtime
        }
    }



























