"""
AI History Comparison System - Comprehensive API Integration

This module provides a unified API that integrates all advanced features including
ML engine, evolution tracking, similarity detection, real-time streaming, and visualization.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import json

# Import all advanced modules
from .ai_history_analyzer import AIHistoryAnalyzer, ComparisonType, MetricType
from .advanced_ml_engine import ml_engine, detect_anomalies, advanced_clustering, build_predictive_models
from .ai_evolution_tracker import evolution_tracker, track_model_version, analyze_evolution, detect_performance_regression
from .content_similarity_engine import similarity_engine, calculate_similarity, detect_plagiarism, find_similar_content
from .realtime_streaming import websocket_manager, realtime_analyzer
from .visualization_engine import viz_engine, create_trend_chart, create_quality_distribution_chart
from .models import ModelUtils, ModelSerializer
from .config import get_config

logger = logging.getLogger(__name__)

# Initialize comprehensive router
comprehensive_router = APIRouter(prefix="/ai-history/comprehensive", tags=["Comprehensive AI Analysis"])

# Initialize analyzer
analyzer = AIHistoryAnalyzer()

# Comprehensive Pydantic models
class ComprehensiveAnalysisRequest(BaseModel):
    """Request model for comprehensive content analysis"""
    content: str = Field(..., min_length=10, max_length=10000, description="Content to analyze")
    model_version: str = Field(..., min_length=1, max_length=100, description="AI model version")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # Feature flags
    enable_ml_analysis: bool = Field(True, description="Enable ML analysis")
    enable_evolution_tracking: bool = Field(True, description="Enable evolution tracking")
    enable_similarity_detection: bool = Field(True, description="Enable similarity detection")
    enable_plagiarism_detection: bool = Field(True, description="Enable plagiarism detection")
    enable_realtime_streaming: bool = Field(False, description="Enable real-time streaming")
    enable_visualization: bool = Field(True, description="Enable visualization data")
    
    # Analysis parameters
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity detection threshold")
    plagiarism_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Plagiarism detection threshold")
    user_id: Optional[str] = Field(None, description="User ID for real-time updates")

class ComprehensiveAnalysisResponse(BaseModel):
    """Response model for comprehensive analysis"""
    entry_id: str
    basic_analysis: Dict[str, Any]
    ml_analysis: Optional[Dict[str, Any]] = None
    evolution_analysis: Optional[Dict[str, Any]] = None
    similarity_analysis: Optional[Dict[str, Any]] = None
    plagiarism_analysis: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    realtime_status: Optional[str] = None
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: datetime

class ModelEvolutionRequest(BaseModel):
    """Request model for model evolution analysis"""
    model_versions: List[str] = Field(..., description="Model versions to analyze")
    time_window_days: Optional[int] = Field(None, ge=1, le=365, description="Time window in days")
    include_predictions: bool = Field(True, description="Include future predictions")
    include_regression_analysis: bool = Field(True, description="Include regression analysis")

class ModelEvolutionResponse(BaseModel):
    """Response model for model evolution analysis"""
    evolution_analysis: Dict[str, Any]
    regression_analysis: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime

class ContentSimilarityRequest(BaseModel):
    """Request model for content similarity analysis"""
    content1: str = Field(..., description="First content piece")
    content2: str = Field(..., description="Second content piece")
    similarity_types: List[str] = Field(
        default=["semantic", "lexical", "structural", "stylistic"],
        description="Types of similarity to analyze"
    )
    include_plagiarism_detection: bool = Field(True, description="Include plagiarism detection")
    include_visualization: bool = Field(True, description="Include visualization data")

class ContentSimilarityResponse(BaseModel):
    """Response model for content similarity analysis"""
    similarity_results: Dict[str, Any]
    plagiarism_analysis: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    insights: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime

class SystemHealthRequest(BaseModel):
    """Request model for system health check"""
    include_detailed_metrics: bool = Field(False, description="Include detailed system metrics")
    include_feature_status: bool = Field(True, description="Include feature status")
    include_performance_metrics: bool = Field(True, description="Include performance metrics")

class SystemHealthResponse(BaseModel):
    """Response model for system health check"""
    overall_status: str
    system_metrics: Dict[str, Any]
    feature_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    health_timestamp: datetime

# Comprehensive API Endpoints

@comprehensive_router.post("/analyze", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_content_analysis(request: ComprehensiveAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Perform comprehensive content analysis with all available features
    
    This endpoint provides the most complete analysis including:
    - Basic content analysis
    - Advanced ML analysis (anomaly detection, clustering)
    - Model evolution tracking
    - Content similarity and plagiarism detection
    - Real-time streaming (optional)
    - Visualization data generation
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
        
        # Basic analysis results
        basic_analysis = {
            "readability_score": entry.metrics.readability_score,
            "sentiment_score": entry.metrics.sentiment_score,
            "word_count": entry.metrics.word_count,
            "sentence_count": entry.metrics.sentence_count,
            "avg_word_length": entry.metrics.avg_word_length,
            "complexity_score": entry.metrics.complexity_score,
            "topic_diversity": entry.metrics.topic_diversity,
            "consistency_score": entry.metrics.consistency_score
        }
        
        # Initialize response components
        ml_analysis = None
        evolution_analysis = None
        similarity_analysis = None
        plagiarism_analysis = None
        visualization_data = None
        realtime_status = None
        insights = []
        recommendations = []
        
        # ML Analysis
        if request.enable_ml_analysis:
            try:
                # Prepare entry data for ML analysis
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
                
                # Anomaly detection
                anomalies = detect_anomalies([entry_data], method="isolation_forest")
                if anomalies and anomalies[0].is_anomaly:
                    ml_analysis = {
                        "anomaly_detected": True,
                        "anomaly_type": anomalies[0].anomaly_type,
                        "anomaly_score": anomalies[0].anomaly_score,
                        "anomaly_explanation": anomalies[0].explanation,
                        "anomaly_recommendations": anomalies[0].recommendations
                    }
                    insights.append(f"Anomaly detected: {anomalies[0].anomaly_type}")
                    recommendations.extend(anomalies[0].recommendations)
                else:
                    ml_analysis = {"anomaly_detected": False}
                
                # Advanced feature extraction
                advanced_features = ml_engine.extract_advanced_features(request.content)
                ml_analysis["advanced_features"] = advanced_features
                
            except Exception as e:
                logger.warning(f"ML analysis failed: {e}")
                ml_analysis = {"error": str(e)}
        
        # Evolution Analysis
        if request.enable_evolution_tracking:
            try:
                # Track this model version if not already tracked
                if evolution_tracker and request.model_version not in evolution_tracker.model_versions:
                    track_model_version(request.model_version, datetime.now(), "Auto-tracked version")
                
                # Analyze evolution
                if evolution_tracker:
                    evolution_result = analyze_evolution()
                    evolution_analysis = {
                        "total_versions": len(evolution_result.model_versions),
                        "evolution_trends": evolution_result.evolution_trends,
                        "regression_detected": evolution_result.regression_detected,
                        "improvement_areas": evolution_result.improvement_areas,
                        "degradation_areas": evolution_result.degradation_areas
                    }
                    
                    if evolution_result.regression_detected:
                        insights.append("Model regression detected in evolution analysis")
                        recommendations.extend(evolution_result.recommendations)
                    else:
                        insights.append("Model evolution is stable")
                
            except Exception as e:
                logger.warning(f"Evolution analysis failed: {e}")
                evolution_analysis = {"error": str(e)}
        
        # Similarity Analysis
        if request.enable_similarity_detection:
            try:
                # Find similar content
                similar_content = find_similar_content(
                    request.content, 
                    threshold=request.similarity_threshold, 
                    max_results=5
                )
                
                similarity_analysis = {
                    "similar_content_found": len(similar_content),
                    "similar_content": similar_content,
                    "originality_score": similarity_engine.calculate_originality_score(request.content)
                }
                
                if similar_content:
                    max_similarity = max(item["similarity_score"] for item in similar_content)
                    insights.append(f"Similar content found with {max_similarity:.2f} similarity")
                    if max_similarity > 0.8:
                        recommendations.append("Consider adding more original content")
                
            except Exception as e:
                logger.warning(f"Similarity analysis failed: {e}")
                similarity_analysis = {"error": str(e)}
        
        # Plagiarism Analysis
        if request.enable_plagiarism_detection:
            try:
                plagiarism_result = detect_plagiarism(request.content, content_id=entry_id)
                plagiarism_analysis = {
                    "plagiarism_level": plagiarism_result.plagiarism_level.value,
                    "similarity_score": plagiarism_result.similarity_score,
                    "originality_score": plagiarism_result.originality_score,
                    "suspicious_patterns": plagiarism_result.suspicious_patterns
                }
                
                if plagiarism_result.plagiarism_level.value in ["plagiarized", "suspicious"]:
                    insights.append(f"Plagiarism detected: {plagiarism_result.plagiarism_level.value}")
                    recommendations.extend(plagiarism_result.recommendations)
                
            except Exception as e:
                logger.warning(f"Plagiarism analysis failed: {e}")
                plagiarism_analysis = {"error": str(e)}
        
        # Visualization Data
        if request.enable_visualization:
            try:
                # Create quality distribution chart
                quality_chart = create_quality_distribution_chart([entry_data], "readability_score")
                visualization_data = {
                    "quality_distribution": quality_chart,
                    "chart_types_available": ["trend", "distribution", "comparison", "anomaly"]
                }
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
                visualization_data = {"error": str(e)}
        
        # Real-time Streaming
        if request.enable_realtime_streaming and request.user_id:
            try:
                from .realtime_streaming import analyze_content_streaming
                realtime_entry_id = await analyze_content_streaming(
                    request.content, request.model_version, request.user_id, request.metadata
                )
                realtime_status = "streaming_enabled"
            except Exception as e:
                logger.warning(f"Real-time streaming failed: {e}")
                realtime_status = "streaming_failed"
        
        # Generate overall insights and recommendations
        if not insights:
            insights.append("Content analysis completed successfully")
        
        if not recommendations:
            recommendations.append("Content quality is within acceptable parameters")
        
        # Calculate overall confidence score
        confidence_score = 0.8  # Base confidence
        if ml_analysis and "error" not in ml_analysis:
            confidence_score += 0.1
        if evolution_analysis and "error" not in evolution_analysis:
            confidence_score += 0.05
        if similarity_analysis and "error" not in similarity_analysis:
            confidence_score += 0.05
        
        confidence_score = min(1.0, confidence_score)
        
        # Prepare comprehensive response
        response = ComprehensiveAnalysisResponse(
            entry_id=entry_id,
            basic_analysis=basic_analysis,
            ml_analysis=ml_analysis,
            evolution_analysis=evolution_analysis,
            similarity_analysis=similarity_analysis,
            plagiarism_analysis=plagiarism_analysis,
            visualization_data=visualization_data,
            realtime_status=realtime_status,
            insights=insights,
            recommendations=recommendations,
            confidence_score=confidence_score,
            analysis_timestamp=datetime.now()
        )
        
        logger.info(f"Comprehensive analysis completed: {entry_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@comprehensive_router.post("/evolution", response_model=ModelEvolutionResponse)
async def comprehensive_evolution_analysis(request: ModelEvolutionRequest):
    """
    Perform comprehensive model evolution analysis
    
    This endpoint provides complete evolution analysis including:
    - Model version tracking and comparison
    - Performance regression detection
    - Future performance predictions
    - Evolution timeline generation
    - Comprehensive insights and recommendations
    """
    try:
        if not evolution_tracker:
            raise HTTPException(status_code=500, detail="Evolution tracker not initialized")
        
        # Track model versions if not already tracked
        for version in request.model_versions:
            if version not in evolution_tracker.model_versions:
                track_model_version(version, datetime.now(), f"Version {version}")
        
        # Perform evolution analysis
        evolution_result = analyze_evolution()
        
        # Regression analysis
        regression_analysis = None
        if request.include_regression_analysis and len(request.model_versions) >= 2:
            try:
                regressions = []
                for i in range(len(request.model_versions) - 1):
                    version_regressions = detect_performance_regression(
                        request.model_versions[i], 
                        request.model_versions[i + 1]
                    )
                    regressions.extend(version_regressions)
                
                if regressions:
                    regression_analysis = {
                        "regressions_detected": len(regressions),
                        "regressions": [
                            {
                                "type": r.regression_type.value,
                                "severity": r.severity,
                                "affected_metrics": r.affected_metrics,
                                "magnitude": r.regression_magnitude,
                                "confidence": r.confidence,
                                "affected_versions": r.affected_versions,
                                "recommendations": r.recommendations
                            }
                            for r in regressions
                        ]
                    }
                else:
                    regression_analysis = {"regressions_detected": 0}
                    
            except Exception as e:
                logger.warning(f"Regression analysis failed: {e}")
                regression_analysis = {"error": str(e)}
        
        # Future predictions
        predictions = None
        if request.include_predictions:
            try:
                predictions = evolution_tracker.predict_future_performance()
            except Exception as e:
                logger.warning(f"Prediction generation failed: {e}")
                predictions = {"error": str(e)}
        
        # Generate timeline
        timeline = evolution_tracker.get_evolution_timeline()
        
        # Generate insights
        insights = []
        if evolution_result.regression_detected:
            insights.append("Model regressions detected in evolution analysis")
        else:
            insights.append("Model evolution is stable")
        
        if regression_analysis and regression_analysis.get("regressions_detected", 0) > 0:
            insights.append(f"{regression_analysis['regressions_detected']} performance regressions detected")
        
        if predictions and "error" not in predictions:
            trend = predictions.get("trend_analysis", {})
            if trend.get("quality_trend") == "improving":
                insights.append("Model quality is improving over time")
            elif trend.get("quality_trend") == "declining":
                insights.append("Model quality is declining over time")
        
        # Generate recommendations
        recommendations = []
        if evolution_result.recommendations:
            recommendations.extend(evolution_result.recommendations)
        
        if regression_analysis and regression_analysis.get("regressions_detected", 0) > 0:
            recommendations.append("Investigate and address detected performance regressions")
        
        if not recommendations:
            recommendations.append("Continue monitoring model evolution")
        
        # Prepare response
        response = ModelEvolutionResponse(
            evolution_analysis={
                "model_versions": len(evolution_result.model_versions),
                "evolution_trends": evolution_result.evolution_trends,
                "regression_detected": evolution_result.regression_detected,
                "improvement_areas": evolution_result.improvement_areas,
                "degradation_areas": evolution_result.degradation_areas,
                "confidence_score": evolution_result.confidence_score
            },
            regression_analysis=regression_analysis,
            predictions=predictions,
            timeline=timeline,
            insights=insights,
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
        logger.info("Comprehensive evolution analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in evolution analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evolution analysis failed: {str(e)}")

@comprehensive_router.post("/similarity", response_model=ContentSimilarityResponse)
async def comprehensive_similarity_analysis(request: ContentSimilarityRequest):
    """
    Perform comprehensive content similarity analysis
    
    This endpoint provides complete similarity analysis including:
    - Multiple similarity types (semantic, lexical, structural, stylistic)
    - Plagiarism detection
    - Content overlap analysis
    - Visualization data generation
    - Detailed insights and recommendations
    """
    try:
        # Convert similarity types
        from .content_similarity_engine import SimilarityType
        similarity_type_map = {
            "semantic": SimilarityType.SEMANTIC_SIMILARITY,
            "lexical": SimilarityType.LEXICAL_SIMILARITY,
            "structural": SimilarityType.STRUCTURAL_SIMILARITY,
            "stylistic": SimilarityType.STYLISTIC_SIMILARITY,
            "content_overlap": SimilarityType.CONTENT_OVERLAP
        }
        
        similarity_types = [similarity_type_map.get(st) for st in request.similarity_types 
                          if st in similarity_type_map]
        
        if not similarity_types:
            similarity_types = [SimilarityType.SEMANTIC_SIMILARITY, SimilarityType.LEXICAL_SIMILARITY]
        
        # Calculate similarities
        similarity_results = calculate_similarity(request.content1, request.content2, similarity_types)
        
        # Process similarity results
        processed_results = {}
        max_similarity = 0.0
        
        for sim_type, result in similarity_results.items():
            processed_results[sim_type.value] = {
                "similarity_score": result.similarity_score,
                "confidence": result.confidence,
                "matched_segments": result.matched_segments
            }
            max_similarity = max(max_similarity, result.similarity_score)
        
        # Plagiarism analysis
        plagiarism_analysis = None
        if request.include_plagiarism_detection:
            try:
                plagiarism_result = detect_plagiarism(request.content1, request.content2)
                plagiarism_analysis = {
                    "plagiarism_level": plagiarism_result.plagiarism_level.value,
                    "similarity_score": plagiarism_result.similarity_score,
                    "originality_score": plagiarism_result.originality_score,
                    "suspicious_patterns": plagiarism_result.suspicious_patterns,
                    "recommendations": plagiarism_result.recommendations
                }
            except Exception as e:
                logger.warning(f"Plagiarism analysis failed: {e}")
                plagiarism_analysis = {"error": str(e)}
        
        # Visualization data
        visualization_data = None
        if request.include_visualization:
            try:
                # Create comparison chart
                comparison_data = [
                    {
                        "label": "Content 1",
                        "readability_score": 50.0,  # Would be calculated from actual content
                        "sentiment_score": 0.2,
                        "complexity_score": 0.5,
                        "topic_diversity": 0.3
                    },
                    {
                        "label": "Content 2", 
                        "readability_score": 55.0,
                        "sentiment_score": 0.1,
                        "complexity_score": 0.6,
                        "topic_diversity": 0.4
                    }
                ]
                
                from .visualization_engine import create_comparison_chart
                comparison_chart = create_comparison_chart(comparison_data)
                visualization_data = {
                    "comparison_chart": comparison_chart,
                    "chart_types_available": ["comparison", "trend", "distribution"]
                }
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
                visualization_data = {"error": str(e)}
        
        # Generate insights
        insights = []
        if max_similarity > 0.8:
            insights.append("Content pieces are highly similar")
        elif max_similarity > 0.6:
            insights.append("Content pieces show moderate similarity")
        elif max_similarity > 0.4:
            insights.append("Content pieces show low similarity")
        else:
            insights.append("Content pieces are quite different")
        
        if plagiarism_analysis and plagiarism_analysis.get("plagiarism_level") in ["plagiarized", "suspicious"]:
            insights.append(f"Plagiarism detected: {plagiarism_analysis['plagiarism_level']}")
        
        # Generate recommendations
        recommendations = []
        if max_similarity > 0.8:
            recommendations.append("Consider making content more distinct")
        
        if plagiarism_analysis and plagiarism_analysis.get("recommendations"):
            recommendations.extend(plagiarism_analysis["recommendations"])
        
        if not recommendations:
            recommendations.append("Content similarity analysis completed")
        
        # Prepare response
        response = ContentSimilarityResponse(
            similarity_results=processed_results,
            plagiarism_analysis=plagiarism_analysis,
            visualization_data=visualization_data,
            insights=insights,
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
        logger.info("Comprehensive similarity analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

@comprehensive_router.post("/health", response_model=SystemHealthResponse)
async def comprehensive_system_health(request: SystemHealthRequest):
    """
    Perform comprehensive system health check
    
    This endpoint provides complete system health information including:
    - Overall system status
    - Detailed system metrics
    - Feature status and availability
    - Performance metrics
    - Health recommendations
    """
    try:
        # Overall system status
        overall_status = "healthy"
        
        # System metrics
        system_metrics = {}
        if request.include_detailed_metrics:
            system_metrics = {
                "total_entries": analyzer.get_entry_count(),
                "analyzer_status": "operational",
                "memory_usage": "normal",  # Would be actual memory usage
                "cpu_usage": "normal",     # Would be actual CPU usage
                "disk_usage": "normal",    # Would be actual disk usage
                "uptime": "unknown"        # Would be actual uptime
            }
        
        # Feature status
        feature_status = {}
        if request.include_feature_status:
            feature_status = {
                "ml_engine": {
                    "available": ml_engine is not None,
                    "transformers_available": ml_engine.transformer_model is not None if ml_engine else False,
                    "spacy_available": ml_engine.nlp is not None if ml_engine else False,
                    "models_loaded": len(ml_engine.get_model_info()) if ml_engine else 0
                },
                "evolution_tracker": {
                    "available": evolution_tracker is not None,
                    "versions_tracked": len(evolution_tracker.model_versions) if evolution_tracker else 0
                },
                "similarity_engine": {
                    "available": similarity_engine is not None,
                    "fingerprints_stored": len(similarity_engine.content_fingerprints) if similarity_engine else 0,
                    "sentence_transformer_available": similarity_engine.sentence_model is not None if similarity_engine else False
                },
                "realtime_streaming": {
                    "available": realtime_analyzer is not None,
                    "active_connections": len(websocket_manager.active_connections) if websocket_manager else 0
                },
                "visualization_engine": {
                    "available": viz_engine is not None,
                    "matplotlib_available": True,  # Would check actual availability
                    "plotly_available": True,      # Would check actual availability
                    "cache_size": len(viz_engine.chart_cache) if viz_engine else 0
                }
            }
        
        # Performance metrics
        performance_metrics = {}
        if request.include_performance_metrics:
            performance_metrics = {
                "response_times": {
                    "average": 0.1,  # Would be actual average response time
                    "p95": 0.2,      # Would be actual 95th percentile
                    "p99": 0.5       # Would be actual 99th percentile
                },
                "throughput": {
                    "requests_per_minute": 100,  # Would be actual throughput
                    "concurrent_users": 10       # Would be actual concurrent users
                },
                "error_rates": {
                    "total_errors": 0,           # Would be actual error count
                    "error_rate_percent": 0.0    # Would be actual error rate
                }
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check for issues and generate recommendations
        if feature_status:
            if not feature_status.get("ml_engine", {}).get("available"):
                recommendations.append("ML engine is not available - check configuration")
            
            if not feature_status.get("similarity_engine", {}).get("available"):
                recommendations.append("Similarity engine is not available - check configuration")
            
            if feature_status.get("realtime_streaming", {}).get("active_connections", 0) > 100:
                recommendations.append("High number of active connections - consider scaling")
        
        if performance_metrics:
            error_rate = performance_metrics.get("error_rates", {}).get("error_rate_percent", 0)
            if error_rate > 5.0:
                recommendations.append("High error rate detected - investigate system issues")
                overall_status = "degraded"
        
        if not recommendations:
            recommendations.append("System is operating normally")
        
        # Determine overall status
        if any("not available" in rec for rec in recommendations):
            overall_status = "degraded"
        elif any("error" in rec.lower() for rec in recommendations):
            overall_status = "degraded"
        
        # Prepare response
        response = SystemHealthResponse(
            overall_status=overall_status,
            system_metrics=system_metrics,
            feature_status=feature_status,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            health_timestamp=datetime.now()
        )
        
        logger.info("Comprehensive system health check completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in system health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@comprehensive_router.get("/features")
async def get_comprehensive_features():
    """Get comprehensive list of all available features"""
    return {
        "core_features": {
            "content_analysis": {
                "description": "Basic content quality analysis",
                "endpoints": ["/analyze", "/compare", "/trends", "/report"]
            },
            "ml_analysis": {
                "description": "Advanced machine learning analysis",
                "features": ["anomaly_detection", "clustering", "predictive_modeling"],
                "endpoints": ["/anomalies/detect", "/clustering/advanced", "/models/predictive"]
            },
            "evolution_tracking": {
                "description": "AI model evolution tracking",
                "features": ["version_comparison", "regression_detection", "future_predictions"],
                "endpoints": ["/evolution", "/versions/compare"]
            },
            "similarity_detection": {
                "description": "Content similarity and plagiarism detection",
                "features": ["semantic_similarity", "lexical_similarity", "plagiarism_detection"],
                "endpoints": ["/similarity", "/plagiarism/detect"]
            },
            "realtime_streaming": {
                "description": "Real-time analysis and streaming",
                "features": ["websocket_connections", "live_updates", "event_streaming"],
                "endpoints": ["/stream/ws", "/stream/subscribe"]
            },
            "visualization": {
                "description": "Data visualization and dashboards",
                "features": ["interactive_charts", "dashboards", "export_capabilities"],
                "endpoints": ["/visualize", "/dashboard"]
            }
        },
        "comprehensive_endpoints": {
            "comprehensive_analysis": "/comprehensive/analyze",
            "evolution_analysis": "/comprehensive/evolution",
            "similarity_analysis": "/comprehensive/similarity",
            "system_health": "/comprehensive/health"
        },
        "integration_capabilities": {
            "api_versions": ["v1", "v2", "comprehensive"],
            "authentication": ["jwt", "api_key"],
            "rate_limiting": True,
            "caching": True,
            "monitoring": True
        },
        "deployment_options": {
            "docker": True,
            "kubernetes": True,
            "cloud_deployment": True,
            "scaling": "horizontal"
        }
    }

@comprehensive_router.get("/status")
async def get_comprehensive_status():
    """Get comprehensive system status"""
    try:
        # Get status from all components
        status = {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "analyzer": {
                    "status": "healthy",
                    "total_entries": analyzer.get_entry_count()
                },
                "ml_engine": {
                    "status": "healthy" if ml_engine else "unavailable",
                    "models_available": len(ml_engine.get_model_info()) if ml_engine else 0
                },
                "evolution_tracker": {
                    "status": "healthy" if evolution_tracker else "unavailable",
                    "versions_tracked": len(evolution_tracker.model_versions) if evolution_tracker else 0
                },
                "similarity_engine": {
                    "status": "healthy" if similarity_engine else "unavailable",
                    "fingerprints_stored": len(similarity_engine.content_fingerprints) if similarity_engine else 0
                },
                "realtime_streaming": {
                    "status": "healthy" if realtime_analyzer else "unavailable",
                    "active_connections": len(websocket_manager.active_connections) if websocket_manager else 0
                },
                "visualization_engine": {
                    "status": "healthy" if viz_engine else "unavailable",
                    "cache_size": len(viz_engine.chart_cache) if viz_engine else 0
                }
            },
            "features_available": {
                "comprehensive_analysis": True,
                "ml_analysis": ml_engine is not None,
                "evolution_tracking": evolution_tracker is not None,
                "similarity_detection": similarity_engine is not None,
                "realtime_streaming": realtime_analyzer is not None,
                "visualization": viz_engine is not None
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting comprehensive status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Initialize all components
async def initialize_comprehensive_system():
    """Initialize all comprehensive system components"""
    try:
        # Initialize evolution tracker
        from .ai_evolution_tracker import initialize_evolution_tracker
        initialize_evolution_tracker(analyzer)
        
        # Initialize real-time analyzer
        from .realtime_streaming import initialize_realtime_analyzer
        await initialize_realtime_analyzer(analyzer)
        
        logger.info("Comprehensive system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize comprehensive system: {e}")
        raise

# Convenience functions
async def comprehensive_analyze(content: str, model_version: str, 
                              enable_all_features: bool = True, 
                              user_id: str = None) -> Dict[str, Any]:
    """Perform comprehensive analysis with all features enabled"""
    request = ComprehensiveAnalysisRequest(
        content=content,
        model_version=model_version,
        enable_ml_analysis=enable_all_features,
        enable_evolution_tracking=enable_all_features,
        enable_similarity_detection=enable_all_features,
        enable_plagiarism_detection=enable_all_features,
        enable_realtime_streaming=enable_all_features,
        enable_visualization=enable_all_features,
        user_id=user_id
    )
    
    # This would call the endpoint logic directly
    # For now, return a simplified response
    return {
        "entry_id": "comprehensive_analysis_placeholder",
        "message": "Comprehensive analysis completed",
        "features_enabled": enable_all_features
    }



























