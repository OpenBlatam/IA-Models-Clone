"""
Enhanced NLP API
================

API REST mejorada para el sistema NLP con optimizaciones avanzadas,
métricas en tiempo real, análisis de tendencias y caché inteligente.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .enhanced_nlp_system import enhanced_nlp_system, EnhancedAnalysisResult
from .nlp_cache import nlp_cache
from .nlp_metrics import nlp_monitoring
from .nlp_trends import nlp_trend_analyzer
from .exceptions import NLPProcessingError, ModelLoadError

logger = logging.getLogger(__name__)

# Create enhanced router
router = APIRouter(prefix="/enhanced-nlp", tags=["Enhanced NLP"])

# Enhanced request/response models
class EnhancedAnalysisRequest(BaseModel):
    """Request model for enhanced text analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=50000)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use intelligent caching")
    quality_check: bool = Field(default=True, description="Perform quality assessment")
    include_trends: bool = Field(default=False, description="Include trend analysis")
    include_anomalies: bool = Field(default=False, description="Include anomaly detection")
    include_predictions: bool = Field(default=False, description="Include predictions")

class EnhancedAnalysisResponse(BaseModel):
    """Response model for enhanced text analysis."""
    text: str
    language: str
    analysis: Dict[str, Any]
    processing_time: float
    cache_hit: bool
    quality_score: float
    confidence: float
    recommendations: List[str]
    trends: Optional[Dict[str, Any]] = None
    anomalies: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(..., description="Texts to analyze", min_items=1, max_items=100)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use intelligent caching")
    quality_check: bool = Field(default=True, description="Perform quality assessment")
    parallel_processing: bool = Field(default=True, description="Process in parallel")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[EnhancedAnalysisResponse]
    total_processing_time: float
    average_processing_time: float
    success_count: int
    error_count: int
    cache_hit_rate: float
    average_quality_score: float
    timestamp: datetime

class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    system: Dict[str, Any]
    cache: Dict[str, Any]
    monitoring: Dict[str, Any]
    trends: Dict[str, Any]
    health: Dict[str, Any]
    timestamp: datetime

class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""
    metric_names: Optional[List[str]] = Field(default=None, description="Specific metrics to analyze")
    hours: int = Field(default=24, description="Time window for analysis", ge=1, le=168)
    include_predictions: bool = Field(default=False, description="Include future predictions")

class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""
    trends: Dict[str, Any]
    anomalies: Dict[str, Any]
    predictions: Optional[Dict[str, Any]] = None
    insights: List[str]
    processing_time: float
    timestamp: datetime

class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    text: str = Field(..., description="Text to assess", min_length=1, max_length=10000)
    language: Optional[str] = Field(default="en", description="Language code")
    assessment_criteria: List[str] = Field(
        default=["readability", "sentiment", "entities", "keywords"],
        description="Criteria for assessment"
    )

class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    text: str
    overall_score: float
    criteria_scores: Dict[str, float]
    recommendations: List[str]
    processing_time: float
    timestamp: datetime

# Dependency to ensure enhanced NLP system is initialized
async def get_enhanced_nlp_system():
    """Get initialized enhanced NLP system."""
    if not enhanced_nlp_system.is_initialized:
        await enhanced_nlp_system.initialize()
    return enhanced_nlp_system

# Enhanced API Endpoints

@router.get("/health", response_model=SystemStatusResponse)
async def get_enhanced_health():
    """Get comprehensive system health status."""
    try:
        status = await enhanced_nlp_system.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_text_enhanced(request: EnhancedAnalysisRequest):
    """Perform enhanced text analysis with optimizations."""
    try:
        result = await enhanced_nlp_system.analyze_text_enhanced(
            text=request.text,
            language=request.language or "en",
            use_cache=request.use_cache,
            quality_check=request.quality_check,
            include_trends=request.include_trends
        )
        
        return EnhancedAnalysisResponse(
            text=result.text,
            language=result.language,
            analysis=result.analysis,
            processing_time=result.processing_time,
            cache_hit=result.cache_hit,
            quality_score=result.quality_score,
            confidence=result.confidence,
            recommendations=result.recommendations or [],
            trends=result.trends,
            anomalies=result.anomalies,
            predictions=result.predictions,
            timestamp=result.timestamp
        )
        
    except NLPProcessingError as e:
        logger.error(f"NLP processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {e}")

@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze_enhanced(request: BatchAnalysisRequest):
    """Perform enhanced batch analysis."""
    try:
        start_time = datetime.now()
        
        results = await enhanced_nlp_system.batch_analyze_enhanced(
            texts=request.texts,
            language=request.language or "en",
            use_cache=request.use_cache,
            quality_check=request.quality_check
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        success_count = len([r for r in results if r.quality_score > 0])
        error_count = len(results) - success_count
        cache_hits = len([r for r in results if r.cache_hit])
        cache_hit_rate = cache_hits / len(results) if results else 0
        avg_quality = sum(r.quality_score for r in results) / len(results) if results else 0
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append(EnhancedAnalysisResponse(
                text=result.text,
                language=result.language,
                analysis=result.analysis,
                processing_time=result.processing_time,
                cache_hit=result.cache_hit,
                quality_score=result.quality_score,
                confidence=result.confidence,
                recommendations=result.recommendations or [],
                timestamp=result.timestamp
            ))
        
        return BatchAnalysisResponse(
            results=response_results,
            total_processing_time=total_time,
            average_processing_time=total_time / len(results) if results else 0,
            success_count=success_count,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            average_quality_score=avg_quality,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")

@router.post("/trends", response_model=TrendAnalysisResponse)
async def analyze_trends(request: TrendAnalysisRequest):
    """Analyze trends and patterns in NLP metrics."""
    try:
        start_time = datetime.now()
        
        # Analyze trends
        trends = await nlp_trend_analyzer.analyze_trends(hours=request.hours)
        
        # Detect anomalies
        anomalies = await nlp_trend_analyzer.detect_anomalies(hours=request.hours)
        
        # Generate predictions if requested
        predictions = None
        if request.include_predictions:
            predictions = await nlp_trend_analyzer.generate_predictions(hours=24)
        
        # Get insights
        insights = nlp_trend_analyzer.get_insights(hours=request.hours)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TrendAnalysisResponse(
            trends=trends,
            anomalies=anomalies,
            predictions=predictions,
            insights=insights,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {e}")

@router.post("/quality", response_model=QualityAssessmentResponse)
async def assess_quality(request: QualityAssessmentRequest):
    """Assess text quality using multiple criteria."""
    try:
        start_time = datetime.now()
        
        # Perform analysis
        result = await enhanced_nlp_system.analyze_text_enhanced(
            text=request.text,
            language=request.language or "en",
            use_cache=True,
            quality_check=True,
            include_trends=False
        )
        
        # Assess quality based on criteria
        criteria_scores = {}
        overall_score = 0.0
        
        for criterion in request.assessment_criteria:
            if criterion == "readability":
                readability = result.analysis.get('readability', {})
                score = readability.get('average_score', 0) / 100
                criteria_scores[criterion] = score
                overall_score += score
            elif criterion == "sentiment":
                sentiment = result.analysis.get('sentiment', {})
                if sentiment.get('ensemble', {}).get('confidence'):
                    score = sentiment['ensemble']['confidence']
                    criteria_scores[criterion] = score
                    overall_score += score
            elif criterion == "entities":
                entities = result.analysis.get('entities', [])
                score = min(1.0, len(entities) / 10)  # Normalize to 0-1
                criteria_scores[criterion] = score
                overall_score += score
            elif criterion == "keywords":
                keywords = result.analysis.get('keywords', [])
                score = min(1.0, len(keywords) / 15)  # Normalize to 0-1
                criteria_scores[criterion] = score
                overall_score += score
        
        # Calculate overall score
        if criteria_scores:
            overall_score = overall_score / len(criteria_scores)
        
        # Generate recommendations
        recommendations = []
        for criterion, score in criteria_scores.items():
            if score < 0.5:
                if criterion == "readability":
                    recommendations.append("Improve text readability by using simpler language")
                elif criterion == "sentiment":
                    recommendations.append("Clarify sentiment by using more explicit language")
                elif criterion == "entities":
                    recommendations.append("Add more specific entities and proper nouns")
                elif criterion == "keywords":
                    recommendations.append("Include more relevant keywords for better SEO")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityAssessmentResponse(
            text=request.text,
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            recommendations=recommendations,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {e}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_enhanced_metrics():
    """Get comprehensive system metrics."""
    try:
        # Performance metrics
        performance = nlp_monitoring.get_performance_metrics()
        
        # System metrics
        system = nlp_monitoring.get_system_metrics()
        
        # Health status
        health = nlp_monitoring.get_health_status()
        
        # Cache metrics
        cache_stats = nlp_cache.get_stats()
        cache_memory = nlp_cache.get_memory_usage()
        
        # Trend summary
        trend_summary = nlp_trend_analyzer.get_trend_summary()
        anomaly_summary = nlp_trend_analyzer.get_anomaly_summary()
        
        return {
            'performance': performance,
            'system': system,
            'health': health,
            'cache': {
                'stats': cache_stats,
                'memory': cache_memory
            },
            'trends': {
                'summary': trend_summary,
                'anomalies': anomaly_summary
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    resolved: Optional[bool] = Query(None, description="Filter by resolution status")
):
    """Get system alerts."""
    try:
        alert_level = None
        if level:
            from .nlp_metrics import AlertLevel
            alert_level = AlertLevel(level)
        
        alerts = nlp_monitoring.get_alerts(level=alert_level, resolved=resolved)
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve a specific alert."""
    try:
        await nlp_monitoring.resolve_alert(alert_id)
        return {"message": f"Alert {alert_id} resolved", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {e}")

@router.post("/cache/optimize")
async def optimize_cache():
    """Manually trigger cache optimization."""
    try:
        await nlp_cache.optimize()
        cache_stats = nlp_cache.get_stats()
        
        return {
            "message": "Cache optimization completed",
            "stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache optimization failed: {e}")

@router.post("/cache/invalidate")
async def invalidate_cache(pattern: Optional[str] = None):
    """Invalidate cache entries."""
    try:
        await nlp_cache.invalidate(pattern)
        
        return {
            "message": f"Cache invalidated for pattern: {pattern or 'all'}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {e}")

@router.get("/insights", response_model=List[str])
async def get_insights(hours: int = Query(24, description="Time window for insights")):
    """Get system insights and recommendations."""
    try:
        insights = nlp_trend_analyzer.get_insights(hours=hours)
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {e}")

# Utility endpoints
@router.get("/supported-languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages."""
    return ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"]

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_system_capabilities():
    """Get system capabilities and features."""
    return {
        "analysis_types": [
            "sentiment_analysis",
            "entity_extraction", 
            "keyword_extraction",
            "topic_modeling",
            "readability_analysis",
            "language_detection",
            "text_classification",
            "summarization",
            "translation"
        ],
        "optimizations": [
            "intelligent_caching",
            "quality_assessment",
            "trend_analysis",
            "anomaly_detection",
            "batch_processing",
            "parallel_processing"
        ],
        "monitoring": [
            "real_time_metrics",
            "performance_tracking",
            "quality_monitoring",
            "alert_system",
            "trend_analysis"
        ],
        "languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"],
        "max_text_length": 50000,
        "max_batch_size": 100,
        "cache_enabled": True,
        "gpu_acceleration": True
    }












