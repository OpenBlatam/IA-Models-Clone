"""
Ultra Quality NLP API
====================

API REST ultra-calidad para análisis de máxima precisión
y evaluación exhaustiva de resultados.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .ultra_quality_nlp import ultra_quality_nlp, UltraQualityResult

logger = logging.getLogger(__name__)

# Create ultra-quality router
router = APIRouter(prefix="/ultra-quality", tags=["Ultra Quality NLP"])

# Ultra-quality request/response models
class UltraQualityRequest(BaseModel):
    """Request model for ultra-quality analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=50000)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use ultra-quality caching")
    quality_check: bool = Field(default=True, description="Perform quality assessment")
    ensemble_validation: bool = Field(default=True, description="Perform ensemble validation")
    cross_validation: bool = Field(default=True, description="Perform cross-validation")

class UltraQualityResponse(BaseModel):
    """Response model for ultra-quality analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    quality_score: float
    confidence_score: float
    ensemble_validation: Dict[str, Any]
    cross_validation: Dict[str, Any]
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class UltraQualityBatchRequest(BaseModel):
    """Request model for ultra-quality batch analysis."""
    texts: List[str] = Field(..., description="Texts to analyze", min_items=1, max_items=100)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use ultra-quality caching")
    quality_check: bool = Field(default=True, description="Perform quality assessment")
    ensemble_validation: bool = Field(default=True, description="Perform ensemble validation")
    cross_validation: bool = Field(default=True, description="Perform cross-validation")
    batch_size: Optional[int] = Field(default=32, description="Batch size for processing", ge=1, le=64)

class UltraQualityBatchResponse(BaseModel):
    """Response model for ultra-quality batch analysis."""
    results: List[UltraQualityResponse]
    total_processing_time: float
    average_processing_time: float
    success_count: int
    error_count: int
    cache_hit_rate: float
    average_quality_score: float
    average_confidence_score: float
    batch_size: int
    timestamp: datetime

class UltraQualityStatusResponse(BaseModel):
    """Response model for ultra-quality system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    quality: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: datetime

class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    text: str = Field(..., description="Text to assess", min_length=1, max_length=10000)
    language: Optional[str] = Field(default="en", description="Language code")
    detailed_assessment: bool = Field(default=True, description="Include detailed assessment")
    ensemble_validation: bool = Field(default=True, description="Include ensemble validation")
    cross_validation: bool = Field(default=True, description="Include cross-validation")

class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    text: str
    overall_quality: float
    sentiment_quality: float
    entity_quality: float
    keyword_quality: float
    topic_quality: float
    readability_quality: float
    confidence_score: float
    ensemble_validation: Dict[str, Any]
    cross_validation: Dict[str, Any]
    recommendations: List[str]
    processing_time: float
    timestamp: datetime

# Dependency to ensure ultra-quality NLP system is initialized
async def get_ultra_quality_nlp_system():
    """Get initialized ultra-quality NLP system."""
    if not ultra_quality_nlp.is_initialized:
        await ultra_quality_nlp.initialize()
    return ultra_quality_nlp

# Ultra-quality API Endpoints

@router.get("/health", response_model=UltraQualityStatusResponse)
async def get_ultra_quality_health():
    """Get ultra-quality system health status."""
    try:
        status = await ultra_quality_nlp.get_ultra_quality_status()
        return UltraQualityStatusResponse(**status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/analyze", response_model=UltraQualityResponse)
async def analyze_ultra_quality(request: UltraQualityRequest):
    """Perform ultra-quality text analysis."""
    try:
        result = await ultra_quality_nlp.analyze_ultra_quality(
            text=request.text,
            language=request.language or "en",
            use_cache=request.use_cache,
            quality_check=request.quality_check,
            ensemble_validation=request.ensemble_validation,
            cross_validation=request.cross_validation
        )
        
        return UltraQualityResponse(
            text=result.text,
            language=result.language,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            readability=result.readability,
            quality_score=result.quality_score,
            confidence_score=result.confidence_score,
            ensemble_validation=result.ensemble_validation,
            cross_validation=result.cross_validation,
            processing_time=result.processing_time,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Ultra-quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-quality analysis failed: {e}")

@router.post("/batch", response_model=UltraQualityBatchResponse)
async def batch_analyze_ultra_quality(request: UltraQualityBatchRequest):
    """Perform ultra-quality batch analysis."""
    try:
        start_time = datetime.now()
        
        # Apply batch size if specified
        if request.batch_size:
            ultra_quality_nlp.config.batch_size = request.batch_size
        
        results = await ultra_quality_nlp.batch_analyze_ultra_quality(
            texts=request.texts,
            language=request.language or "en",
            use_cache=request.use_cache,
            quality_check=request.quality_check,
            ensemble_validation=request.ensemble_validation,
            cross_validation=request.cross_validation
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        success_count = len([r for r in results if r.quality_score > 0])
        error_count = len(results) - success_count
        cache_hits = len([r for r in results if r.cache_hit])
        cache_hit_rate = cache_hits / len(results) if results else 0
        
        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        confidence_scores = [r.confidence_score for r in results if r.confidence_score > 0]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append(UltraQualityResponse(
                text=result.text,
                language=result.language,
                sentiment=result.sentiment,
                entities=result.entities,
                keywords=result.keywords,
                topics=result.topics,
                readability=result.readability,
                quality_score=result.quality_score,
                confidence_score=result.confidence_score,
                ensemble_validation=result.ensemble_validation,
                cross_validation=result.cross_validation,
                processing_time=result.processing_time,
                cache_hit=result.cache_hit,
                timestamp=result.timestamp
            ))
        
        return UltraQualityBatchResponse(
            results=response_results,
            total_processing_time=total_time,
            average_processing_time=total_time / len(results) if results else 0,
            success_count=success_count,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            average_quality_score=avg_quality,
            average_confidence_score=avg_confidence,
            batch_size=request.batch_size or 32,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Ultra-quality batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-quality batch analysis failed: {e}")

@router.post("/quality", response_model=QualityAssessmentResponse)
async def assess_quality_ultra(request: QualityAssessmentRequest):
    """Assess text quality with ultra-quality analysis."""
    try:
        start_time = datetime.now()
        
        # Perform analysis
        result = await ultra_quality_nlp.analyze_ultra_quality(
            text=request.text,
            language=request.language or "en",
            use_cache=True,
            quality_check=True,
            ensemble_validation=request.ensemble_validation,
            cross_validation=request.cross_validation
        )
        
        # Extract quality assessment
        quality_score = result.quality_score
        confidence_score = result.confidence_score
        
        # Calculate individual quality scores
        sentiment_quality = 0.0
        entity_quality = 0.0
        keyword_quality = 0.0
        topic_quality = 0.0
        readability_quality = 0.0
        
        # Assess sentiment quality
        if result.sentiment and 'ensemble' in result.sentiment:
            sentiment_quality = result.sentiment['ensemble'].get('confidence', 0.0)
        
        # Assess entity quality
        if result.entities:
            confidences = [e.get('confidence', 0) for e in result.entities]
            entity_quality = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Assess keyword quality
        if result.keywords:
            keyword_quality = min(1.0, len(result.keywords) / 15)
        
        # Assess topic quality
        if result.topics:
            topic_quality = min(1.0, len(result.topics) / 5)
        
        # Assess readability quality
        if result.readability and 'average_score' in result.readability:
            readability_quality = result.readability['average_score'] / 100
        
        # Generate recommendations
        recommendations = []
        if quality_score < 0.7:
            recommendations.append("Consider improving text quality for better analysis")
        if sentiment_quality < 0.5:
            recommendations.append("Sentiment analysis quality could be improved")
        if entity_quality < 0.5:
            recommendations.append("Entity extraction quality could be enhanced")
        if keyword_quality < 0.5:
            recommendations.append("Keyword extraction could be improved")
        if topic_quality < 0.5:
            recommendations.append("Topic extraction quality could be enhanced")
        if readability_quality < 0.5:
            recommendations.append("Text readability could be improved")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityAssessmentResponse(
            text=request.text,
            overall_quality=quality_score,
            sentiment_quality=sentiment_quality,
            entity_quality=entity_quality,
            keyword_quality=keyword_quality,
            topic_quality=topic_quality,
            readability_quality=readability_quality,
            confidence_score=confidence_score,
            ensemble_validation=result.ensemble_validation,
            cross_validation=result.cross_validation,
            recommendations=recommendations,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {e}")

@router.get("/status", response_model=UltraQualityStatusResponse)
async def get_ultra_quality_status():
    """Get ultra-quality system status."""
    try:
        status = await ultra_quality_nlp.get_ultra_quality_status()
        return UltraQualityStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get ultra-quality status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra-quality status: {e}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_ultra_quality_metrics():
    """Get ultra-quality system metrics."""
    try:
        status = await ultra_quality_nlp.get_ultra_quality_status()
        
        # Performance metrics
        performance = status.get('performance', {})
        
        # Quality metrics
        quality = status.get('quality', {})
        
        # System metrics
        system = status.get('system', {})
        
        # Cache metrics
        cache = status.get('cache', {})
        
        # Memory metrics
        memory = status.get('memory', {})
        
        return {
            'performance': performance,
            'quality': quality,
            'system': system,
            'cache': cache,
            'memory': memory,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get ultra-quality metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra-quality metrics: {e}")

@router.get("/quality-trends", response_model=Dict[str, Any])
async def get_quality_trends():
    """Get quality trends and statistics."""
    try:
        # Get quality trends from the system
        quality_trends = ultra_quality_nlp.quality_tracker.get_quality_trends()
        confidence_trends = ultra_quality_nlp.confidence_tracker.get_confidence_trends()
        
        return {
            'quality_trends': quality_trends,
            'confidence_trends': confidence_trends,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get quality trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quality trends: {e}")

@router.get("/validation-report", response_model=Dict[str, Any])
async def get_validation_report():
    """Get ensemble and cross-validation report."""
    try:
        # Get validation statistics
        status = await ultra_quality_nlp.get_ultra_quality_status()
        
        # Calculate validation metrics
        quality_stats = status.get('quality', {})
        performance_stats = status.get('performance', {})
        
        validation_report = {
            'quality_metrics': {
                'average_quality_score': quality_stats.get('average_quality_score', 0),
                'average_confidence_score': quality_stats.get('average_confidence_score', 0),
                'quality_samples': quality_stats.get('quality_samples', 0),
                'confidence_samples': quality_stats.get('confidence_samples', 0)
            },
            'performance_metrics': {
                'success_rate': performance_stats.get('success_rate', 0),
                'cache_hit_rate': performance_stats.get('cache_hit_rate', 0),
                'average_processing_time': performance_stats.get('average_processing_time', 0)
            },
            'validation_status': {
                'ensemble_validation_enabled': ultra_quality_nlp.config.ensemble_validation,
                'cross_validation_enabled': ultra_quality_nlp.config.cross_validation,
                'quality_threshold': ultra_quality_nlp.config.quality_threshold,
                'confidence_threshold': ultra_quality_nlp.config.confidence_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return validation_report
        
    except Exception as e:
        logger.error(f"Failed to get validation report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation report: {e}")

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_ultra_quality_capabilities():
    """Get ultra-quality system capabilities."""
    return {
        "ultra_quality_mode": True,
        "comprehensive_analysis": True,
        "ensemble_methods": True,
        "cross_validation": True,
        "quality_assessment": True,
        "confidence_scoring": True,
        "analysis_tasks": [
            "sentiment_analysis",
            "entity_extraction",
            "keyword_extraction",
            "topic_modeling",
            "readability_analysis"
        ],
        "quality_features": [
            "ensemble_validation",
            "cross_validation",
            "confidence_scoring",
            "quality_assessment",
            "trend_analysis"
        ],
        "supported_languages": ["en", "es", "fr", "de"],
        "max_text_length": 50000,
        "max_batch_size": 100,
        "quality_threshold": 0.9,
        "confidence_threshold": 0.95,
        "ultra_quality_optimization": True
    }

# Utility endpoints
@router.get("/supported-languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages."""
    return ["en", "es", "fr", "de"]

@router.get("/analysis-tasks", response_model=List[str])
async def get_analysis_tasks():
    """Get available analysis tasks."""
    return ["sentiment", "entities", "keywords", "topics", "readability"]

@router.get("/quality-metrics", response_model=List[str])
async def get_quality_metrics():
    """Get available quality metrics."""
    return [
        "quality_score",
        "confidence_score",
        "ensemble_validation",
        "cross_validation",
        "sentiment_quality",
        "entity_quality",
        "keyword_quality",
        "topic_quality",
        "readability_quality"
    ]












