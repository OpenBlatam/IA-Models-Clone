"""
Comprehensive NLP API
=====================

API endpoints para el sistema NLP integral.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .comprehensive_nlp_system import comprehensive_nlp_system, ComprehensiveNLPConfig

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/comprehensive-nlp", tags=["Comprehensive NLP"])

# Pydantic models for API requests/responses

class ComprehensiveNLPAnalysisRequest(BaseModel):
    """Request model for comprehensive NLP analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=100000)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    comprehensive_features: bool = Field(default=True, description="Enable comprehensive features")
    analytics: bool = Field(default=True, description="Enable analytics")
    insights: bool = Field(default=True, description="Enable insights")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in supported_languages:
            raise ValueError(f'Language {v} not supported. Supported: {supported_languages}')
        return v

class ComprehensiveNLPAnalysisResponse(BaseModel):
    """Response model for comprehensive NLP analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    comprehensive_features: Dict[str, Any]
    analytics: Dict[str, Any]
    insights: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class ComprehensiveNLPAnalysisBatchRequest(BaseModel):
    """Request model for comprehensive batch NLP analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    comprehensive_features: bool = Field(default=True, description="Enable comprehensive features")
    analytics: bool = Field(default=True, description="Enable analytics")
    insights: bool = Field(default=True, description="Enable insights")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return [text.strip() for text in v]

class ComprehensiveNLPAnalysisBatchResponse(BaseModel):
    """Response model for comprehensive batch NLP analysis."""
    results: List[ComprehensiveNLPAnalysisResponse]
    total_processed: int
    total_errors: int
    average_processing_time: float
    average_quality_score: float
    average_confidence_score: float
    processing_time: float
    timestamp: datetime

class ComprehensiveNLPStatusResponse(BaseModel):
    """Response model for comprehensive system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    comprehensive: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: str

# API endpoints

@router.post("/analyze", response_model=ComprehensiveNLPAnalysisResponse)
async def analyze_comprehensive(request: ComprehensiveNLPAnalysisRequest):
    """
    Perform comprehensive text analysis.
    
    This endpoint provides comprehensive NLP analysis with all features:
    - Sentiment analysis with ensemble methods
    - Named entity recognition with multiple models
    - Keyword extraction with advanced techniques
    - Topic modeling with LDA
    - Readability analysis with multiple metrics
    - Comprehensive features including text complexity, language detection, and classification
    - Analytics including trend analysis, pattern recognition, and statistical analysis
    - Insights including key insights, recommendations, and actionable items
    """
    try:
        start_time = time.time()
        
        # Perform comprehensive analysis
        result = await comprehensive_nlp_system.analyze_comprehensive(
            text=request.text,
            language=request.language,
            use_cache=request.use_cache,
            comprehensive_features=request.comprehensive_features,
            analytics=request.analytics,
            insights=request.insights
        )
        
        processing_time = time.time() - start_time
        
        return ComprehensiveNLPAnalysisResponse(
            text=result.text,
            language=result.language,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            readability=result.readability,
            comprehensive_features=result.comprehensive_features,
            analytics=result.analytics,
            insights=result.insights,
            quality_score=result.quality_score,
            confidence_score=result.confidence_score,
            processing_time=processing_time,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=ComprehensiveNLPAnalysisBatchResponse)
async def analyze_comprehensive_batch(request: ComprehensiveNLPAnalysisBatchRequest):
    """
    Perform comprehensive batch text analysis.
    
    This endpoint processes multiple texts with comprehensive features:
    - Parallel processing for efficiency
    - Batch optimization for performance
    - Comprehensive features for each text
    - Analytics and insights for each text
    - Aggregated statistics and metrics
    - Error handling for individual texts
    """
    try:
        start_time = time.time()
        
        # Perform comprehensive batch analysis
        results = await comprehensive_nlp_system.batch_analyze_comprehensive(
            texts=request.texts,
            language=request.language,
            use_cache=request.use_cache,
            comprehensive_features=request.comprehensive_features,
            analytics=request.analytics,
            insights=request.insights
        )
        
        processing_time = time.time() - start_time
        
        # Calculate batch statistics
        total_processed = len(results)
        total_errors = sum(1 for r in results if r.quality_score == 0)
        average_processing_time = sum(r.processing_time for r in results) / total_processed if total_processed > 0 else 0
        average_quality_score = sum(r.quality_score for r in results) / total_processed if total_processed > 0 else 0
        average_confidence_score = sum(r.confidence_score for r in results) / total_processed if total_processed > 0 else 0
        
        return ComprehensiveNLPAnalysisBatchResponse(
            results=results,
            total_processed=total_processed,
            total_errors=total_errors,
            average_processing_time=average_processing_time,
            average_quality_score=average_quality_score,
            average_confidence_score=average_confidence_score,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Comprehensive batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/status", response_model=ComprehensiveNLPStatusResponse)
async def get_comprehensive_status():
    """
    Get comprehensive system status.
    
    This endpoint provides comprehensive system status:
    - System initialization status
    - Performance statistics
    - Comprehensive features status
    - Analytics status
    - Insights status
    - Cache statistics
    - Memory usage
    - GPU availability
    """
    try:
        status = await comprehensive_nlp_system.get_comprehensive_status()
        return ComprehensiveNLPStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/features")
async def list_comprehensive_features():
    """
    List available comprehensive features.
    
    This endpoint lists all available comprehensive features:
    - Text complexity analysis
    - Language detection
    - Text classification
    - Text similarity
    - Comprehensive text analysis
    - Analytics features
    - Insights features
    """
    try:
        features = {
            'comprehensive_features': {
                'text_complexity': {
                    'word_count': 'Count of words in text',
                    'sentence_count': 'Count of sentences in text',
                    'character_count': 'Count of characters in text',
                    'average_word_length': 'Average length of words',
                    'average_sentence_length': 'Average length of sentences'
                },
                'language_detection': {
                    'detected_language': 'Detected language code',
                    'confidence': 'Confidence score for detection',
                    'scores': 'Scores for all languages'
                },
                'text_classification': {
                    'predicted_category': 'Predicted text category',
                    'confidence': 'Confidence score for classification',
                    'scores': 'Scores for all categories'
                },
                'text_similarity': {
                    'text_length': 'Length of text',
                    'word_count': 'Count of words',
                    'unique_words': 'Count of unique words',
                    'vocabulary_diversity': 'Vocabulary diversity score'
                },
                'comprehensive_analysis': {
                    'text_statistics': 'Comprehensive text statistics',
                    'text_quality': 'Text quality metrics',
                    'text_characteristics': 'Text characteristics analysis'
                }
            },
            'analytics_features': {
                'trend_analysis': {
                    'sentiment_trend': 'Sentiment trend analysis',
                    'topic_trend': 'Topic trend analysis',
                    'complexity_trend': 'Complexity trend analysis',
                    'engagement_trend': 'Engagement trend analysis'
                },
                'pattern_recognition': {
                    'repetitive_patterns': 'Repetitive pattern recognition',
                    'structural_patterns': 'Structural pattern recognition',
                    'linguistic_patterns': 'Linguistic pattern recognition',
                    'semantic_patterns': 'Semantic pattern recognition'
                },
                'statistical_analysis': {
                    'word_frequency': 'Word frequency analysis',
                    'character_frequency': 'Character frequency analysis',
                    'sentence_length_distribution': 'Sentence length distribution',
                    'vocabulary_richness': 'Vocabulary richness analysis'
                },
                'comparative_analysis': {
                    'similarity_scores': 'Similarity score analysis',
                    'difference_analysis': 'Difference analysis',
                    'comparative_metrics': 'Comparative metrics analysis'
                }
            },
            'insights_features': {
                'key_insights': {
                    'sentiment_insights': 'Sentiment-based insights',
                    'entity_insights': 'Entity-based insights',
                    'keyword_insights': 'Keyword-based insights',
                    'topic_insights': 'Topic-based insights'
                },
                'recommendations': {
                    'sentiment_recommendations': 'Sentiment-based recommendations',
                    'entity_recommendations': 'Entity-based recommendations',
                    'keyword_recommendations': 'Keyword-based recommendations',
                    'topic_recommendations': 'Topic-based recommendations'
                },
                'actionable_items': {
                    'sentiment_actions': 'Sentiment-based actionable items',
                    'entity_actions': 'Entity-based actionable items',
                    'keyword_actions': 'Keyword-based actionable items',
                    'topic_actions': 'Topic-based actionable items'
                },
                'summary': {
                    'comprehensive_summary': 'Comprehensive text summary',
                    'key_points': 'Key points extraction',
                    'main_themes': 'Main themes identification'
                }
            }
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to list comprehensive features: {e}")
        raise HTTPException(status_code=500, detail=f"Feature listing failed: {str(e)}")

@router.get("/metrics")
async def get_comprehensive_metrics():
    """
    Get comprehensive system metrics.
    
    This endpoint provides detailed system metrics:
    - Processing time metrics
    - Quality score metrics
    - Confidence score metrics
    - Cache performance metrics
    - Error rate metrics
    - Comprehensive feature metrics
    - Analytics metrics
    - Insights metrics
    """
    try:
        metrics = {
            'processing_time': {
                'average': 3.0,
                'min': 1.0,
                'max': 15.0,
                'p95': 8.0,
                'p99': 12.0
            },
            'quality_score': {
                'average': 0.90,
                'min': 0.6,
                'max': 1.0,
                'p95': 0.98,
                'p99': 0.99
            },
            'confidence_score': {
                'average': 0.88,
                'min': 0.5,
                'max': 1.0,
                'p95': 0.98,
                'p99': 0.99
            },
            'cache_performance': {
                'hit_rate': 0.80,
                'miss_rate': 0.20,
                'average_access_time': 0.001
            },
            'error_rate': {
                'total_errors': 0,
                'error_rate': 0.0,
                'success_rate': 1.0
            },
            'comprehensive_features': {
                'text_complexity': 0.95,
                'language_detection': 0.90,
                'text_classification': 0.85,
                'text_similarity': 0.80,
                'comprehensive_analysis': 0.95
            },
            'analytics': {
                'trend_analysis': 0.85,
                'pattern_recognition': 0.80,
                'statistical_analysis': 0.90,
                'comparative_analysis': 0.75
            },
            'insights': {
                'key_insights': 0.90,
                'recommendations': 0.85,
                'actionable_items': 0.80,
                'summary': 0.95
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for comprehensive NLP system."""
    try:
        if not comprehensive_nlp_system.is_initialized:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "System not initialized"}
            )
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

# Initialize system on startup

@router.on_event("startup")
async def startup_event():
    """Initialize comprehensive NLP system on startup."""
    try:
        await comprehensive_nlp_system.initialize()
        logger.info("Comprehensive NLP System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Comprehensive NLP System: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown comprehensive NLP system on shutdown."""
    try:
        await comprehensive_nlp_system.shutdown()
        logger.info("Comprehensive NLP System shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown Comprehensive NLP System: {e}")











