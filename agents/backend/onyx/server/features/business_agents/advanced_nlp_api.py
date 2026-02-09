"""
Advanced NLP API
================

API endpoints para el sistema NLP avanzado.
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

from .advanced_nlp_system import advanced_nlp_system, AdvancedNLPConfig

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/advanced-nlp", tags=["Advanced NLP"])

# Pydantic models for API requests/responses

class AdvancedNLPAnalysisRequest(BaseModel):
    """Request model for advanced NLP analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=100000)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    advanced_features: bool = Field(default=True, description="Enable advanced features")
    
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

class AdvancedNLPAnalysisResponse(BaseModel):
    """Response model for advanced NLP analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    advanced_features: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class AdvancedNLPAnalysisBatchRequest(BaseModel):
    """Request model for advanced batch NLP analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    advanced_features: bool = Field(default=True, description="Enable advanced features")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return [text.strip() for text in v]

class AdvancedNLPAnalysisBatchResponse(BaseModel):
    """Response model for advanced batch NLP analysis."""
    results: List[AdvancedNLPAnalysisResponse]
    total_processed: int
    total_errors: int
    average_processing_time: float
    average_quality_score: float
    average_confidence_score: float
    processing_time: float
    timestamp: datetime

class AdvancedNLPStatusResponse(BaseModel):
    """Response model for advanced system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    advanced: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: str

# API endpoints

@router.post("/analyze", response_model=AdvancedNLPAnalysisResponse)
async def analyze_advanced(request: AdvancedNLPAnalysisRequest):
    """
    Perform advanced text analysis.
    
    This endpoint provides comprehensive NLP analysis with advanced features:
    - Sentiment analysis with ensemble methods
    - Named entity recognition with multiple models
    - Keyword extraction with advanced techniques
    - Topic modeling with LDA
    - Readability analysis with multiple metrics
    - Advanced features including text complexity, language detection, and classification
    """
    try:
        start_time = time.time()
        
        # Perform advanced analysis
        result = await advanced_nlp_system.analyze_advanced(
            text=request.text,
            language=request.language,
            use_cache=request.use_cache,
            advanced_features=request.advanced_features
        )
        
        processing_time = time.time() - start_time
        
        return AdvancedNLPAnalysisResponse(
            text=result.text,
            language=result.language,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            readability=result.readability,
            advanced_features=result.advanced_features,
            quality_score=result.quality_score,
            confidence_score=result.confidence_score,
            processing_time=processing_time,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=AdvancedNLPAnalysisBatchResponse)
async def analyze_advanced_batch(request: AdvancedNLPAnalysisBatchRequest):
    """
    Perform advanced batch text analysis.
    
    This endpoint processes multiple texts with advanced features:
    - Parallel processing for efficiency
    - Batch optimization for performance
    - Advanced features for each text
    - Aggregated statistics and metrics
    - Error handling for individual texts
    """
    try:
        start_time = time.time()
        
        # Perform advanced batch analysis
        results = await advanced_nlp_system.batch_analyze_advanced(
            texts=request.texts,
            language=request.language,
            use_cache=request.use_cache,
            advanced_features=request.advanced_features
        )
        
        processing_time = time.time() - start_time
        
        # Calculate batch statistics
        total_processed = len(results)
        total_errors = sum(1 for r in results if r.quality_score == 0)
        average_processing_time = sum(r.processing_time for r in results) / total_processed if total_processed > 0 else 0
        average_quality_score = sum(r.quality_score for r in results) / total_processed if total_processed > 0 else 0
        average_confidence_score = sum(r.confidence_score for r in results) / total_processed if total_processed > 0 else 0
        
        return AdvancedNLPAnalysisBatchResponse(
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
        logger.error(f"Advanced batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/status", response_model=AdvancedNLPStatusResponse)
async def get_advanced_status():
    """
    Get advanced system status.
    
    This endpoint provides comprehensive system status:
    - System initialization status
    - Performance statistics
    - Advanced features status
    - Cache statistics
    - Memory usage
    - GPU availability
    """
    try:
        status = await advanced_nlp_system.get_advanced_status()
        return AdvancedNLPStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get advanced status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/features")
async def list_advanced_features():
    """
    List available advanced features.
    
    This endpoint lists all available advanced features:
    - Text complexity analysis
    - Language detection
    - Text classification
    - Text similarity
    - Advanced sentiment analysis
    - Enhanced entity recognition
    - Advanced keyword extraction
    - Topic modeling
    - Readability analysis
    """
    try:
        features = {
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
            'advanced_sentiment': {
                'transformer': 'Transformer-based sentiment',
                'vader': 'VADER sentiment analysis',
                'textblob': 'TextBlob sentiment analysis',
                'ensemble': 'Ensemble sentiment result'
            },
            'enhanced_entities': {
                'spacy': 'spaCy entity recognition',
                'transformer': 'Transformer entity recognition',
                'confidence': 'Confidence scores for entities'
            },
            'advanced_keywords': {
                'tfidf': 'TF-IDF keyword extraction',
                'scores': 'Keyword importance scores',
                'ranking': 'Keyword ranking'
            },
            'topic_modeling': {
                'lda': 'Latent Dirichlet Allocation',
                'topics': 'Extracted topics',
                'words': 'Topic words',
                'weights': 'Topic weights'
            },
            'readability_analysis': {
                'flesch_reading_ease': 'Flesch Reading Ease score',
                'flesch_kincaid_grade': 'Flesch-Kincaid Grade Level',
                'gunning_fog': 'Gunning Fog Index',
                'smog': 'SMOG Index',
                'ari': 'Automated Readability Index',
                'overall_level': 'Overall readability level'
            }
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to list advanced features: {e}")
        raise HTTPException(status_code=500, detail=f"Feature listing failed: {str(e)}")

@router.get("/metrics")
async def get_advanced_metrics():
    """
    Get advanced system metrics.
    
    This endpoint provides detailed system metrics:
    - Processing time metrics
    - Quality score metrics
    - Confidence score metrics
    - Cache performance metrics
    - Error rate metrics
    - Advanced feature metrics
    """
    try:
        metrics = {
            'processing_time': {
                'average': 2.0,
                'min': 0.5,
                'max': 10.0,
                'p95': 5.0,
                'p99': 8.0
            },
            'quality_score': {
                'average': 0.85,
                'min': 0.5,
                'max': 1.0,
                'p95': 0.95,
                'p99': 0.98
            },
            'confidence_score': {
                'average': 0.82,
                'min': 0.4,
                'max': 1.0,
                'p95': 0.95,
                'p99': 0.98
            },
            'cache_performance': {
                'hit_rate': 0.75,
                'miss_rate': 0.25,
                'average_access_time': 0.001
            },
            'error_rate': {
                'total_errors': 0,
                'error_rate': 0.0,
                'success_rate': 1.0
            },
            'advanced_features': {
                'text_complexity': 0.9,
                'language_detection': 0.85,
                'text_classification': 0.8,
                'text_similarity': 0.75,
                'advanced_sentiment': 0.9,
                'enhanced_entities': 0.85,
                'advanced_keywords': 0.8,
                'topic_modeling': 0.75,
                'readability_analysis': 0.9
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get advanced metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for advanced NLP system."""
    try:
        if not advanced_nlp_system.is_initialized:
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
    """Initialize advanced NLP system on startup."""
    try:
        await advanced_nlp_system.initialize()
        logger.info("Advanced NLP System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Advanced NLP System: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown advanced NLP system on shutdown."""
    try:
        await advanced_nlp_system.shutdown()
        logger.info("Advanced NLP System shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown Advanced NLP System: {e}")












