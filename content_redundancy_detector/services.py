"""
Enhanced Service functions for advanced content analysis
- Improved error handling with detailed validation
- Better type hints for IDE support
- Async/await patterns for non-blocking operations
- Webhook integration for real-time notifications
- Comprehensive logging and metrics
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import traceback

try:
    from types import AnalysisData, SimilarityData, QualityData
except ImportError:
    # Fallback type definitions if types module not available
    AnalysisData = Dict[str, Any]
    SimilarityData = Dict[str, Any]
    QualityData = Dict[str, Any]

try:
    from utils import (
        extract_words, calculate_content_hash, calculate_redundancy_score,
        calculate_similarity, find_common_words, calculate_readability_score,
        get_quality_rating, get_quality_suggestions, validate_content_length,
        create_timestamp, measure_time
    )
except ImportError:
    logger.warning("utils module not available, some functions may not work")

try:
    from cache import (
        get_cached_analysis_result, cache_analysis_result,
        get_cached_similarity_result, cache_similarity_result,
        get_cached_quality_result, cache_quality_result
    )
except ImportError:
    logger.warning("cache module not available, caching disabled")
    def get_cached_analysis_result(*args, **kwargs):
        return None
    def cache_analysis_result(*args, **kwargs):
        pass
    def get_cached_similarity_result(*args, **kwargs):
        return None
    def cache_similarity_result(*args, **kwargs):
        pass
    def get_cached_quality_result(*args, **kwargs):
        return None
    def cache_quality_result(*args, **kwargs):
        pass

try:
    from analytics import record_analysis
except ImportError:
    logger.warning("analytics module not available")
    def record_analysis(*args, **kwargs):
        pass

try:
    from ai_ml_enhanced import ai_ml_engine
except ImportError:
    logger.warning("ai_ml_enhanced module not available")
    ai_ml_engine = None

try:
    from config import settings
except ImportError:
    logger.warning("config module not available")
    class Settings:
        app_version = "2.0.0"
    settings = Settings()

try:
    from webhooks import send_webhook, WebhookEvent
except ImportError:
    try:
        # Try importing with relative path
        import sys
        import os
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        from webhooks import send_webhook, WebhookEvent
    except ImportError:
        logger.warning("webhooks module not available, using fallback")
        async def send_webhook(*args, **kwargs):
            """Fallback webhook sender - no-op"""
            return {"status": "disabled"}
        class WebhookEvent:
            """Fallback WebhookEvent"""
            ANALYSIS_COMPLETED = "analysis_completed"
            SIMILARITY_COMPLETED = "similarity_completed"
            QUALITY_COMPLETED = "quality_completed"
            BATCH_COMPLETED = "batch_completed"
            BATCH_FAILED = "batch_failed"
            SYSTEM_ERROR = "system_error"

logger = logging.getLogger(__name__)

# Decorator fallback
try:
    from functools import wraps
    def measure_time(func):
        """Fallback decorator if measure_time not available"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
except ImportError:
    def measure_time(func):
        return func


def analyze_content(content: str, request_id: Optional[str] = None, user_id: Optional[str] = None) -> AnalysisData:
    """
    Analyze content for redundancy and basic metrics with caching
    
    Args:
        content: Text content to analyze
        request_id: Optional request ID for tracking
        user_id: Optional user ID for tracking
        
    Returns:
        Dict containing analysis results
        
    Raises:
        ValueError: If content validation fails
        TypeError: If content is not a string
    """
    # Enhanced validation
    if not isinstance(content, str):
        raise TypeError(f"Content must be a string, got {type(content).__name__}")
    
    if not content or not content.strip():
        raise ValueError("Content cannot be empty or whitespace only")
    
    try:
        validate_content_length(content)
    except (ImportError, NameError) as e:
        logger.warning(f"Content length validation not available: {e}")
    except Exception as e:
        raise ValueError(f"Content validation failed: {str(e)}")
    
    # Check cache first
    try:
        cached_result = get_cached_analysis_result(content)
        if cached_result is not None:
            logger.debug(f"Returning cached analysis result for request {request_id}")
            # Fire webhook for cached result asynchronously
            asyncio.create_task(send_webhook(
                WebhookEvent.ANALYSIS_COMPLETED,
                {"cached": True, "result": cached_result},
                request_id=request_id,
                user_id=user_id
            ))
            return cached_result
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")
    
    try:
        # Extract words and calculate metrics
        words = extract_words(content)
        content_hash = calculate_content_hash(content)
        word_count = len(words)
        character_count = len(content)
        unique_words = len(set(words))
        redundancy_score = calculate_redundancy_score(content)
        
        result: AnalysisData = {
            "content_hash": content_hash,
            "word_count": word_count,
            "character_count": character_count,
            "unique_words": unique_words,
            "redundancy_score": redundancy_score,
            "timestamp": create_timestamp(),
            "request_id": request_id,
            "user_id": user_id
        }
        
        # Cache the result
        try:
            cache_analysis_result(content, result)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        # Record for analytics
        try:
            record_analysis("content", result)
        except Exception as e:
            logger.warning(f"Failed to record analytics: {e}")
        
        # Fire webhook asynchronously
        try:
            asyncio.create_task(send_webhook(
                WebhookEvent.ANALYSIS_COMPLETED,
                {"result": result},
                request_id=request_id,
                user_id=user_id
            ))
        except Exception as e:
            logger.warning(f"Failed to send webhook: {e}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing content: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Fire error webhook
        try:
            asyncio.create_task(send_webhook(
                WebhookEvent.SYSTEM_ERROR,
                {"error": error_msg, "operation": "analyze_content"},
                request_id=request_id,
                user_id=user_id
            ))
        except Exception:
            pass
        
        raise ValueError(error_msg) from e


@measure_time
def detect_similarity(text1: str, text2: str, threshold: float, request_id: Optional[str] = None, user_id: Optional[str] = None) -> SimilarityData:
    """
    Detect similarity between two texts with caching
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        threshold: Similarity threshold
        
    Returns:
        Dict containing similarity results
        
    Raises:
        ValueError: If text validation fails
    """
    # Guard clauses for text validation
    if not text1 or not text2:
        raise ValueError("Both texts are required")
    
    validate_content_length(text1)
    validate_content_length(text2)
    
    # Check cache first
    cached_result = get_cached_similarity_result(text1, text2, threshold)
    if cached_result is not None:
        logger.debug("Returning cached similarity result")
        return cached_result
    
    try:
        # Calculate similarity metrics
        similarity_score = calculate_similarity(text1, text2)
        is_similar = similarity_score >= threshold
        common_words = find_common_words(text1, text2)
        
        result = {
            "similarity_score": similarity_score,
            "is_similar": is_similar,
            "common_words": common_words,
            "timestamp": create_timestamp()
        }
        
        # Cache the result
        cache_similarity_result(text1, text2, threshold, result)
        
        # Record for analytics
        record_analysis("similarity", result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting similarity: {e}")
        raise


@measure_time
def assess_quality(content: str) -> QualityData:
    """
    Assess content quality and readability with caching
    
    Args:
        content: Text content to assess
        
    Returns:
        Dict containing quality assessment results
        
    Raises:
        ValueError: If content validation fails
    """
    # Guard clause for content validation
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    # Check cache first
    cached_result = get_cached_quality_result(content)
    if cached_result is not None:
        logger.debug("Returning cached quality result")
        return cached_result
    
    try:
        # Calculate quality metrics
        readability_score = calculate_readability_score(content)
        complexity_score = calculate_redundancy_score(content) * 100
        quality_rating = get_quality_rating(readability_score)
        suggestions = get_quality_suggestions(content, readability_score)
        
        result = {
            "readability_score": readability_score,
            "complexity_score": complexity_score,
            "quality_rating": quality_rating,
            "suggestions": suggestions,
            "timestamp": create_timestamp()
        }
        
        # Cache the result
        cache_quality_result(content, result)
        
        # Record for analytics
        record_analysis("quality", result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error assessing quality: {e}")
        raise


def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics
    
    Returns:
        Dict containing system statistics
    """
    return {
        "total_endpoints": 6,
        "features": [
            "Content redundancy analysis",
            "Text similarity comparison",
            "Content quality assessment",
            "Health check",
            "System statistics",
            "Error handling"
        ],
        "version": "1.0.0",
        "status": "active"
    }


def get_health_status() -> Dict[str, Any]:
    """
    Get system health status
    
    Returns:
        Dict containing health status
    """
    return {
        "status": "healthy",
        "timestamp": create_timestamp(),
        "version": settings.app_version,
        "ai_ml_initialized": ai_ml_engine.initialized
    }


# Enhanced AI/ML Service Functions

async def analyze_sentiment(content: str) -> Dict[str, Any]:
    """
    Analyze sentiment of content using AI/ML
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict containing sentiment analysis results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.analyze_sentiment(content)
        record_analysis("sentiment", result)
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise


async def detect_language(content: str) -> Dict[str, Any]:
    """
    Detect language of content
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict containing language detection results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.detect_language(content)
        record_analysis("language", result)
        return result
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise


async def extract_topics(texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
    """
    Extract topics from a collection of texts
    
    Args:
        texts: List of texts to analyze
        num_topics: Number of topics to extract
        
    Returns:
        Dict containing topic extraction results
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    for text in texts:
        validate_content_length(text)
    
    try:
        result = await ai_ml_engine.extract_topics(texts, num_topics)
        record_analysis("topics", result)
        return result
    except Exception as e:
        logger.error(f"Error in topic extraction: {e}")
        raise


async def calculate_semantic_similarity(text1: str, text2: str) -> Dict[str, Any]:
    """
    Calculate semantic similarity between two texts using AI/ML
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Dict containing semantic similarity results
    """
    if not text1 or not text2:
        raise ValueError("Both texts are required")
    
    validate_content_length(text1)
    validate_content_length(text2)
    
    try:
        result = await ai_ml_engine.calculate_semantic_similarity(text1, text2)
        record_analysis("semantic_similarity", result)
        return result
    except Exception as e:
        logger.error(f"Error in semantic similarity calculation: {e}")
        raise


async def detect_plagiarism(content: str, reference_texts: List[str], threshold: float = 0.8) -> Dict[str, Any]:
    """
    Detect potential plagiarism in content
    
    Args:
        content: Text content to check for plagiarism
        reference_texts: List of reference texts to compare against
        threshold: Similarity threshold for plagiarism detection
        
    Returns:
        Dict containing plagiarism detection results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    if not reference_texts:
        raise ValueError("Reference texts are required")
    
    validate_content_length(content)
    for ref_text in reference_texts:
        validate_content_length(ref_text)
    
    try:
        result = await ai_ml_engine.detect_plagiarism(content, reference_texts, threshold)
        record_analysis("plagiarism", result)
        return result
    except Exception as e:
        logger.error(f"Error in plagiarism detection: {e}")
        raise


async def extract_entities(content: str) -> Dict[str, Any]:
    """
    Extract named entities from content
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict containing entity extraction results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.extract_entities(content)
        record_analysis("entities", result)
        return result
    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")
        raise


async def generate_summary(content: str, max_length: int = 150) -> Dict[str, Any]:
    """
    Generate summary of content using AI/ML
    
    Args:
        content: Text content to summarize
        max_length: Maximum length of summary
        
    Returns:
        Dict containing summary results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.generate_summary(content, max_length)
        record_analysis("summary", result)
        return result
    except Exception as e:
        logger.error(f"Error in text summarization: {e}")
        raise


async def analyze_readability_advanced(content: str) -> Dict[str, Any]:
    """
    Advanced readability analysis using AI/ML
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict containing advanced readability analysis results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.analyze_readability(content)
        record_analysis("readability_advanced", result)
        return result
    except Exception as e:
        logger.error(f"Error in advanced readability analysis: {e}")
        raise


async def comprehensive_analysis(content: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis combining all AI/ML features
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dict containing comprehensive analysis results
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    validate_content_length(content)
    
    try:
        result = await ai_ml_engine.comprehensive_analysis(content)
        record_analysis("comprehensive", result)
        return result
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise


async def batch_analyze_content(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple texts in batch for efficiency
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        List of analysis results
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    for text in texts:
        validate_content_length(text)
    
    try:
        # Process texts in parallel
        tasks = [comprehensive_analysis(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "text_index": i,
                    "timestamp": create_timestamp()
                })
            else:
                processed_results.append(result)
        
        record_analysis("batch", {"count": len(texts), "results": len(processed_results)})
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise