"""
AI/ML Service - Advanced AI/ML operations
"""

import asyncio
import logging
from typing import Dict, Any, List

try:
    from utils import validate_content_length, create_timestamp
except ImportError:
    logging.warning("utils module not available")
    def validate_content_length(text): pass
    def create_timestamp(): return __import__("time").time()

try:
    from analytics import record_analysis
except ImportError:
    def record_analysis(*args, **kwargs): pass

# Try new refactored ML engine first, fallback to old one
try:
    from ml.engine import ai_ml_engine
    logger = logging.getLogger(__name__)
    logger.info("Using refactored ML engine")
except ImportError:
    try:
        from ai_ml_enhanced import ai_ml_engine
        logger = logging.getLogger(__name__)
        logger.warning("Using legacy ai_ml_enhanced module")
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("ai_ml_enhanced module not available")
        ai_ml_engine = None

logger = logging.getLogger(__name__)


def _check_ai_ml_engine() -> None:
    """Helper to check if AI/ML engine is available"""
    if not ai_ml_engine:
        raise RuntimeError("AI/ML engine not available. Please check configuration and dependencies.")


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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
    try:
        result = await ai_ml_engine.calculate_semantic_similarity(text1, text2)
        record_analysis("semantic_similarity", result)
        return result
    except Exception as e:
        logger.error(f"Error in semantic similarity calculation: {e}")
        raise


async def detect_plagiarism(
    content: str,
    reference_texts: List[str],
    threshold: float = 0.8
) -> Dict[str, Any]:
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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
    
    _check_ai_ml_engine()
    
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






