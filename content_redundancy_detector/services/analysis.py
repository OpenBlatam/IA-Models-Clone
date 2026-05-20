"""
Analysis Service - Content analysis and redundancy detection
"""

import logging
from typing import Dict, Any, Optional

try:
    from schemas import AnalysisData
except ImportError:
    AnalysisData = Dict[str, Any]

try:
    from utils import (
        extract_words, calculate_content_hash, calculate_redundancy_score,
        validate_content_length, create_timestamp
    )
except ImportError:
    logging.warning("utils module not available")
    def extract_words(text): return []
    def calculate_content_hash(text): return ""
    def calculate_redundancy_score(text): return 0.0
    def validate_content_length(text): pass
    def create_timestamp(): return __import__("time").time()

try:
    from .decorators import with_caching, with_webhooks, with_analytics, handle_errors
except ImportError:
    # Fallback if decorators not available
    def with_caching(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def with_webhooks(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def with_analytics(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def handle_errors(func):
        return func

logger = logging.getLogger(__name__)


def _cache_key_from_content(content: str, **kwargs) -> str:
    """Generate cache key from content"""
    return f"analysis:{content[:100]}"


@handle_errors
@with_analytics("content")
@with_webhooks("ANALYSIS_COMPLETED")
@with_caching(cache_key_func=_cache_key_from_content)
def analyze_content(
    content: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> AnalysisData:
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
    
    return result

