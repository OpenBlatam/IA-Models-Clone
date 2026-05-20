"""
Similarity Service - Text similarity detection
"""

import logging
from typing import Dict, Any, Optional

try:
    from schemas import SimilarityData
except ImportError:
    SimilarityData = Dict[str, Any]

try:
    from utils import (
        calculate_similarity, find_common_words,
        validate_content_length, create_timestamp
    )
except ImportError:
    logging.warning("utils module not available")
    def calculate_similarity(text1, text2): return 0.0
    def find_common_words(text1, text2): return []
    def validate_content_length(text): pass
    def create_timestamp(): return __import__("time").time()

try:
    from .decorators import with_caching, with_analytics, handle_errors
except ImportError:
    # Fallback if decorators not available
    def with_caching(*args, **kwargs):
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


def _cache_key_from_texts(text1: str, text2: str, threshold: float, **kwargs) -> str:
    """Generate cache key from texts and threshold"""
    return f"similarity:{text1[:50]}:{text2[:50]}:{threshold}"


@handle_errors
@with_analytics("similarity")
@with_caching(cache_key_func=_cache_key_from_texts)
def detect_similarity(
    text1: str,
    text2: str,
    threshold: float,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> SimilarityData:
    """
    Detect similarity between two texts with caching
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        threshold: Similarity threshold
        request_id: Optional request ID for tracking
        user_id: Optional user ID for tracking
        
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
    
    return result

