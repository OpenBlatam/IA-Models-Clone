"""
Domain Services
Business logic that doesn't naturally fit in entities
"""

import hashlib
from ..domain.entities import ContentAnalysis, SimilarityAnalysis, QualityAnalysis
from ..domain.value_objects import AnalysisResult
import time


async def analyze_content_domain(content: str) -> ContentAnalysis:
    """
    Domain service for content analysis
    Contains pure business logic
    """
    from utils import extract_words, calculate_content_hash, calculate_redundancy_score
    
    words = extract_words(content)
    content_hash = calculate_content_hash(content)
    word_count = len(words)
    character_count = len(content)
    unique_words = len(set(words))
    redundancy_score = calculate_redundancy_score(content)
    
    return ContentAnalysis(
        content=content,
        content_hash=content_hash,
        word_count=word_count,
        character_count=character_count,
        unique_words=unique_words,
        redundancy_score=redundancy_score,
        timestamp=time.time()
    )


async def detect_similarity_domain(text1: str, text2: str, threshold: float) -> SimilarityAnalysis:
    """Domain service for similarity detection"""
    from utils import calculate_similarity, find_common_words
    
    similarity_score = calculate_similarity(text1, text2)
    is_similar = similarity_score >= threshold
    common_words = find_common_words(text1, text2)
    
    return SimilarityAnalysis(
        text1=text1,
        text2=text2,
        similarity_score=similarity_score,
        is_similar=is_similar,
        common_words=common_words,
        threshold=threshold,
        timestamp=time.time()
    )


async def assess_quality_domain(content: str) -> QualityAnalysis:
    """Domain service for quality assessment"""
    from utils import calculate_readability_score, calculate_redundancy_score
    from utils import get_quality_rating, get_quality_suggestions
    
    readability_score = calculate_readability_score(content)
    complexity_score = calculate_redundancy_score(content) * 100
    quality_rating = get_quality_rating(readability_score)
    suggestions = get_quality_suggestions(content, readability_score)
    
    return QualityAnalysis(
        content=content,
        readability_score=readability_score,
        complexity_score=complexity_score,
        quality_rating=quality_rating,
        suggestions=suggestions,
        timestamp=time.time()
    )






