"""
Utility functions for content analysis
"""

import hashlib
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from difflib import SequenceMatcher
import logging

from types import AnalysisData, SimilarityData, QualityData

logger = logging.getLogger(__name__)


def extract_words(text: str) -> List[str]:
    """Extract words from text using regex"""
    if not text:
        return []
    
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    except Exception as e:
        logger.error(f"Error extracting words: {e}")
        return []


def calculate_content_hash(content: str) -> str:
    """Calculate MD5 hash of content"""
    if not content:
        return ""
    
    try:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash: {e}")
        return ""


def calculate_redundancy_score(content: str) -> float:
    """Calculate redundancy score based on word repetition"""
    words = extract_words(content)
    if not words:
        return 0.0
    
    try:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        
        if total_words == 0:
            return 0.0
        
        redundancy = 1.0 - (unique_words / total_words)
        return min(redundancy, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating redundancy score: {e}")
        return 0.0


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    
    try:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def find_common_words(text1: str, text2: str) -> List[str]:
    """Find common words between two texts"""
    if not text1 or not text2:
        return []
    
    try:
        words1 = set(extract_words(text1))
        words2 = set(extract_words(text2))
        return list(words1.intersection(words2))
    except Exception as e:
        logger.error(f"Error finding common words: {e}")
        return []


def calculate_readability_score(content: str) -> float:
    """Calculate readability score using simplified Flesch formula"""
    if not content:
        return 0.0
    
    try:
        sentences = re.split(r'[.!?]+', content)
        words = extract_words(content)
        
        if not sentences or not words:
            return 0.0
        
        # Filter empty sentences
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, readability))
        
    except Exception as e:
        logger.error(f"Error calculating readability score: {e}")
        return 0.0


def get_quality_rating(score: float) -> str:
    """Get quality rating based on readability score"""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Needs Improvement"


def get_quality_suggestions(content: str, score: float) -> List[str]:
    """Get quality improvement suggestions based on score"""
    suggestions = []
    
    if score < 40:
        suggestions.extend([
            "Consider using shorter sentences",
            "Use simpler and clearer words",
            "Improve text structure and organization",
            "Avoid excessive word repetition"
        ])
    elif score < 60:
        suggestions.extend([
            "Reduce average sentence length",
            "Use clearer and more direct vocabulary",
            "Improve text flow"
        ])
    elif score < 80:
        suggestions.append("Text is good, consider small clarity improvements")
    else:
        suggestions.append("Text has excellent quality and readability")
    
    return suggestions


def validate_content_length(content: str, max_length: int = 10000, min_length: int = 10) -> None:
    """Validate content length with guard clause pattern"""
    if len(content) > max_length:
        raise ValueError(f"Content too long. Maximum: {max_length} characters")
    
    if len(content) < min_length:
        raise ValueError(f"Content too short. Minimum: {min_length} characters")


def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    # Remove control characters and normalize spaces
    text = " ".join(text.split())
    return text.strip()


def create_timestamp() -> str:
    """Create ISO timestamp"""
    return datetime.utcnow().isoformat()


def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Add execution time to result if it's a dict
        if isinstance(result, dict):
            result['analysis_time'] = execution_time
        
        return result
    return wrapper